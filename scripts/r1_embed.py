#!/usr/bin/env python3
"""R1 (R-150 item 6, user-approved 2026-10-28; research REGRID_SPEC §2): exact embedding of a restart into a box
twice as wide, with no interpolation.

The new mesh has width 2W, the same number of root blocks per direction (so root cells x2 coarser) and
num_levels + 1. The old box sits centred on new root-block boundaries (offset o = n_root/4 root blocks; needs
n_root divisible by 4), so every old block keeps its physical size and position one level deeper:
    (level L, lx) -> (L + 1, lx + o 2^(L + 1 - root_level))
and its data record (u0, b0, u_adm, the restart_state block; data_size bytes, ghosts included) is copied
bitwise. The new root blocks outside the old box are leaves (refined further only where 2:1 balance needs it),
written as ZERO records: the code's load hook (<job> restart_embed_fill, restart_state.cpp) fills them with the
atmosphere and the puncture background and re-solves the metric (the regrid path), and books the added mass in
MCLOSE as regrid_embed. The block list is written in the tree's Z-order (MeshBlockTree::CreateZOrderedLLList,
meshblock_tree.cpp:336-350: children x fastest, depth first), which the reader rebuilds (build_tree.cpp:441).

Restart layout (src/outputs/restart.cpp:276-362; dyn_grmhd + CFC, no z4c/turbulence): the parameter text through
'<par_end>\\n'; nmb_total, root_level (int32); RegionSize (9 doubles: x1min x2min x3min x1max x2max x3max dx1 dx2
dx3); mesh and MeshBlock RegionIndcs (19 int32 each); time, dt (double); ncycle (int32); nmb x LogicalLocation
(lx1 lx2 lx3 level, int32); nmb costs (float32); data_size (uint64); nmb records of data_size bytes.

Parameters: <mesh> x?min/x?max doubled; <mesh_refinement> num_levels, every <refined_region*> level and the
level-indexed <problem> keys (amr_level_bh, _bh_fine, _core, _debris, _floor, _stream, amr_pin_level; the full
list read by the source, dyngr_tov.cpp / build_tree.cpp:99) + 1; length keys unchanged (physical); <job>
restart_embed_fill = true, restart_embed_box = the old box. The same edits go into the -i deck (--deck), since a
restart's -i deck overrides the parameter copy.

Checks (REGRID_SPEC §2 validation (i), (ii)): every old block's record is byte-identical in the output (sha256
per block); the new tree is Z-sorted, tiles the box exactly (volume), has no overlaps and is 2:1 balanced
across faces, edges and corners.
    r1_embed.py IN.rst OUT.rst [--deck IN.par --deck-out OUT.par] [--manifest OUT.json]
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys

import numpy as np

LEVEL_KEYS = ('amr_level_bh', 'amr_level_bh_fine', 'amr_level_core', 'amr_level_debris', 'amr_level_floor',
              'amr_level_stream', 'amr_pin_level')


def read(fn):
    with open(fn, 'rb') as f:
        head = f.read(1 << 22)
        k = head.index(b'<par_end>') + len('<par_end>')
        if head[k:k + 1] == b'\n':   # exactly the one newline the writer emits (parameter_input.cpp:792);
            k += 1                      # the int32 nmb_total that follows can itself start with 0x0a or 0x0d
        f.seek(k)
        nmb, root = struct.unpack('2i', f.read(8))
        size = list(struct.unpack('9d', f.read(72)))
        indcs = f.read(2 * 19 * 4)
        time, dt = struct.unpack('2d', f.read(16))
        ncycle, = struct.unpack('i', f.read(4))
        ll = np.frombuffer(f.read(nmb * 16), dtype=np.int32).reshape(nmb, 4).copy()
        cost = np.frombuffer(f.read(nmb * 4), dtype=np.float32).copy()
        dsize, = struct.unpack('Q', f.read(8))
        off = f.tell()
    if os.path.getsize(fn) != off + nmb * dsize:
        sys.exit(f'{fn}: size {os.path.getsize(fn)} != header + {nmb} x {dsize} (unsupported layout)')
    mi = np.frombuffer(indcs[:76], dtype=np.int32)
    return dict(par=head[:k], nmb=nmb, root=root, size=size, indcs=indcs, mesh_indcs=mi, time=time, dt=dt,
                ncycle=ncycle, ll=ll, cost=cost, dsize=dsize, off=off)


def zkey(lx, ly, lz, lev, lmax):
    s = lmax - lev
    x, y, z = lx << s, ly << s, lz << s
    k = 0
    for b in range(lmax + 1):
        k |= (((x >> b) & 1) << (3 * b)) | (((y >> b) & 1) << (3 * b + 1)) | (((z >> b) & 1) << (3 * b + 2))
    return k


def edit_par(text, old, new_box, deck=False):
    """the parameter edits; text is the PAR_DUMP (or an input deck)"""
    out, block = [], None
    seen = set()
    for line in text.split('\n'):
        m = re.match(r'^\s*<([^>]+)>', line)
        if m:
            block = m.group(1).strip()
            out.append(line)
            if block == 'job':
                out.append(f'restart_embed_fill = true   # R1 (r1_embed.py): fill the new outer blocks at load')
                out.append('restart_embed_box = ' + ','.join(repr(v) for v in old)
                           + '   # R1: the embedded (old) box x1min,x1max,x2min,x2max,x3min,x3max')
                seen.add('job')
            continue
        kv = re.match(r'^(\s*)([A-Za-z0-9_]+)(\s*=\s*)([^#\s]+)(.*)$', line)
        if kv and block is not None:
            ind, key, eq, val, rest = kv.groups()
            nv = None
            if block == 'mesh' and key in ('x1min', 'x1max', 'x2min', 'x2max', 'x3min', 'x3max'):
                nv = repr(new_box[('x1min', 'x1max', 'x2min', 'x2max', 'x3min', 'x3max').index(key)])
            elif block == 'mesh_refinement' and key == 'num_levels':
                nv = str(int(val) + 1)
            elif block.startswith('refined_region') and key == 'level':
                nv = str(int(val) + 1)
            elif block == 'problem' and key in LEVEL_KEYS:
                nv = str(int(val) + 1)
            elif block == 'job' and key in ('restart_embed_fill', 'restart_embed_box'):
                continue                      # an earlier embedding's keys (replaced above)
            if nv is not None:
                line = f'{ind}{key}{eq}{nv}{rest}'
                seen.add((block, key))
        out.append(line)
    if 'job' not in seen:
        sys.exit('no <job> block')
    return '\n'.join(out), seen


def neighbours_ok(leaves, lmax):
    """2:1 balance across faces/edges/corners: for every leaf, the leaves touching it differ by <= 1 level"""
    # integer boxes at the finest level
    lo = np.array([[b[0] << (lmax - b[3]), b[1] << (lmax - b[3]), b[2] << (lmax - b[3])] for b in leaves])
    w = np.array([1 << (lmax - b[3]) for b in leaves])
    hi = lo + w[:, None]
    lev = np.array([b[3] for b in leaves])
    bad = []
    for i in range(len(leaves)):
        touch = np.all((lo <= hi[i]) & (hi >= lo[i]), axis=1)
        touch[i] = False
        if touch.any() and np.abs(lev[touch] - lev[i]).max() > 1:
            bad.append(i)
    return bad, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp')
    ap.add_argument('out')
    ap.add_argument('--deck')
    ap.add_argument('--deck-out')
    ap.add_argument('--manifest')
    a = ap.parse_args()
    r = read(a.inp)
    R = r['root']
    mi = r['mesh_indcs']
    # RegionIndcs: ng, nx1, nx2, nx3, is, ie, js, je, ks, ke, cnx1, cnx2, cnx3, cis, cie, cjs, cje, cks, cke
    ng, nx1, nx2, nx3 = (int(v) for v in mi[:4])
    mbi = np.frombuffer(r['indcs'][76:], dtype=np.int32)
    nroot = [nx1 // int(mbi[1]), nx2 // int(mbi[2]), nx3 // int(mbi[3])]
    if len(set(nroot)) != 1 or nroot[0] % 4 or (1 << R) != nroot[0]:
        sys.exit(f'need equal root block counts divisible by 4 with 2^root_level = n_root: {nroot}, root {R}')
    n = nroot[0]
    o = n // 4
    x1min, x2min, x3min, x1max, x2max, x3max, dx1, dx2, dx3 = r['size']
    old = [x1min, x1max, x2min, x2max, x3min, x3max]
    c = [0.5 * (x1min + x1max), 0.5 * (x2min + x2max), 0.5 * (x3min + x3max)]
    W = [x1max - x1min, x2max - x2min, x3max - x3min]
    new_box = [c[0] - W[0], c[0] + W[0], c[1] - W[1], c[1] + W[1], c[2] - W[2], c[2] + W[2]]
    nsize = [new_box[0], new_box[2], new_box[4], new_box[1], new_box[3], new_box[5], 2 * dx1, 2 * dx2, 2 * dx3]

    # the old blocks, one level deeper, offset by o root blocks
    blocks = []                                  # (lx1, lx2, lx3, level, old index or -1)
    for i, (lx, ly, lz, L) in enumerate(r['ll']):
        s = (L + 1) - R
        blocks.append((int(lx) + o * (1 << s), int(ly) + o * (1 << s), int(lz) + o * (1 << s), int(L) + 1, i))
    # the new root blocks outside the old box
    for k in range(n):
        for j in range(n):
            for i in range(n):
                if o <= i < o + n // 2 and o <= j < o + n // 2 and o <= k < o + n // 2:
                    continue
                blocks.append((i, j, k, R, -1))
    # 2:1 balance: split a new block while a neighbour is >= 2 levels finer
    while True:
        lmax = max(b[3] for b in blocks)
        # split the new blocks that are the coarser side of a violation (an old one there is an input error)
        lev = np.array([b[3] for b in blocks])
        lo = np.array([[b[0] << (lmax - b[3]), b[1] << (lmax - b[3]), b[2] << (lmax - b[3])] for b in blocks])
        w = np.array([1 << (lmax - b[3]) for b in blocks])
        hi = lo + w[:, None]
        todo = set()
        for i in range(len(blocks)):
            touch = np.all((lo <= hi[i]) & (hi >= lo[i]), axis=1)
            touch[i] = False
            if touch.any() and lev[touch].max() - lev[i] > 1:
                if blocks[i][4] >= 0:
                    sys.exit(f'old block {blocks[i]} is 2+ levels coarser than a neighbour (input unbalanced?)')
                todo.add(i)
        if not todo:
            break
        nb = []
        for i, b in enumerate(blocks):
            if i in todo:
                for ch in range(8):
                    nb.append((2 * b[0] + (ch & 1), 2 * b[1] + ((ch >> 1) & 1), 2 * b[2] + ((ch >> 2) & 1),
                               b[3] + 1, -1))
            else:
                nb.append(b)
        blocks = nb
    lmax = max(b[3] for b in blocks)
    blocks.sort(key=lambda b: zkey(b[0], b[1], b[2], b[3], lmax))

    # check (ii): Z-sorted (by construction), exact tiling, no overlap, 2:1
    vol = sum((1 << (lmax - b[3]))**3 for b in blocks)
    full = (n << (lmax - R))**3
    bad, lo, hi = neighbours_ok([b[:4] for b in blocks], lmax)
    keys = set()
    for b in blocks:
        s = lmax - b[3]
        keys.add((b[0] << s, b[1] << s, b[2] << s))
    overl = len(keys) != len(blocks)
    if vol != full or bad or overl:
        sys.exit(f'tree check failed: volume {vol}/{full}, unbalanced {len(bad)}, overlap {overl}')

    # parameters
    par_txt = r['par'].decode()
    new_par, seen = edit_par(par_txt, old, new_box)
    new_par_b = new_par.encode()
    nnew = sum(1 for b in blocks if b[4] < 0)
    ds = r['dsize']
    hsh_in = []
    with open(a.inp, 'rb') as fi:
        for i in range(r['nmb']):
            fi.seek(r['off'] + i * ds)
            hsh_in.append(hashlib.sha256(fi.read(ds)).hexdigest())
    zero = bytes(ds)
    with open(a.inp, 'rb') as fi, open(a.out, 'wb') as fo:
        fo.write(new_par_b)
        fo.write(struct.pack('2i', len(blocks), R))
        fo.write(struct.pack('9d', *nsize))
        fo.write(r['indcs'])
        fo.write(struct.pack('2d', r['time'], r['dt']))
        fo.write(struct.pack('i', r['ncycle']))
        fo.write(np.array([b[:4] for b in blocks], dtype=np.int32).tobytes())
        fo.write(np.array([r['cost'][b[4]] if b[4] >= 0 else 1.0 for b in blocks], dtype=np.float32).tobytes())
        fo.write(struct.pack('Q', ds))
        for b in blocks:
            if b[4] >= 0:
                fi.seek(r['off'] + b[4] * ds)
                fo.write(fi.read(ds))
            else:
                fo.write(zero)

    # check (i): re-read the output, each old block's record byte-identical
    o2 = read(a.out)
    nbad = 0
    with open(a.out, 'rb') as f:
        for g, b in enumerate(blocks):
            if b[4] < 0:
                continue
            f.seek(o2['off'] + g * ds)
            if hashlib.sha256(f.read(ds)).hexdigest() != hsh_in[b[4]]:
                nbad += 1
    if nbad or o2['nmb'] != len(blocks):
        sys.exit(f'bitwise check failed: {nbad} old blocks differ')

    if a.deck:
        d, _ = edit_par(open(a.deck).read(), old, new_box, deck=True)
        open(a.deck_out, 'w').write(d)
    man = dict(tool='r1_embed.py', input=os.path.abspath(a.inp), output=os.path.abspath(a.out),
               t=r['time'], ncycle=r['ncycle'], old_box=old, new_box=new_box, n_root=n, offset_root_blocks=o,
               root_level=R, levels_old=[int(r['ll'][:, 3].min() - R), int(r['ll'][:, 3].max() - R)],
               levels_new=[min(b[3] for b in blocks) - R, lmax - R], nmb_old=r['nmb'], nmb_new=len(blocks),
               new_blocks=nnew, data_size=ds,
               edited=sorted(str(s) for s in seen),
               checks=dict(old_blocks_bitwise=True, z_sorted=True, tiling=True, balance_2to1=True))
    if a.manifest:
        json.dump(man, open(a.manifest, 'w'), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
