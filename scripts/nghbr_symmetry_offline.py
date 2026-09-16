#!/usr/bin/env python3
"""Offline audit of the SetNeighbors neighbor table, straight from a restart file.

A restart file carries the complete MeshBlockTree (the list of LogicalLocations)
plus the parameter deck, which is everything MeshBlock::SetNeighbors depends on.
So the neighbor table of a real, large, failing run can be rebuilt and checked on
a laptop in seconds -- no build, no MPI, no allocation. Use this BEFORE spending a
job on a topology question.

It checks the invariant every boundary-exchange consumer assumes (see
src/cfc/SETNEIGHBORS_HANDOFF.md section 8.1):

    nghbr(b,n) = {gid=T, dest=d}   =>   nghbr(T,d) = {gid=b, dest=n}

and, for the 'parity' rule, also that nothing the rule declines to register
leaves a ghost region unfilled -- a rule can be perfectly symmetric and still be
wrong by dropping data, so symmetry alone is not sufficient.

Two registration rules are implemented so they can be compared on identical
input:

  parity  the pre-fee50981 guard: register a coarser diagonal only when this
          block's octant parity matches the direction on every nonzero axis
  head    fee50981 + 8641e00f: register unless the target's reciprocal query
          finds a genuine same-level leaf (corners additionally require the
          target's own GetLeaf to resolve back to this block)

Usage:
    python3 scripts/nghbr_symmetry_offline.py <restart-file.rst> [--rule both]

The in-tree equivalent, for a live run, is ATHENAK_CHECK_NGHBR_SYMMETRY=1
(MeshBlock::CheckNeighborSymmetry). The two were cross-checked against each
other on the TDE production topology and on
inputs/tests/amr_hydro_nghbr_asymmetry.athinput; they agree exactly.
"""

import itertools
import re
import struct
import sys

import struct, sys

def read_restart_topology(path):
    with open(path,'rb') as f:
        blob = f.read(4*1024*1024)   # par dump + header + lloc list fits easily
    marker = b'<par_end>'
    i = blob.find(marker)
    assert i >= 0, "no <par_end>"
    # ParameterDump writes "<par_end>\n"; find end of that line
    j = blob.index(b'\n', i) + 1
    pardump = blob[:j].decode('utf-8', 'replace')
    off = j
    nmb_total, root_level = struct.unpack_from('<ii', blob, off); off += 8
    mesh_size = struct.unpack_from('<9d', blob, off); off += 72
    mesh_indcs = struct.unpack_from('<19i', blob, off); off += 76
    mb_indcs   = struct.unpack_from('<19i', blob, off); off += 76
    time, dt = struct.unpack_from('<2d', blob, off); off += 16
    ncycle, = struct.unpack_from('<i', blob, off); off += 4
    llocs = []
    for m in range(nmb_total):
        lx1,lx2,lx3,lev = struct.unpack_from('<4i', blob, off + 16*m)
        llocs.append((lx1,lx2,lx3,lev))
    return dict(pardump=pardump, nmb_total=nmb_total, root_level=root_level,
                mesh_size=mesh_size, mesh_indcs=mesh_indcs, mb_indcs=mb_indcs,
                time=time, dt=dt, ncycle=ncycle, llocs=llocs)


# ---------------------------------------------------------------- NeighborIndex
def NeighborIndex(ix, iy, iz, n1, n2):
    if abs(ix) + abs(iy) + abs(iz) == 0: return -1
    if abs(ix*iy*iz) > 1: return -1
    if iz == 0:
        if ix*iy == 0:
            sub = n1 + 2*n2
            return abs(ix)*2*(ix+1) + abs(iy)*2*(iy+5) + sub
        return 16 + (ix+1) + 2*(iy+1) + n1
    else:
        if ix*iy == 0:
            sub = n1 + 2*n2
            return 24 + abs(ix)*(ix+9) + abs(iy)*(iy+17) + 2*(iz+1) + sub
        return 48 + (ix+1)//2 + (iy+1) + 2*(iz+1)

# ---------------------------------------------------------------- tree
class Node:
    __slots__ = ('lx1','lx2','lx3','level','kids','gid')
    def __init__(s, lx1,lx2,lx3,level):
        s.lx1,s.lx2,s.lx3,s.level = lx1,lx2,lx3,level
        s.kids = None          # None == leaf (pleaf_ == nullptr)
        s.gid = -1
    def leaf(s, ox1,ox2,ox3):  # GetLeaf
        return s.kids[ox1 + 2*ox2 + 4*ox3]

class Tree:
    def __init__(s, llocs, root_level, nmb_rootx, bcs):
        s.root = Node(0,0,0,0)
        s.root_level = root_level
        s.nmb_rootx = nmb_rootx          # (nx,ny,nz) at root_level
        s.bcs = bcs                      # dict face -> str
        s.byloc = {}
        for gid,(lx1,lx2,lx3,lev) in enumerate(llocs):
            s._insert(lx1,lx2,lx3,lev,gid)
    def _insert(s, lx1,lx2,lx3,lev,gid):
        node = s.root
        for L in range(lev):
            sh = lev - L - 1
            if node.kids is None:
                node.kids = [None]*8
                for k in range(8):
                    ox,oy,oz = k&1, (k>>1)&1, (k>>2)&1
                    node.kids[k] = Node(2*node.lx1+ox, 2*node.lx2+oy,
                                        2*node.lx3+oz, node.level+1)
            ox = (lx1>>sh)&1; oy = (lx2>>sh)&1; oz = (lx3>>sh)&1
            node = node.kids[ox + 2*oy + 4*oz]
        assert node.kids is None and node.gid == -1
        node.gid = gid
        s.byloc[(lx1,lx2,lx3,lev)] = node

    def nmb_at(s, axis, ll):
        return s.nmb_rootx[axis] << (ll - s.root_level)

    def FindNeighbor(s, loc, ox1, ox2, ox3):
        lx1,lx2,lx3,ll = loc
        lx, ly, lz = lx1+ox1, lx2+ox2, lx3+ox3
        for axis,(v,lo,hi) in enumerate(((lx,'ix1_bc','ox1_bc'),(ly,'ix2_bc','ox2_bc'),(lz,'ix3_bc','ox3_bc'))):
            n = s.nmb_at(axis, ll)
            if v < 0:
                if s.bcs[lo] in ('periodic','shear_periodic'): v = n-1
                else: return None
            elif v >= n:
                if s.bcs[hi] == 'periodic': v = 0
                else: return None
            if axis==0: lx=v
            elif axis==1: ly=v
            else: lz=v
        if ll < 1: return s.root
        bt = s.root
        ox=oy=oz=0
        for L in range(ll):
            if bt.kids is None:
                if L == ll-1: return bt
                raise RuntimeError("tree broken (descend)")
            sh = ll - L - 1
            ox = (lx>>sh)&1; oy = (ly>>sh)&1; oz = (lz>>sh)&1
            bt = bt.kids[ox + 2*oy + 4*oz]
            if bt is None: raise RuntimeError("tree broken (null)")
        if bt.kids is None: return bt
        ox = 1 if ox1 < 0 else 0
        oy = 1 if ox2 < 0 else 0
        oz = 1 if ox3 < 0 else 0
        btleaf = bt.leaf(ox,oy,oz)
        if btleaf.kids is None: return bt
        raise RuntimeError("tree broken (2:1 violated)")

# ---------------------------------------------------------------- SetNeighbors
NN = 56

def set_neighbors(tree, llocs, rule):
    """rule: 'parity' (pre-fee50981), 'head' (fee50981+8641e00f), or 'parity_free'.
    Returns nghbr[gid] = list of (gid,lev,dest) or None."""
    nmb = len(llocs)
    nghbr = [[None]*NN for _ in range(nmb)]
    nfx=nfy=nfz=2
    for b in range(nmb):
        lx1,lx2,lx3,lev = llocs[b]
        loc = (lx1,lx2,lx3,lev)
        myfx1 = lx1 & 1; myfx2 = lx2 & 1; myfx3 = lx3 & 1
        myox1 = myfx1*2-1; myox2 = myfx2*2-1; myox3 = myfx3*2-1
        row = nghbr[b]
        def put(i, node, dest):
            row[i] = (node.gid, node.level, dest)

        # ---- x1 faces
        for n in (-1,1):
            nt = tree.FindNeighbor(loc, n,0,0)
            if nt is None: continue
            if nt.kids is not None:
                ffx = 1-(n+1)//2
                for fz in range(nfz):
                    for fy in range(nfy):
                        nf = nt.leaf(ffx,fy,fz)
                        put(NeighborIndex(n,0,0,fy,fz), nf, NeighborIndex(-n,0,0,fy,fz))
            else:
                if nt.level == lev:
                    i, d = NeighborIndex(n,0,0,0,0), NeighborIndex(-n,0,0,0,0)
                else:
                    i, d = NeighborIndex(n,0,0,myfx2,myfx3), NeighborIndex(-n,0,0,myfx2,myfx3)
                put(i, nt, d)
        # ---- x2 faces
        for m in (-1,1):
            nt = tree.FindNeighbor(loc, 0,m,0)
            if nt is None: continue
            if nt.kids is not None:
                ffy = 1-(m+1)//2
                for fz in range(nfz):
                    for fx in range(nfx):
                        nf = nt.leaf(fx,ffy,fz)
                        put(NeighborIndex(0,m,0,fx,fz), nf, NeighborIndex(0,-m,0,fx,fz))
            else:
                if nt.level == lev:
                    i, d = NeighborIndex(0,m,0,0,0), NeighborIndex(0,-m,0,0,0)
                else:
                    i, d = NeighborIndex(0,m,0,myfx1,myfx3), NeighborIndex(0,-m,0,myfx1,myfx3)
                put(i, nt, d)
        # ---- x1x2 edges
        for m in (-1,1):
            for n in (-1,1):
                nt = tree.FindNeighbor(loc, n,m,0)
                if nt is None: continue
                if nt.kids is not None:
                    ffx = 1-(n+1)//2; ffy = 1-(m+1)//2
                    for fz in range(nfz):
                        nf = nt.leaf(ffx,ffy,fz)
                        put(NeighborIndex(n,m,0,fz,0), nf, NeighborIndex(-n,-m,0,fz,0))
                else:
                    if nt.level == lev:
                        i,d = NeighborIndex(n,m,0,0,0), NeighborIndex(-n,-m,0,0,0)
                        reg = True
                    else:
                        i,d = NeighborIndex(n,m,0,myfx3,0), NeighborIndex(-n,-m,0,myfx3,0)
                        if rule == 'parity':
                            reg = (myox1==n and myox2==m)
                        else:
                            r = tree.FindNeighbor((nt.lx1,nt.lx2,nt.lx3,nt.level), -n,-m,0)
                            reg = (r is None) or (r.kids is not None)
                    if reg: put(i, nt, d)
        # ---- x3 faces
        for l in (-1,1):
            nt = tree.FindNeighbor(loc, 0,0,l)
            if nt is None: continue
            if nt.kids is not None:
                ffz = 1-(l+1)//2
                for fy in range(nfy):
                    for fx in range(nfx):
                        nf = nt.leaf(fx,fy,ffz)
                        put(NeighborIndex(0,0,l,fx,fy), nf, NeighborIndex(0,0,-l,fx,fy))
            else:
                if nt.level == lev:
                    i,d = NeighborIndex(0,0,l,0,0), NeighborIndex(0,0,-l,0,0)
                else:
                    i,d = NeighborIndex(0,0,l,myfx1,myfx2), NeighborIndex(0,0,-l,myfx1,myfx2)
                put(i, nt, d)
        # ---- x3x1 edges
        for l in (-1,1):
            for n in (-1,1):
                nt = tree.FindNeighbor(loc, n,0,l)
                if nt is None: continue
                if nt.kids is not None:
                    ffx = 1-(n+1)//2; ffz = 1-(l+1)//2
                    for fy in range(nfy):
                        nf = nt.leaf(ffx,fy,ffz)
                        put(NeighborIndex(n,0,l,fy,0), nf, NeighborIndex(-n,0,-l,fy,0))
                else:
                    if nt.level == lev:
                        i,d = NeighborIndex(n,0,l,0,0), NeighborIndex(-n,0,-l,0,0)
                        reg = True
                    else:
                        i,d = NeighborIndex(n,0,l,myfx2,0), NeighborIndex(-n,0,-l,myfx2,0)
                        if rule == 'parity':
                            reg = (myox1==n and myox3==l)
                        else:
                            r = tree.FindNeighbor((nt.lx1,nt.lx2,nt.lx3,nt.level), -n,0,-l)
                            reg = (r is None) or (r.kids is not None)
                    if reg: put(i, nt, d)
        # ---- x2x3 edges
        for l in (-1,1):
            for m in (-1,1):
                nt = tree.FindNeighbor(loc, 0,m,l)
                if nt is None: continue
                if nt.kids is not None:
                    ffy = 1-(m+1)//2; ffz = 1-(l+1)//2
                    for fx in range(nfy):     # NOTE: nfy, mirrors the C++ (harmless, nfx==nfy)
                        nf = nt.leaf(fx,ffy,ffz)
                        put(NeighborIndex(0,m,l,fx,0), nf, NeighborIndex(0,-m,-l,fx,0))
                else:
                    if nt.level == lev:
                        i,d = NeighborIndex(0,m,l,0,0), NeighborIndex(0,-m,-l,0,0)
                        reg = True
                    else:
                        i,d = NeighborIndex(0,m,l,myfx1,0), NeighborIndex(0,-m,-l,myfx1,0)
                        if rule == 'parity':
                            reg = (myox2==m and myox3==l)
                        else:
                            r = tree.FindNeighbor((nt.lx1,nt.lx2,nt.lx3,nt.level), 0,-m,-l)
                            reg = (r is None) or (r.kids is not None)
                    if reg: put(i, nt, d)
        # ---- corners
        for l in (-1,1):
            for m in (-1,1):
                for n in (-1,1):
                    nt = tree.FindNeighbor(loc, n,m,l)
                    if nt is None: continue
                    if nt.kids is not None:
                        nt = nt.leaf(1-(n+1)//2, 1-(m+1)//2, 1-(l+1)//2)
                    if nt.level >= lev:
                        reg = True
                    elif rule == 'parity':
                        reg = (myox1==n and myox2==m and myox3==l)
                    else:  # head: fee50981 + 8641e00f corner correction
                        r = tree.FindNeighbor((nt.lx1,nt.lx2,nt.lx3,nt.level), -n,-m,-l)
                        if r is None:
                            reg = True
                        elif r.kids is None:
                            reg = False
                        else:
                            rl = r.leaf(1-(-n+1)//2, 1-(-m+1)//2, 1-(-l+1)//2)
                            reg = (rl.gid == b)
                    if reg:
                        put(NeighborIndex(n,m,l,0,0), nt, NeighborIndex(-n,-m,-l,0,0))
    return nghbr

# ---------------------------------------------------------------- symmetry check
def check(nghbr, llocs):
    """The invariant BuildRankPackedVarMetadata / multigrid_bvals depend on:
    if b's slot n names (T, dest=d), then T's slot d must name (b, dest=n)."""
    bad = []
    for b,row in enumerate(nghbr):
        for n,e in enumerate(row):
            if e is None: continue
            t, tlev, d = e
            te = nghbr[t][d]
            if te is None:
                bad.append(('ORPHAN', b, n, t, d, None))
            elif te[0] != b:
                bad.append(('STOLEN', b, n, t, d, te[0]))
            elif te[2] != n:
                bad.append(('DESTMISMATCH', b, n, t, d, te[2]))
    return bad


def coverage_check(tree, llocs, nghbr_parity):
    """For every coarser diagonal candidate the parity rule DROPS, verify the
    same target gid is registered at the lower-order (face/edge) slot whose
    coarse-recv index range provably extends into the dropped direction."""
    holes = []
    for b,(lx1,lx2,lx3,lev) in enumerate(llocs):
        loc=(lx1,lx2,lx3,lev); f=(lx1&1, lx2&1, lx3&1)
        row = nghbr_parity[b]
        for d in itertools.product((-1,0,1),repeat=3):
            if sum(1 for x in d if x)<2: continue     # diagonals only
            nt = tree.FindNeighbor(loc,*d)
            if nt is None or nt.kids is not None or nt.level>=lev: continue
            S = [i for i in range(3) if d[i]!=0 and (f[i]*2-1)!=d[i]]
            if not S: continue                        # parity matched: registered
            if len(S)==3: holes.append(('ALL3',b,d)); continue
            dr = list(d)
            for i in S: dr[i]=0
            # the reduced direction's slot: free axes are exactly S
            if dr[0]!=0 and dr[1]!=0:   n1,n2 = f[2],0
            elif dr[0]!=0 and dr[2]!=0: n1,n2 = f[1],0
            elif dr[1]!=0 and dr[2]!=0: n1,n2 = f[0],0
            elif dr[0]!=0:              n1,n2 = f[1],f[2]
            elif dr[1]!=0:              n1,n2 = f[0],f[2]
            else:                       n1,n2 = f[0],f[1]
            slot = NeighborIndex(dr[0],dr[1],dr[2],n1,n2)
            e = row[slot]
            if e is None or e[0]!=nt.gid:
                holes.append(('NOCOVER',b,d,dr,nt.gid,e))
                continue
            # index range extends toward +i when f[i]==0, toward -i when f[i]==1
            for i in S:
                if (1 if f[i]==0 else -1) != d[i]:
                    holes.append(('WRONGSIDE',b,d,i))
    return holes



def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument('restart', help='path to a .rst restart file')
    ap.add_argument('--rule', default='both', choices=('parity','head','both'))
    args = ap.parse_args()

    d = read_restart_topology(args.restart)
    bcs = {}
    for key in ('ix1_bc','ox1_bc','ix2_bc','ox2_bc','ix3_bc','ox3_bc'):
        m = re.search(r'^\s*%s\s*=\s*(\S+)' % key, d['pardump'], re.M)
        bcs[key] = m.group(1) if m else 'outflow'
    mi, mbi = d['mesh_indcs'], d['mb_indcs']
    nmb_rootx = (mi[1]//mbi[1], mi[2]//mbi[2], mi[3]//mbi[3])
    levels = sorted(set(l[3] for l in d['llocs']))
    print("%s" % args.restart)
    print("  root grid %s at level %d | %d MeshBlocks | levels %s | cycle %d, t=%g"
          % (nmb_rootx, d['root_level'], d['nmb_total'], levels, d['ncycle'], d['time']))
    print("  bcs %s" % {k.replace('_bc',''): v for k, v in bcs.items()})

    tree = Tree(d['llocs'], d['root_level'], nmb_rootx, bcs)
    rules = ('parity','head') if args.rule == 'both' else (args.rule,)
    worst = 0
    for rule in rules:
        ng = set_neighbors(tree, d['llocs'], rule)
        n_entries = sum(1 for r in ng for e in r if e is not None)
        bad = check(ng, d['llocs'])
        kinds = {}
        for k in bad:
            kinds[k[0]] = kinds.get(k[0], 0) + 1
        extra = ''
        if rule == 'parity':
            holes = coverage_check(tree, d['llocs'], ng)
            extra = '  ghost-holes=%d' % len(holes)
            worst += len(holes)
        print("  rule=%-7s registrations=%6d  asymmetric=%5d  %s%s"
              % (rule, n_entries, len(bad), kinds if kinds else '', extra))
        for kind, b, n, t, dd, other in bad[:12]:
            print("      %-12s gid=%d slot=%d -> gid=%d dest=%d ; that slot holds %s"
                  % (kind, b, n, t, dd, other))
        if len(bad) > 12:
            print("      ... %d more" % (len(bad) - 12))
        worst += len(bad)
    return 1 if worst else 0


if __name__ == '__main__':
    sys.exit(main())
