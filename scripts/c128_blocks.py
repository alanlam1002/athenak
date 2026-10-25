#!/usr/bin/env python3
"""R-140 item 7 (D9, c128): per-level MeshBlock counts against t from existing runs, and the ideal
Berger-Oliger gain
    S = sum_l N_l / sum_l N_l 2^-(L-l),   L = the finest level present,
i.e. the cost of a global (finest-level) dt over the cost of per-level time steps with a factor-2
subcycle per level, counting MeshBlock-updates only (no CFC/MG, no ghost/AMR overheads; argued: an upper
bound on the hydro gain: the CFC elliptic solve, which dominated the wall in an earlier GPU profile, is not
subcycled by BO).
Source: the 3D dumps (prim_3d every 12 M + prim_3d_peri), which cover the whole mesh; only the block
headers are read (bin_convert.read_binary's record layout: 6 int32 index, 4 int32 logical (lx1, lx2, lx3,
level), 6 location reals, then the data, skipped). The level is the logical level minus the root's
(the out.log 'Physical level' numbering). Also the out.log segment-start tables, as a check.
    c128_blocks.py OUTDIR LABEL=RUNDIR [...]
"""
import glob
import os
import re
import sys

import numpy as np

B = 'cfc_tde_wd_imbh_parabolic_tracker'


def levels(fn):
    with open(fn, 'rb') as fp:
        fp.seek(0, 2); size = fp.tell(); fp.seek(0)
        fp.readline()
        n = int(fp.readline().split(b'=')[-1])
        ph = {}
        for _ in range(n - 1):
            k, v = [x.strip() for x in fp.readline().decode().split('=')]
            ph[k] = v
        loc, var = int(ph['size of location']), int(ph['size of variable'])
        nv = int(fp.readline().split(b'=')[-1])
        fp.readline()
        hs = int(fp.readline().split(b'=')[-1])
        fp.seek(hs, 1)
        lev = []
        while fp.tell() < size:
            idx = np.frombuffer(fp.read(24), dtype=np.int32)
            lg = np.frombuffer(fp.read(16), dtype=np.int32)
            fp.seek(6 * loc, 1)
            ncell = (idx[1] - idx[0] + 1) * (idx[3] - idx[2] + 1) * (idx[5] - idx[4] + 1)
            fp.seek(int(ncell) * nv * var, 1)
            lev.append(lg[3])
    return float(ph['time']), np.array(lev)


def outlog(d):
    """segment-start tables: (line, {physical level: N})"""
    rows, cur = [], None
    for ln in open(f'{d}/out.log', errors='ignore'):
        m = re.match(r'\s+Physical level = (\d+) \(logical level = (\d+)\): (\d+) MeshBlocks', ln)
        if m:
            cur = cur or {}
            cur[int(m.group(1))] = int(m.group(3))
        elif cur is not None:
            rows.append(cur); cur = None
    return rows


def gain(N):
    L = max(N)
    return sum(N.values()) / sum(n * 2.0**-(L - l) for l, n in N.items())


def plot(out, lines):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [ln.split(',') for ln in lines[1:] if ',dump,' in ln]
    labs = list(dict.fromkeys(r[0] for r in rows))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for lab in labs:
        R = sorted({(float(r[1]), float(r[-2])) for r in rows if r[0] == lab})
        ax[0].plot(*zip(*R), '-' if 'C' in lab.split('_')[1] else '--', label=lab)
    ax[0].set_xlabel('t [M]'); ax[0].set_ylabel('ideal BO gain S'); ax[0].legend(fontsize=7)
    for k, lab in enumerate(('rp10_CD', 'rp18_CD')):
        R = sorted({tuple(float(x) for x in [r[1]] + r[4:9]) for r in rows if r[0] == lab})
        R = np.array(R)
        ax[k + 1].stackplot(R[:, 0], R[:, 1:].T, labels=[f'level {l}' for l in range(5)])
        ax[k + 1].set_title(f'{lab}: MeshBlocks per level'); ax[k + 1].set_xlabel('t [M]')
        ax[k + 1].legend(fontsize=7, loc='upper left')
    fig.tight_layout(); fig.savefig(f'{out}/c128_blocks.png', dpi=80); plt.close(fig)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    lines = ['label,t,src,L,' + ','.join(f'N{l}' for l in range(8)) + ',Ntot,S_BO,frac_finest']
    summ = ['label,t_first,t_last,S_min,S_median,S_max,S_mean_timeweighted']
    for spec in sys.argv[2:]:
        lab, d = spec.split('=', 1)
        recs = []
        files = sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin')) + sorted(glob.glob(f'{d}/bin/{B}.prim_3d_peri.*.bin'))
        for f in files:
            t, lev = levels(f)
            root = lev.min()
            N = {int(l - root): int((lev == l).sum()) for l in np.unique(lev)}
            recs.append((t, 'dump', N))
        tabs = outlog(d)
        for i, N in enumerate(tabs):
            recs.append((np.nan, f'outlog_seg{i}', N))
        recs.sort(key=lambda x: (np.isnan(x[0]), x[0]))
        S = []
        for t, src, N in recs:
            L = max(N); tot = sum(N.values()); s = gain(N)
            if src == 'dump':
                S.append((t, s))
            lines.append(f'{lab},{t:.4g},{src},{L},' + ','.join(str(N.get(l, 0)) for l in range(8))
                         + f',{tot},{s:.4f},{N[L] / tot:.4f}')
        S = np.array(sorted(set(S)))
        tw = np.trapezoid(S[:, 1], S[:, 0]) / (S[-1, 0] - S[0, 0])
        summ.append(f'{lab},{S[0, 0]:.4g},{S[-1, 0]:.4g},{S[:, 1].min():.4f},{np.median(S[:, 1]):.4f},'
                    f'{S[:, 1].max():.4f},{tw:.4f}')
    open(f'{out}/c128_blocks.csv', 'w').write('\n'.join(lines) + '\n')
    plot(out, lines)
    open(f'{out}/c128_gain_summary.csv', 'w').write('\n'.join(summ) + '\n')
    print('\n'.join(summ))


if __name__ == '__main__':
    main()
