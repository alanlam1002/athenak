#!/usr/bin/env python3
"""c183 (R-210 stage 2): near-hole shells for the self-gravity beds (c175_r210 X2x/X2b_L3/L4 Michel, X3x/X3b_L4 hydrostatic).
Per adm_3d dump: beta_res = beta - beta0, beta0 = the background-only identity run's shift at t = 0 on the same mesh (I2x_L{3,4} /
I3x_L4, residual_scale 0: beta = beta0 exactly); |beta_res| (coordinate) max and mean in radial shells about the hole (origin), and
|(L beta_res)^FD| = |d_i b_j + d_j b_i - 2/3 delta_ij d_k b^k| (flat conformal Killing operator, second-order central differences
inside each MeshBlock, the block-edge cells dropped). Per trd_3d dump: the trace-free deposit tr_sig_tt and kcfc_bg's extra tau
source tr_dsK, sqrt-free cell sums times dV per shell (coordinate volume). Shells [1,2), [2,3), [3,4), [4,6), [6,10), [10,20) M.
    c183_r210_shells.py OUT.txt RUN:GATE [RUN:GATE ...]"""
import glob, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402
SH = [(1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 20)]


def blocks(P):
    g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']; h = (g[:, 1] - g[:, 0])/n
    return g, n, h


def rgrid(g, n, h, b):
    c = lambda lo: g[b, lo] + (np.arange(n) + 0.5)*h[b]
    z, y, x = np.meshgrid(c(4), c(2), c(0), indexing='ij')
    return np.sqrt(x*x + y*y + z*z)


def main(out, specs):
    fo = open(out, 'w')
    pr = lambda s: (print(s), fo.write(s + '\n'))
    for sp in specs:
        run, gate = sp.split(':')
        G = bc.read_binary(sorted(glob.glob(f'{gate}/bin/*adm_3d.00000.bin'))[0])
        gg, n, hg = blocks(G)
        key = {tuple(np.round(gg[b], 9)): b for b in range(len(gg))}
        for f in sorted(glob.glob(f'{run}/bin/*adm_3d.*.bin')):
            P = bc.read_binary(f); g, n, h = blocks(P)
            acc = {s: [0.0, 0.0, 0, 0.0] for s in SH}   # max|b|, sum|b|, n, max|Lb|
            miss = 0
            for b in range(len(g)):
                bg = key.get(tuple(np.round(g[b], 9)))
                if bg is None:
                    miss += 1; continue
                br = [np.asarray(P['mb_data'][k][b], float) - np.asarray(G['mb_data'][k][bg], float)
                      for k in ('adm_betax', 'adm_betay', 'adm_betaz')]
                r = rgrid(g, n, h, b); mb = np.sqrt(sum(c*c for c in br))
                d = [[np.gradient(br[j], h[b], axis=2 - i) for i in range(3)] for j in range(3)]   # d[j][i] = d_i b_j
                div = d[0][0] + d[1][1] + d[2][2]
                L2 = np.zeros_like(r)
                for i in range(3):
                    for j in range(3):
                        L = d[j][i] + d[i][j] - (2.0/3.0)*(div if i == j else 0.0)
                        L2 += L*L
                Lb = np.sqrt(L2)
                inner = (slice(1, -1),)*3
                for s in SH:
                    m = (r >= s[0]) & (r < s[1])
                    if m.any():
                        a = acc[s]; a[0] = max(a[0], mb[m].max()); a[1] += mb[m].sum(); a[2] += m.sum()
                    mi = m[inner]
                    if mi.any():
                        acc[s][3] = max(acc[s][3], Lb[inner][mi].max())
            pr(f'{run} t={P["time"]:7.2f} (unmatched blocks {miss}) ' + ' | '.join(
                f'r[{s[0]},{s[1]}): |b_res| max {a[0]:.2e} mean {a[1]/max(a[2],1):.2e} |Lb_res| max {a[3]:.2e}' for s, a in acc.items()))
        for f in sorted(glob.glob(f'{run}/bin/*trd_3d.*.bin')):
            P = bc.read_binary(f); g, n, h = blocks(P)
            acc = {s: [0.0, 0.0] for s in SH}
            for b in range(len(g)):
                r = rgrid(g, n, h, b); dv = h[b]**3
                st = np.asarray(P['mb_data']['tr_sig_tt'][b], float); ds = np.asarray(P['mb_data']['tr_dsK'][b], float)
                for s in SH:
                    m = (r >= s[0]) & (r < s[1])
                    acc[s][0] += st[m].sum()*dv; acc[s][1] += ds[m].sum()*dv
            pr(f'{run} t={P["time"]:7.2f} deposit ' + ' | '.join(f'r[{s[0]},{s[1]}): sig_tt {a[0]:+.3e} dsK {a[1]:+.3e}' for s, a in acc.items()))
    fo.close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
