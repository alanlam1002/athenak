#!/usr/bin/env python3
"""c178 (R-215 item 1): the B.1 max-defect check, Ma = rms 1e-8 + mg_min_vcycles 1 vs Mb = mg_norm max 1e-7 + mg_min_vcycles 1,
t = 41.78 -> 60 from B1N_cell 00001.rst, mg_log 2 (athenak wip/kbg 5628336a: MultigridDriver::LogSolve prints per solve the
V-cycles, the final volume-weighted L2 and max |defect| over the MeshBlock cells, and the position / level of the max).
Per arm and solver (X, beta, psi, alpha_psi), over all solves of the window (and t >= 43, i.e. without the restart's first
cycles): distributions (5/50/95% and max) of the max |defect|, the rms (l2), the V-cycles; and for the location of the
max |defect|:
  r_BH = |x - x_BH(t)| (x_BH = puncture_x - frame_disp, c160_coldmax.xbh);
  r_c / R_* with x_c the rho dV centroid of rho >= 0.01 rho_c and R_* the radius about x_c enclosing 99% of that mass, both
    linearly interpolated in time between the prim_3d dumps (every 6 M);
  the refinement interface: in the prim_3d dump nearest in time (the mesh can change between dumps; |dt| <= 3 M), the
    distance in cells (of the containing block) from the point to the nearest block face shared with a block of a
    different level (probe points 1, 2, 4 cells away along +-x, +-y, +-z): 'iface' if <= 1 cell, the share of solves.
    c178_maxdef_ana.py OUT.txt LABEL=RUNDIR [...]"""
import glob, re, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh, par   # noqa: E402

PAT = re.compile(r'^MGLOG (\S+) cyc (\d+) t (\S+) it (\d+) wall (\S+) conv0 (\S+) conv (\S+) ref (\S+) target (\S+) '
                 r'l2 (\S+) max (\S+) src_l2 (\S+) src_max (\S+)(?: at (\S+) (\S+) (\S+) lev (\d+) ch (\d+))?')


def star(d, rhoc):
    o = []
    for f in sorted(glob.glob(f'{d}/bin/*.prim_3d.0*.bin')):
        P = bc.read_binary(f); g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
        rho = np.concatenate([np.asarray(P['mb_data']['dens'][b], float).ravel() for b in range(len(g))])
        h = (g[:, 1] - g[:, 0])/n
        ax = lambda lo: (g[:, lo][:, None] + (np.arange(n) + 0.5)[None, :]*h[:, None])
        x = np.broadcast_to(ax(0)[:, None, None, :], (len(g), n, n, n)).ravel()
        y = np.broadcast_to(ax(2)[:, None, :, None], (len(g), n, n, n)).ravel()
        z = np.broadcast_to(ax(4)[:, :, None, None], (len(g), n, n, n)).ravel()
        m = rho*np.repeat(h**3, n**3); s = rho >= 0.01*rhoc
        xc = np.array([(m[s]*c[s]).sum()/m[s].sum() for c in (x, y, z)])
        r = np.sqrt((x[s] - xc[0])**2 + (y[s] - xc[1])**2 + (z[s] - xc[2])**2); o_ = np.argsort(r)
        cm = np.cumsum(m[s][o_]); R = r[o_][np.searchsorted(cm, 0.99*cm[-1])]
        o.append((P['time'], xc, R, g, n))
    return o


def block_of(g, p):
    ins = np.where((p[0] >= g[:, 0]) & (p[0] < g[:, 1]) & (p[1] >= g[:, 2]) & (p[1] < g[:, 3]) &
                   (p[2] >= g[:, 4]) & (p[2] < g[:, 5]))[0]
    return ins[0] if len(ins) else -1


def iface_cells(g, n, p):
    b = block_of(g, p)
    if b < 0:
        return np.nan
    w = g[b, 1] - g[b, 0]; h = w/n
    for k in (1, 2, 4):
        for ax in range(3):
            for sgn in (-1, 1):
                q = np.array(p, float); q[ax] += sgn*k*h
                c = block_of(g, q)
                if c >= 0 and not np.isclose(g[c, 1] - g[c, 0], w):
                    return k
    return np.inf


def q(a):
    a = np.asarray(a, float)
    return '%.2e/%.2e/%.2e max %.2e' % (np.percentile(a, 5), np.percentile(a, 50), np.percentile(a, 95), a.max()) if len(a) else '-'


def main(out, specs):
    fo = open(out, 'w')
    pr = lambda s: (print(s), fo.write(s + '\n'))
    for s in specs:
        lab, d = s.split('=', 1)
        rhoc = par(d, 'rhoc')
        st = star(d, rhoc); ts = np.array([x[0] for x in st])
        L = [m.groups() for m in (PAT.match(l) for l in open(f'{d}/out.log')) if m]
        cyc = [int(x[1]) for x in L]
        wall = re.findall(r'cpu time used\s*=\s*(\S+)', open(f'{d}/out.log').read())
        pr(f'===== {lab} ({d}): {len(L)} MGLOG lines, cycles {min(cyc)}-{max(cyc)}; wall {wall[-1] if wall else "?"} s '
           f'-> {float(wall[-1])/(max(cyc) - min(cyc) + 1):.3f} s/cycle' if wall else '')
        for nm in ('X', 'beta', 'psi', 'alpha_psi'):
            R = [x for x in L if x[0] == nm]
            for sel, tag in ((lambda x: True, 'all'), (lambda x: float(x[2]) >= 43.0, 't>=43')):
                S = [x for x in R if sel(x)]
                mx = [float(x[10]) for x in S]; l2 = [float(x[9]) for x in S]; it = np.array([int(x[3]) for x in S])
                rbh, rc, ifc = [], [], []
                for x in S:
                    if x[13] is None:
                        continue
                    t = float(x[2]); p = np.array([float(x[13]), float(x[14]), float(x[15])])
                    rbh.append(np.linalg.norm(p - xbh(d, t)))
                    i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2)); f = (t - ts[i])/(ts[i + 1] - ts[i])
                    xc = (1 - f)*st[i][1] + f*st[i + 1][1]; Rs = (1 - f)*st[i][2] + f*st[i + 1][2]
                    rc.append(np.linalg.norm(p - xc)/Rs)
                    j = int(np.argmin(np.abs(ts - t)))
                    ifc.append(iface_cells(st[j][3], st[j][4], p))
                ifc = np.array(ifc)
                vc = ' '.join(f'{k}:{(it == k).mean():.2f}' for k in range(1, 6)) + f' >5:{(it > 5).mean():.2f}'
                pr(f'{lab} {nm:9s} [{tag:5s}] n={len(S):4d} max|def| {q(mx)} | l2 {q(l2)} | max/l2 med {np.median(np.array(mx)/np.array(l2)):.0f} '
                   f'| V-cycles mean {it.mean():.2f} ({vc})')
                if len(rc):
                    lev = [int(x[16]) for x in S if x[16] is not None]
                    pr(f'{"":24s} at: r_BH {q(rbh)} | r_c/R* {q(rc)} (share inside R*: {(np.array(rc) < 1).mean():.2f}) | '
                       f'levels {dict(zip(*np.unique(lev, return_counts=True)))} | coarse-fine face <= 1 cell: {(ifc <= 1).mean():.2f}, '
                       f'<= 4 cells: {(ifc <= 4).mean():.2f}')
        # the most frequent max positions (rounded to 0.5 M)
        P = [(round(float(x[13])*2)/2, round(float(x[14])*2)/2, round(float(x[15])*2)/2) for x in L if x[13] is not None and x[0] in ('psi', 'alpha_psi')]
        u, c = np.unique(np.array(P), axis=0, return_counts=True); o = np.argsort(-c)[:5]
        pr(f'{lab} psi/alpha_psi most frequent max-|defect| positions (0.5 M bins): ' + '; '.join(f'{tuple(u[k])} x{c[k]}' for k in o))
        pr(f'{lab} star: ' + '; '.join(f't={x[0]:.1f} x_c=({x[1][0]:.2f},{x[1][1]:.2f},{x[1][2]:.2f}) R*={x[2]:.3f} x_BH=({xbh(d, x[0])[0]:.2f},{xbh(d, x[0])[1]:.2f},{xbh(d, x[0])[2]:.2f})' for x in st))
    fo.close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
