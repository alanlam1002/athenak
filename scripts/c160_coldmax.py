#!/usr/bin/env python3
"""c160 (R-194 item 4, user-approved 2026-10-31): the density-maximum cell of the B.1 arms (C-155: B1C K/K0 0.008, KW 0.097
at t = 120), offline from the prim_3d dumps (every 6 M). Per arm and dump: the rho_max cell's rho/rho_c and K/K0, its
position relative to the star (the rho dV centroid of rho >= 0.01 rho_c; r/R*, R* = 0.677) and to the hole (x_BH =
puncture_x - frame_disp, the user history), its MeshBlock's level, its distance in cells to the block's nearest face and
the level across that face, and its 26 neighbours' rho/rho_c and K/K0 (min, median, max; same block only) -- whether it is a
single-cell spike; plus the five densest cells.
    c160_coldmax.py OUT.csv RUNDIR[,RUNDIR...] [--tmin 90]"""
import glob, os, re, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
GAM = 5.0/3.0
RSTAR = 0.677


def par(d, k, dflt=0.0):
    m = re.search(rf'^\s*{k}\s*=\s*([-+.\deE]+)', open(f'{d}/parfile.par').read(), re.M)
    return float(m.group(1)) if m else dflt


def xbh(d, t):
    p0 = np.array([par(d, f'puncture_x{a}') for a in (1, 2, 3)])
    f = glob.glob(f'{d}/*.user.hst')[0]
    hdr = open(f).readlines()[1]
    if 'frame_disp_x1' not in hdr:
        return p0
    h = np.loadtxt(f)
    c = [int(re.search(rf'\[(\d+)\]=frame_disp_x{a}', hdr).group(1)) - 1 for a in (1, 2, 3)]
    i = np.argmin(np.abs(h[:, 0] - t))
    return p0 - h[i, c]


def level_at(g, lev, x):
    """the level of the block containing point x (-1: none)"""
    m = ((g[:, 0] <= x[0]) & (x[0] < g[:, 1]) & (g[:, 2] <= x[1]) & (x[1] < g[:, 3]) &
         (g[:, 4] <= x[2]) & (x[2] < g[:, 5]))
    return int(lev[m][0]) if m.any() else -1


def one(d, f, rhoc, k0):
    P = bc.read_binary(f)
    g = np.asarray(P['mb_geometry'], dtype=float); lev = np.asarray(P['mb_logical'])[:, 3]
    n = P['nx1_out_mb']; q = P['mb_data']
    rho = np.array([np.asarray(q['dens'][b]) for b in range(len(g))])   # (nmb, k, j, i)
    p = np.array([np.asarray(q['press'][b]) for b in range(len(g))])
    K = p/rho**GAM/k0
    b, k, j, i = np.unravel_index(np.argmax(rho), rho.shape)
    dx = (g[b, 1] - g[b, 0])/n
    xc = np.array([g[b, 0] + (i + 0.5)*dx, g[b, 2] + (j + 0.5)*dx, g[b, 4] + (k + 0.5)*dx])
    # the star's centroid and the hole
    m = rho >= 0.01*rhoc
    dv = ((g[:, 1] - g[:, 0])/n)**3
    w = rho*dv[:, None, None, None]
    cen = []
    for a, (lo, hi) in enumerate(((0, 1), (2, 3), (4, 5))):
        c1 = g[:, lo][:, None] + (np.arange(n) + 0.5)[None, :]*((g[:, hi] - g[:, lo])/n)[:, None]
        sh = [1, 1, 1, 1]; sh[0] = len(g); sh[3 - a] = n
        cen.append((w*m*c1.reshape(sh)).sum()/(w*m).sum())
    cen = np.array(cen)
    xb = xbh(d, float(P['time']))
    # nearest block face and the level across it
    lo = np.array([i, j, k]); dist = np.minimum(lo, n - 1 - lo)
    a = int(np.argmin(dist)); side = -1 if lo[a] < n - 1 - lo[a] else 1
    xo = xc.copy(); xo[a] += side*(dist[a] + 1.0)*dx
    lx = level_at(g, lev, xo)
    # neighbours (same block)
    sl = tuple(slice(max(c - 1, 0), min(c + 2, n)) for c in (k, j, i))
    nr, nk = rho[b][sl].ravel()/rhoc, K[b][sl].ravel()
    self_ = (rho[b][sl] == rho[b, k, j, i]).ravel()
    nr, nk = nr[~self_], nk[~self_]
    top = np.sort(rho.ravel())[::-1][:5]/rhoc
    return dict(t=float(P['time']), rhomax=rho[b, k, j, i]/rhoc, K=K[b, k, j, i], x=xc[0], y=xc[1], z=xc[2],
                r_star=np.linalg.norm(xc - cen)/RSTAR, r_bh=np.linalg.norm(xc - xb),
                cen_bh=np.linalg.norm(cen - xb), level=int(lev[b]), dx=dx, face_cells=int(dist[a]),
                face_axis=a, level_across=lx, nb_rho_min=nr.min(), nb_rho_med=np.median(nr), nb_rho_max=nr.max(),
                nb_K_min=nk.min(), nb_K_med=np.median(nk), nb_K_max=nk.max(),
                top5=' '.join(f'{v:.3f}' for v in top))


def main():
    out, runs = sys.argv[1], sys.argv[2].split(',')
    tmin = float(sys.argv[sys.argv.index('--tmin') + 1]) if '--tmin' in sys.argv else 90.0
    rows = []
    for d in runs:
        rhoc, k0 = par(d, 'rhoc'), par(d, 'kappa')
        for f in sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin')):
            if bc.read_binary(f)['time'] < tmin - 1e-6:
                continue
            r = one(d, f, rhoc, k0)
            r = {'arm': os.path.basename(d.rstrip('/')), **r}
            rows.append(r)
            print(', '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items()), flush=True)
    keys = list(rows[0].keys())
    with open(out, 'w') as fo:
        fo.write(','.join(keys) + '\n')
        for r in rows:
            fo.write(','.join(str(r[k]) for k in keys) + '\n')


if __name__ == '__main__':
    main()
