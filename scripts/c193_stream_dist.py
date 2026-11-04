#!/usr/bin/env python3
"""c193 (R-225 item 3): the stream's K distribution (c169's stream: 1e-6 <= rho/rho_c < 0.01 outside the Hill sphere), per
density band, at the dumps nearest the given times: mass quantiles 5/50/95 of K/K0, the mass share with K/K0 > 10 and > 100,
and the share of the stream's mass-weighted <K/K0> carried by cells with K/K0 > 100 (does a tail carry the mean?).
    c193_stream_dist.py T1,T2,... LABEL=RUNDIR ..."""
import glob, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh, par   # noqa: E402
GAM = 5.0/3.0


def wq(a, w, qs):
    o = np.argsort(a); c = np.cumsum(w[o]); c /= c[-1]
    return [a[o][np.searchsorted(c, q)] for q in qs]


TT = [float(x) for x in sys.argv[1].split(',')]
for s in sys.argv[2:]:
    lab, d = s.split('=', 1)
    rhoc, k0, mbh = par(d, 'rhoc'), par(d, 'kappa'), par(d, 'puncture_mass', 1.0)
    fs = glob.glob(f'{d}/bin/*.prim_3d.0*.bin') + glob.glob(f'{d}/bin/*.prim_3d_peri.0*.bin')
    tf = [(bc.read_binary(f)['time'], f) for f in fs]
    for T in TT:
        t, f = min(tf, key=lambda x: abs(x[0] - T))
        if abs(t - T) > 2.5:
            continue
        P = bc.read_binary(f); g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']; q = P['mb_data']
        cat = lambda k: np.concatenate([np.asarray(q[k][b]).ravel() for b in range(len(g))]).astype(float)
        rho, p = cat('dens'), cat('press')
        h = (g[:, 1] - g[:, 0])/n; dv = np.repeat(h**3, n**3)
        ax = lambda lo: (lo[:, None] + (np.arange(n) + 0.5)[None, :]*h[:, None])
        X, Y, Z = ax(g[:, 0]), ax(g[:, 2]), ax(g[:, 4])
        x = np.broadcast_to(X[:, None, None, :], (len(g), n, n, n)).ravel()
        y = np.broadcast_to(Y[:, None, :, None], (len(g), n, n, n)).ravel()
        z = np.broadcast_to(Z[:, :, None, None], (len(g), n, n, n)).ravel()
        K = p/rho**GAM/k0; m = rho*dv; d01 = rho >= 0.01*rhoc
        xc = np.array([(m[d01]*c[d01]).sum()/m[d01].sum() for c in (x, y, z)]); D = np.linalg.norm(xc - xbh(d, t))
        rr = np.sqrt((x - xc[0])**2 + (y - xc[1])**2 + (z - xc[2])**2); mrem = m[d01].sum()
        for _ in range(3):
            rh = D*(mrem/(3.0*mbh))**(1.0/3.0); inh = rr < rh; mrem = m[inh].sum()
        for lo, hi in ((1e-6, 1e-2), (1e-6, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2)):
            st = (~inh) & (rho >= lo*rhoc) & (rho < hi*rhoc)
            if st.sum() == 0:
                continue
            k, w = K[st], m[st]; M = w.sum(); mean = (k*w).sum()/M
            q5, q50, q95 = wq(k, w, (0.05, 0.5, 0.95))
            print(f'{lab:5s} t={t:6.1f} rho/rho_c {lo:.0e}-{hi:.0e}: M {M:.3e} K/K0 q5 {q5:9.3g} q50 {q50:9.3g} q95 {q95:9.3g} '
                  f'mean {mean:9.3g} | M(K>10) {w[k > 10].sum()/M:.3f} M(K>100) {w[k > 100].sum()/M:.3f} '
                  f'mean share K>100 {(k*w)[k > 100].sum()/(k*w).sum():.3f}', flush=True)
