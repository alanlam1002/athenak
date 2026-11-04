#!/usr/bin/env python3
"""c194 (R-227: why De's sub-threshold stream tail is hotter than Bbr's). At a time where a dediag_3d and a prim_3d dump
coincide (De) and Bbr's prim_3d at the same time: per rho/rho_c band around the switch (rho_switch = 3.2e-8 = 1e-4 rho_c on D),
the stream's (outside the Hill sphere) mass quantiles of K/K0:
  De: the prim K (what ran: tracer in codes 11-13, energy in 14/15/0) and, for codes 11/12 (where de_ken_ktr is formed), the
      energy state's K_en = (1 + dk) K_tr, i.e. what the gas would carry if it switched to the energy solution;
  Bbr: the (energy-only) prim K.
    c194_threshold.py T DE_DIR BBR_DIR"""
import glob, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh, par   # noqa: E402
GAM = 5.0/3.0
BANDS = [(1e-6, 1e-5), (1e-5, 3e-5), (3e-5, 1e-4), (1e-4, 3e-4), (3e-4, 1e-3), (1e-3, 1e-2)]


def wq(a, w, qs):
    if len(a) == 0:
        return [np.nan]*len(qs)
    o = np.argsort(a); c = np.cumsum(w[o]); c /= c[-1]
    return [a[o][min(np.searchsorted(c, q), len(a) - 1)] for q in qs]


def load(d, T, diag):
    fp = min(glob.glob(f'{d}/bin/*.prim_3d.0*.bin') + glob.glob(f'{d}/bin/*.prim_3d_peri.0*.bin'),
             key=lambda f: abs(bc.read_binary(f)['time'] - T))
    P = bc.read_binary(fp); g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
    cat = lambda S, k: np.concatenate([np.asarray(S['mb_data'][k][b]).ravel() for b in range(len(g))]).astype(float)
    rho, p = cat(P, 'dens'), cat(P, 'press')
    br = dk = None
    if diag:
        fd = min(glob.glob(f'{d}/bin/*.dediag_3d.*.bin'), key=lambda f: abs(bc.read_binary(f)['time'] - T))
        Dg = bc.read_binary(fd)
        assert Dg['time'] == P['time'] and np.array_equal(g, np.asarray(Dg['mb_geometry'], float))
        br = cat(Dg, 'de_branch').round().astype(int); dk = cat(Dg, 'de_ken_ktr')
    h = (g[:, 1] - g[:, 0])/n; m = rho*np.repeat(h**3, n**3)
    ax = lambda lo: (g[:, lo][:, None] + (np.arange(n) + 0.5)[None, :]*h[:, None])
    x = np.broadcast_to(ax(0)[:, None, None, :], (len(g), n, n, n)).ravel()
    y = np.broadcast_to(ax(2)[:, None, :, None], (len(g), n, n, n)).ravel()
    z = np.broadcast_to(ax(4)[:, :, None, None], (len(g), n, n, n)).ravel()
    rhoc, k0, mbh = par(d, 'rhoc'), par(d, 'kappa'), par(d, 'puncture_mass', 1.0)
    d01 = rho >= 0.01*rhoc
    xc = np.array([(m[d01]*c[d01]).sum()/m[d01].sum() for c in (x, y, z)]); D = np.linalg.norm(xc - xbh(d, P['time']))
    rr = np.sqrt((x - xc[0])**2 + (y - xc[1])**2 + (z - xc[2])**2); mrem = m[d01].sum()
    for _ in range(3):
        rh = D*(mrem/(3.0*mbh))**(1.0/3.0); inh = rr < rh; mrem = m[inh].sum()
    return P['time'], rho/rhoc, p/rho**GAM/k0, m, ~inh, br, dk


T = float(sys.argv[1])
t, r, K, m, out, br, dk = load(sys.argv[2], T, True)
tb, rb, Kb, mb, outb, _, _ = load(sys.argv[3], T, False)
print(f'De t={t:.2f}, Bbr t={tb:.2f}; stream cells (outside r_H); K/K0 mass quantiles 5/50/95')
for lo, hi in BANDS:
    s = out & (r >= lo) & (r < hi); sb = outb & (rb >= lo) & (rb < hi)
    codes = ' '.join(f'{c}:{m[s & (br == c)].sum()/max(m[s].sum(), 1e-300):.2f}' for c in (0, 11, 12, 13, 14, 15)
                     if m[s & (br == c)].sum() > 0.005*m[s].sum())
    q = wq(K[s], m[s], (0.05, 0.5, 0.95)); qb = wq(Kb[sb], mb[sb], (0.05, 0.5, 0.95))
    tr = s & ((br == 11) | (br == 12))
    ken = (1.0 + dk[tr])*K[tr]
    qe = wq(ken, m[tr], (0.05, 0.5, 0.95))
    print(f'{lo:.0e}-{hi:.0e}: De M {m[s].sum():.2e} [{codes}] K {q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g} | '
          f'De tracer cells K_en {qe[0]:.3g}/{qe[1]:.3g}/{qe[2]:.3g} | Bbr M {mb[sb].sum():.2e} K {qb[0]:.3g}/{qb[1]:.3g}/{qb[2]:.3g}')
