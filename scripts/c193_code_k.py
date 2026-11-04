#!/usr/bin/env python3
"""c193 (R-225 item 3): K/K0 per D-ent/D-int branch code and class, at a time where a dediag_3d and a prim_3d dump coincide
(same mesh, cell by cell): per class and code the mass share and the mass quantiles 5/50/95 of K/K0 (prim = what the run used).
    c193_code_k.py T RUNDIR [RUNDIR ...]"""
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
    return [a[o][min(np.searchsorted(c, q), len(a) - 1)] for q in qs]


T = float(sys.argv[1])
for d in sys.argv[2:]:
    rhoc, k0, mbh, tfl = par(d, 'rhoc'), par(d, 'kappa'), par(d, 'puncture_mass', 1.0), par(d, 'tfloor')
    fd = min(glob.glob(f'{d}/bin/*.dediag_3d.*.bin'), key=lambda f: abs(bc.read_binary(f)['time'] - T))
    fp = min(glob.glob(f'{d}/bin/*.prim_3d.0*.bin') + glob.glob(f'{d}/bin/*.prim_3d_peri.0*.bin'),
             key=lambda f: abs(bc.read_binary(f)['time'] - T))
    Dg, P = bc.read_binary(fd), bc.read_binary(fp)
    g = np.asarray(Dg['mb_geometry'], float); n = Dg['nx1_out_mb']
    assert Dg['time'] == P['time'] and np.array_equal(g, np.asarray(P['mb_geometry'], float)), 'dumps not co-located'
    cat = lambda S, k: np.concatenate([np.asarray(S['mb_data'][k][b]).ravel() for b in range(len(g))]).astype(float)
    br = cat(Dg, 'de_branch').round().astype(int); rho, p, Tm = cat(P, 'dens'), cat(P, 'press'), cat(P, 'temperature')
    h = (g[:, 1] - g[:, 0])/n; m = rho*np.repeat(h**3, n**3); K = p/rho**GAM/k0; fl = Tm <= 1.0001*tfl
    ax = lambda lo: (g[:, lo][:, None] + (np.arange(n) + 0.5)[None, :]*h[:, None])
    x = np.broadcast_to(ax(0)[:, None, None, :], (len(g), n, n, n)).ravel()
    y = np.broadcast_to(ax(2)[:, None, :, None], (len(g), n, n, n)).ravel()
    z = np.broadcast_to(ax(4)[:, :, None, None], (len(g), n, n, n)).ravel()
    d01, d1 = rho >= 0.01*rhoc, rho >= 0.1*rhoc
    xc = np.array([(m[d01]*c[d01]).sum()/m[d01].sum() for c in (x, y, z)]); D = np.linalg.norm(xc - xbh(d, Dg['time']))
    rr = np.sqrt((x - xc[0])**2 + (y - xc[1])**2 + (z - xc[2])**2); mrem = m[d01].sum()
    for _ in range(3):
        rh = D*(mrem/(3.0*mbh))**(1.0/3.0); inh = rr < rh; mrem = m[inh].sum()
    st = (~inh) & (rho >= 1e-6*rhoc) & (rho < 0.01*rhoc)
    cls = {'core': d1, 'outer': d01 & ~d1, 'stream': st, 'st_lo(1e-6..1e-4)': st & (rho < 1e-4*rhoc),
           'st_hi(1e-3..1e-2)': st & (rho >= 1e-3*rhoc)}
    print(f'== {d} t={Dg["time"]:.2f}')
    for c, s in cls.items():
        M = m[s].sum()
        out = []
        for b in sorted(set(br[s])):
            q = s & (br == b)
            if m[q].sum()/M < 0.005:
                continue
            k5, k50, k95 = wq(K[q], m[q], (0.05, 0.5, 0.95))
            out.append(f'{b}: {m[q].sum()/M:.3f} K/K0 {k50:.3g} [{k5:.2g},{k95:.2g}] fl {m[q & fl].sum()/m[q].sum():.2f}')
        print(f'  {c:18s} M {M:.3e} | ' + ' | '.join(out))
