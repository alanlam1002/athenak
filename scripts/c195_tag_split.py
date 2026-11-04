#!/usr/bin/env python3
"""c195 (R-227 (iii)): the warm tail split by Lagrangian tag (wip/dent 42486e68, <mhd> lagrangian_tags). Per prim_3d dump:
  classes as c169_b1_ana (core rho >= 0.1 rho_c; outer 0.01-0.1 rho_c; rem = Hill sphere about the rho >= 0.01 rho_c centroid;
  stream = outside r_H, 1e-6 <= rho/rho_c < 0.01).
  Per class: mass M, the mass share with K/K0 > 1.5 ('tail'), and the tail's origin split by tag: tag0 = s_00 (the initial
  0.01-0.1 rho_c band), tag1 = s_01 (the initial rho >= 0.1 rho_c), other = 1 - s_00 - s_01 (initial stream/atmosphere).
  Per tag (Lagrangian view): the tagged mass sum m s_n, its share per current class, its mass-weighted K/K0 quantiles 5/50/95,
  and its share with K/K0 > 1.5.  Tag bounds: min/max of s_00, s_01, s_00 + s_01 over cells with rho >= 1e-6 rho_c.
  Runs without tags (no s_01 / s_02): only the class tail shares.
    c195_tag_split.py LABEL=RUNDIR ...   (all prim_3d / prim_3d_peri dumps)"""
import glob, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh, par   # noqa: E402
GAM = 5.0/3.0
TAIL = 1.5


def wq(a, w, qs):
    if len(a) == 0 or w.sum() <= 0:
        return [np.nan]*len(qs)
    o = np.argsort(a); c = np.cumsum(w[o]); c /= c[-1]
    return [a[o][min(np.searchsorted(c, q), len(a) - 1)] for q in qs]


def one(d, f):
    P = bc.read_binary(f); g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
    cat = lambda k: np.concatenate([np.asarray(P['mb_data'][k][b]).ravel() for b in range(len(g))]).astype(float)
    rho, p = cat('dens'), cat('press')
    tags = 's_01' in P['var_names'] and 's_02' in P['var_names']
    s0, s1 = (cat('s_00'), cat('s_01')) if tags else (None, None)
    h = (g[:, 1] - g[:, 0])/n; m = rho*np.repeat(h**3, n**3)
    ax = lambda lo: (g[:, lo][:, None] + (np.arange(n) + 0.5)[None, :]*h[:, None])
    x = np.broadcast_to(ax(0)[:, None, None, :], (len(g), n, n, n)).ravel()
    y = np.broadcast_to(ax(2)[:, None, :, None], (len(g), n, n, n)).ravel()
    z = np.broadcast_to(ax(4)[:, :, None, None], (len(g), n, n, n)).ravel()
    rhoc, k0, mbh = par(d, 'rhoc'), par(d, 'kappa'), par(d, 'puncture_mass', 1.0)
    r = rho/rhoc; K = p/rho**GAM/k0
    d01 = r >= 0.01
    xc = np.array([(m[d01]*c[d01]).sum()/m[d01].sum() for c in (x, y, z)]); D = np.linalg.norm(xc - xbh(d, P['time']))
    rr = np.sqrt((x - xc[0])**2 + (y - xc[1])**2 + (z - xc[2])**2); mrem = m[d01].sum()
    for _ in range(3):
        rh = D*(mrem/(3.0*mbh))**(1.0/3.0); inh = rr < rh; mrem = m[inh].sum()
    cls = {'core': r >= 0.1, 'outer': d01 & (r < 0.1), 'rem': inh, 'stream': (~inh) & (r >= 1e-6) & (r < 0.01)}
    hot = K > TAIL
    out = [f"t={P['time']:7.2f}"]
    for c, s in cls.items():
        M = m[s].sum(); Mt = m[s & hot].sum()
        e = f"{c}: M {M:.3e} tail {Mt/M if M > 0 else np.nan:.4f}"
        if tags and Mt > 0:
            a0 = (m*s0)[s & hot].sum()/Mt; a1 = (m*s1)[s & hot].sum()/Mt
            e += f" [tag0 {a0:.3f} tag1 {a1:.3f} other {1 - a0 - a1:.3f}]"
        out.append(e)
    print(' | '.join(out))
    if tags:
        live = r >= 1e-6
        print(f"    bounds: s_00 [{s0[live].min():.3e}, {s0[live].max():.6f}] s_01 [{s1[live].min():.3e}, {s1[live].max():.6f}] "
              f"sum [{(s0 + s1)[live].min():.3e}, {(s0 + s1)[live].max():.6f}]")
        for nm, s in (('tag0', s0), ('tag1', s1)):
            w = m*s; W = w.sum()
            sh = ' '.join(f"{c} {w[cm].sum()/W:.3f}" for c, cm in cls.items())
            q = wq(K, w, (0.05, 0.5, 0.95))
            print(f"    {nm}: M {W:.4e} where [{sh}] K/K0 {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f} tail {w[hot].sum()/W:.4f}")
    sys.stdout.flush()


for a in sys.argv[1:]:
    lab, d = a.split('=', 1)
    fs = glob.glob(f'{d}/bin/*.prim_3d.0*.bin') + glob.glob(f'{d}/bin/*.prim_3d_peri.0*.bin')
    print(f'== {lab} ({d})')
    for f in sorted(fs, key=lambda f: bc.read_binary(f)['time']):
        one(d, f)
