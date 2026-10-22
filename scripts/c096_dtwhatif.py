#!/usr/bin/env python3
"""C-096 (R-099 B2): offline (S0-style) dt what-ifs on a fixed-binary (>= 30c52c1d) restart.

Reuses c046_s0.py's restart pass (load: ideal-gas C2P of the stored conserved state, the CFL
wave speeds of dyn_grmhd_newdt.cpp with gr_dt = true and B = 0, CFL 0.4) and its zone model
(non-excised cells with r_BH < r_s and rho < rho_thr: v = beta'/alpha capped at |v| <= v_max
along beta', cold), for these variants of <coord> excise_lapse and <mhd> atmosphere_zone_vmax:
  base   0.25 / 0.90   (the production deck)
  (i)    0.30 / 0.90
  (ii)   0.25 / 0.95
  (iii)  0.30 / 0.95
dt_skip_excised = true drops cells with alpha < excise_lapse from the CFL. Per variant: dt, the
gain over base, and the NLIM limiting cells (r_BH, level, rho/rho_c, alpha, |beta'|/alpha,
zone/capped, |lam| = |alpha-term - shift|). ARGUED approximations: the C2P/floors and the zone's
own state are not replayed (c046_s0.py's caveats); tier3 is ignored (it acts at r >= r_s).

    c096_dtwhatif.py RESTART [--rs 6] [--thr 1e-10] [--nlim 5] [--out FILE]
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c046_s0 as S                                # noqa: E402
from c045_dt import RHOC, CFL                      # noqa: E402

VARIANTS = (('base', 0.25, 0.90), ('(i)', 0.30, 0.90), ('(ii)', 0.25, 0.95), ('(iii)', 0.30, 0.95))


def variant(B, Cc, exl, vmax, rs, thr):
    """per-candidate-cell dt (r_BH < RCAND) under (excise_lapse, v_max); far cells keep base dt"""
    exc = Cc['alpha'] < exl
    zone = (Cc['r'] < rs) & (Cc['rho'] < thr * RHOC) & ~exc
    dt = np.where(exc, np.inf, Cc['dt'])                    # baseline speeds where not zone
    beta = [Cc['b0'], Cc['b1'], Cc['b2']]
    gd = [[Cc['g00'], Cc['g01'], Cc['g02']], [Cc['g01'], Cc['g11'], Cc['g12']],
          [Cc['g02'], Cc['g12'], Cc['g22']]]
    bnorm = np.sqrt(np.maximum(sum(gd[i][j] * beta[i] * beta[j] for i in range(3) for j in range(3)), 0))
    vb = bnorm / Cc['alpha']
    f = np.minimum(1.0, vmax / np.maximum(vb, 1e-300))
    v = [f * beta[i] / Cc['alpha'] for i in range(3)]
    v2 = np.minimum((f * vb)**2, 0.99)
    lam, _, at, bt = S.lam_hydro(Cc['alpha'], beta, v, [Cc['gi0'], Cc['gi1'], Cc['gi2']], v2,
                                 np.zeros_like(v2))
    dtz = CFL * Cc['dx'] / np.maximum(lam, 1e-300)
    dt = np.where(zone, dtz, dt)
    # speeds for reporting: zone cells from the zone model, others recomputed from the baseline
    lam_rep = np.where(zone, lam, CFL * Cc['dx'] / np.where(np.isfinite(Cc['dt']), Cc['dt'], np.inf))
    far = B['dt'][B['r'] >= S.RCAND]
    dfar = far.min() if far.size else np.inf
    return dict(dt=dt, zone=zone, capped=zone & (f < 1), vb=vb, lam=lam_rep, at=np.where(zone, at, np.nan),
                bt=np.where(zone, bt, np.nan), dfar=dfar, exc=exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('restart')
    ap.add_argument('--rs', type=float, default=6.0)
    ap.add_argument('--thr', type=float, default=1e-10)
    ap.add_argument('--nlim', type=int, default=5)
    ap.add_argument('--out')
    a = ap.parse_args()
    t, B, Cc = S.load(a.restart)
    rows = [f'{a.restart}: t = {t:.3f}; candidate cells r_BH < {S.RCAND:g}: {Cc["r"].size}; '
            f'zone r_s {a.rs:g}, rho_thr {a.thr:g} rho_c']
    base = None
    for lab, exl, vm in VARIANTS:
        w = variant(B, Cc, exl, vm, a.rs, a.thr)
        dmin = min(w['dt'].min(), w['dfar'])
        base = base or dmin
        exr = Cc['r'][w['exc']].max() if w['exc'].any() else 0.0
        rows.append(f'-- {lab}: excise_lapse {exl:.2f}, v_max {vm:.2f}: dt {dmin:.4f} (x{dmin / base:.3f}); '
                    f'excised out to r_BH {exr:.3f}; zone {w["zone"].sum()} cells, capped {w["capped"].sum()}; '
                    f'far (r >= {S.RCAND:g}) min dt {w["dfar"]:.4f}')
        o = np.argsort(w['dt'])[:a.nlim]
        for j in o:
            rows.append(f'     dt {w["dt"][j]:.4f} r_BH {Cc["r"][j]:.3f} L{Cc["lev"][j]} rho {Cc["rho"][j] / RHOC:.1e} '
                        f'alpha {Cc["alpha"][j]:.3f} |beta|/alpha {w["vb"][j]:.3f} '
                        f'{"zone-capped" if w["capped"][j] else ("zone" if w["zone"][j] else "non-zone")} '
                        f'|lam| {w["lam"][j]:.3f}'
                        + (f' = |a-term {w["at"][j]:+.3f} - shift {w["bt"][j]:+.3f}|' if w['zone'][j] else ''))
    txt = '\n'.join(rows)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
