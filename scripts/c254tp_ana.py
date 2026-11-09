#!/usr/bin/env python3
"""c254 (R-254 items 2-3): return-rate curves from c254tp_integrate.py runs.
Mdot_ret(t) = mass-weighted histogram of the first-return times to R_ret (log bins, 12 per decade), against t - t_p.
Per run: the peak (time, rate), the late slope (least-squares in log-log over t in [3 t_peak, 0.8 t_end - t_p]), the local
slope (centred, 3-bin smoothed), the break time = the first time after the peak from which the local slope stays below
-(5/3 + 9/4)/2 = -1.958; the masses by outcome. Physical units for M_BH = 1e4 M_sun (GM/c^3 = 0.04925 s; M* = 7.47e-5 M_BH =
0.747 M_sun).
    c254tp_ana.py OUTPREFIX LABEL=RUN.npz [LABEL=RUN.npz ...]"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MS = 7.47e-5; TSEC = 4.925e-6*1e4; YR = 3.156e7; MSUN_PER_MS = 0.747
out = sys.argv[1]; runs = [a.split('=', 1) for a in sys.argv[2:]]
SL = -(5/3 + 9/4)/2
rows = []
curves = {}
for lab, f in runs:
    R = np.load(f)
    tp, tmb, W = float(R['tp']), float(R['tmb_ref']), float(R['W'])
    st, tev = R['status'], R['t_ev']
    m = st == 1
    tr = tev[m] - tp
    t_end = tp + 10*tmb   # as integrated (tend_tmb = 10 x the canonical t_mb)
    lo, hi = max(tr.min()*0.9, 50.0), (t_end - tp)
    edges = np.logspace(np.log10(lo), np.log10(hi), int(12*np.log10(hi/lo)) + 1)
    cnt, _ = np.histogram(tr, bins=edges)
    md = cnt*W/np.diff(edges)/MS                        # M*/M
    tc = np.sqrt(edges[1:]*edges[:-1])
    ok = cnt >= 5
    ip = np.argmax(np.where(ok, md, 0))
    tpk, mpk = tc[ip], md[ip]
    fit = ok & (tc >= 3*tpk) & (tc <= 0.8*(t_end - tp))
    slope = np.polyfit(np.log(tc[fit]), np.log(md[fit]), 1)[0] if fit.sum() >= 3 else np.nan
    lt, lm = np.log(tc), np.log(np.where(ok, md, np.nan))
    loc = np.gradient(lm, lt)
    loc = np.convolve(np.nan_to_num(loc, nan=0.0), np.ones(3)/3, mode='same')
    brk = np.nan
    after = np.nonzero((tc > tpk) & ok)[0]
    for k in after:
        tail = loc[k:][ok[k:]]
        if len(tail) >= 3 and np.all(tail < SL):
            brk = tc[k]; break
    out_m = {k: W*np.sum(st == k)/MS for k in range(1, 6)}
    rows.append((lab, str(R['soft']), int(R['N']), int(R['seed']), tmb, tpk, mpk, slope, brk, out_m))
    curves[lab] = (tc[ok], md[ok], tmb)
    np.savetxt(f'{out}_{lab}.csv', np.c_[tc[ok], md[ok], md[ok]*MSUN_PER_MS/TSEC*YR, loc[ok]], delimiter=',',
               header='t_minus_tp_M,Mdot_Mstar_per_M,Mdot_Msun_per_yr_1e4,local_slope', comments='')
print(f"{'run':24s} {'soft':5s} {'N':>7s} {'t_mb':>7s} {'t_peak':>8s} {'t_pk/tmb':>8s} {'Mdot_pk M*/M':>12s} {'Msun/yr':>9s} "
      f"{'late slope':>10s} {'t_break':>8s} {'t_br/tmb':>8s} | returned / remnant / swallowed / escaped / out (M*)")
for lab, sf, N, sd, tmb, tpk, mpk, sl, brk, om in rows:
    print(f"{lab:24s} {sf:5s} {N:7d} {tmb:7.0f} {tpk:8.0f} {tpk/tmb:8.2f} {mpk:12.3e} {mpk*MSUN_PER_MS/TSEC*YR:9.3g} "
          f"{sl:10.2f} {brk:8.0f} {brk/tmb:8.2f} | " + ' / '.join(f'{om[k]:.4f}' for k in range(1, 6)))
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for lab, (tc, md, tmb) in curves.items():
    ax.loglog(tc, md, label=lab)
t0 = np.array([2e4, 2e5])
ax.loglog(t0, 1e-8*(t0/2e4)**(-5/3), color='#52514e', ls='--', lw=1, label='t^-5/3')
ax.loglog(t0, 1e-8*(t0/2e4)**(-9/4), color='#52514e', ls=':', lw=1, label='t^-9/4')
ax.set_xlabel('t - t_p  [M]'); ax.set_ylabel('return rate to 3 r_p  [M*/M]'); ax.legend(frameon=False, fontsize=7)
ax.grid(True, which='both', color='#e4e3df', lw=0.5)
fig.tight_layout(); fig.savefig(f'{out}.png', dpi=130)
