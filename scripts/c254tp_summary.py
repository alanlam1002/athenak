#!/usr/bin/env python3
"""c254 (R-254 items 2-3): robust return-rate metrics from c254tp_integrate.py runs (the single late power law of c254tp_ana.py
mixes the post-peak decline, the cutoff at the inner edge of the E ~ 0 gap and the sparse floor beyond it).
Per run (t measured from t_p; Mdot in M*/M; 20 log bins per decade):
  t_peak, Mdot_peak (peak bin);
  s_decl = the log-log slope from t_peak to the last bin before Mdot < 0.3 Mdot_peak (the post-peak, pre-cutoff decline, vs -5/3);
  t_cut  = the first time after the peak with Mdot < 1e-2 Mdot_peak (the gap edge);
  s_floor, N_floor = the Poisson maximum-likelihood power-law slope of the return times beyond 1.3 t_cut (to t_end), and the number of
           particles it rests on (vs -9/4; Coughlin & Nixon 2019);
  t50, t90, t99 = the times by which 50/90/99% of the mass returned by t_end has returned; M_ret, M_floor (beyond 1.3 t_cut) in M*.
    c254tp_summary.py OUT.csv LABEL=RUN.npz [...]"""
import sys
import numpy as np
from scipy.optimize import minimize_scalar

MS = 7.47e-5
out = sys.argv[1]; rows = []
for lab, f in (a.split('=', 1) for a in sys.argv[2:]):
    R = np.load(f)
    tp, tmb, W = float(R['tp']), float(R['tmb_ref']), float(R['W'])
    tr = np.sort(R['t_ev'][R['status'] == 1] - tp)
    t_end = 10*tmb
    edges = np.logspace(np.log10(max(tr[0]*0.95, 50)), np.log10(t_end), 21*int(np.ceil(np.log10(t_end/max(tr[0]*0.95, 50)))))
    cnt, _ = np.histogram(tr, bins=edges); md = cnt*W/np.diff(edges)/MS; tc = np.sqrt(edges[1:]*edges[:-1])
    ip = int(np.argmax(md)); tpk, mpk = tc[ip], md[ip]
    j = ip
    while j + 1 < len(md) and md[j + 1] >= 0.3*mpk:
        j += 1
    s_decl = np.polyfit(np.log(tc[ip:j + 1]), np.log(np.maximum(md[ip:j + 1], 1e-300)), 1)[0] if j > ip + 1 else np.nan
    k = ip
    while k < len(md) and md[k] >= 1e-2*mpk:
        k += 1
    t_cut = tc[k] if k < len(md) else np.nan
    fl = tr[tr > 1.3*t_cut] if np.isfinite(t_cut) else np.array([])
    a0, b0 = (1.3*t_cut, t_end)
    if len(fl) >= 5:
        lx = np.log(fl)
        def nll(s):   # pdf ~ t^s on [a0, b0]
            if abs(s + 1) < 1e-9:
                Z = np.log(b0/a0)
            else:
                Z = (b0**(s + 1) - a0**(s + 1))/(s + 1)
            return -(s*lx.sum() - len(fl)*np.log(Z))
        s_floor = minimize_scalar(nll, bounds=(-8, 0), method='bounded').x
    else:
        s_floor = np.nan
    cm = np.arange(1, len(tr) + 1)/len(tr)
    q = [tr[np.searchsorted(cm, x)] for x in (0.5, 0.9, 0.99)]
    rows.append(dict(run=lab, soft=str(R['soft']), N=int(R['N']), seed=int(R['seed']), t0_minus_tp=float(R['t0']) - tp,
                     t_peak=tpk, t_peak_tmb=tpk/tmb, Mdot_peak=mpk, s_decl=s_decl, t_cut=t_cut, t_cut_tmb=t_cut/tmb,
                     s_floor=s_floor, N_floor=len(fl), t50=q[0], t90=q[1], t99=q[2], M_ret=W*len(tr)/MS,
                     M_floor=W*len(fl)/MS, M_remnant=W*np.sum(R['status'] == 2)/MS))
keys = list(rows[0])
with open(out, 'w') as fo:
    fo.write(','.join(keys) + '\n')
    for r in rows:
        fo.write(','.join(f'{r[k]:.5g}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
print(f"{'run':18s} {'N':>6s} {'t_pk':>6s} {'/tmb':>5s} {'Mdot_pk':>9s} {'s_decl':>6s} {'t_cut':>7s} {'/tmb':>5s} {'s_floor':>7s} "
      f"{'N_fl':>5s} {'t50':>6s} {'t90':>6s} {'t99':>7s} {'M_ret':>6s} {'M_floor':>8s} {'M_rem':>8s}")
for r in rows:
    print(f"{r['run']:18s} {r['N']:6d} {r['t_peak']:6.0f} {r['t_peak_tmb']:5.2f} {r['Mdot_peak']:9.3e} {r['s_decl']:6.2f} "
          f"{r['t_cut']:7.0f} {r['t_cut_tmb']:5.2f} {r['s_floor']:7.2f} {r['N_floor']:5d} {r['t50']:6.0f} {r['t90']:6.0f} "
          f"{r['t99']:7.0f} {r['M_ret']:6.4f} {r['M_floor']:8.2e} {r['M_remnant']:8.1e}")
