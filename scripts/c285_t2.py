#!/usr/bin/env python3
"""c285 test 2 (R-271 addendum): is the gas above the ceiling band unchanged? W32C (WENOZ + atm_ceiling) vs W32 (WENOZ, no ceiling)
to t = 200, with PPMX L32 (and L32C) for scale.
    c285_t2.py ANADIR
Inputs (ANADIR): envelope_{W32C,W32,L32,L32C}.csv (c282_envelope.py), outflow_*.csv (c206_outflow.py), coldcore_*.csv (c247_coldcore.py).
Tolerances, stated in C-254 before the result:
  envelope bands above the threshold (rho >= 1e-18 rho_c; the band itself is below 3.1e-19) and the wind M(> 1.5 R*):
      d = |X(W32C) - X(W32)| must lie within W32's own dump-to-dump change s = |X_W32(t) - X_W32(t - 12)| and, where it is smaller,
      within |X(W32) - X(L32)|:  tol(t) = min(s, |W32 - L32|) if that is > 0, else s
  core and 0.5-0.8 R* shell K_P/K0: |d| <= 1e-4
Per quantity: the dumps compared (t >= 12), the number within tolerance, and max d/tol. Writes c285_t2.png and c285_t2_table.txt."""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = sys.argv[1]
RUNS = ('W32C', 'W32', 'L32', 'L32C')
ST = {'W32C': ('#d9731a', '-', 'o', 'WENOZ + ceiling (W32C)'), 'W32': ('#8c3b00', '--', 's', 'WENOZ (W32)'),
      'L32': ('#1f5fa8', '--', '^', 'PPMX (L32)'), 'L32C': ('#5aa0e0', '-', 'v', 'PPMX + ceiling (L32C)')}


def rd(f):
    return list(csv.DictReader(open(f))) if os.path.exists(f) else []


def series(run, kind, key, shell=None, band=None):
    rows = rd(f'{A}/{kind}_{run}.csv')
    if shell is not None:
        rows = [r for r in rows if r['shell'] == shell and r['band'] == band]
    t, x = [], []
    for r in rows:
        try:
            t.append(float(r['t'])); x.append(float(r[key]))
        except (ValueError, KeyError):
            pass
    o = np.argsort(t)
    return np.array(t)[o], np.array(x)[o]


def at(t, x, tm):
    if len(t) == 0: return np.nan
    i = np.argmin(abs(t - tm))
    return x[i] if abs(t[i] - tm) < 1.0 else np.nan


Q = []
for sh in ('1-1.5', '1.5-3', '3-6'):
    for b in ('ge1e-7', '1e-12_1e-7', '1e-18_1e-12'):
        Q.append((f'M d={sh} {b}', ('envelope', 'M_over_Mstar', sh, b), 'scatter'))
Q.append(('wind M(>1.5R*)', ('outflow', 'M_gt1.5', None, None), 'scatter'))
Q.append(('core K_P/K0', ('coldcore', 'core_KP', None, None), 'abs1e-4'))
Q.append(('shell 0.5-0.8 K_P/K0', ('coldcore', 's58_KP', None, None), 'abs1e-4'))

lines = ['quantity                         n_dumps  n_within  max d/tol   (d = |W32C - W32|; tol as in the docstring)']
detail = []
for name, (kind, key, sh, b), rule in Q:
    S = {r: series(r, kind, key, sh, b) for r in RUNS}
    tw, xw = S['W32']; tc, xc = S['W32C']; tl, xl = S['L32']
    n = ok = 0; mx = 0.0
    for i, tm in enumerate(tw):
        if tm < 11.0 or tm > 200.5: continue
        c = at(tc, xc, tm); l = at(tl, xl, tm)
        if not np.isfinite(c): continue
        d = abs(c - xw[i])
        if rule == 'abs1e-4':
            tol = 1e-4
        else:
            s = abs(xw[i] - xw[i-1]) if i > 0 else np.nan
            wl = abs(xw[i] - l) if np.isfinite(l) else np.inf
            tol = min(s, wl) if min(s, wl) > 0 else s
        n += 1
        if tol > 0 and np.isfinite(tol):
            r = d/tol; mx = max(mx, r); ok += (r <= 1.0)
        else:
            ok += (d == 0.0); mx = max(mx, 0.0 if d == 0 else np.inf)
        detail.append(f'{name:32s} t={tm:6.1f} W32={xw[i]:.4e} W32C={c:.4e} L32={l:.4e} d={d:.3e} tol={tol:.3e}')
    lines.append(f'{name:32s} {n:7d}  {ok:8d}  {mx:9.3g}')
open(f'{A}/c285_t2_table.txt', 'w').write('\n'.join(lines) + '\n\n' + '\n'.join(detail) + '\n')
print('\n'.join(lines))

fig, ax = plt.subplots(3, 4, figsize=(20, 13))
k = 0
for name, (kind, key, sh, b), rule in Q:
    a = ax.flat[k]; k += 1
    for r in RUNS:
        t, x = series(r, kind, key, sh, b)
        if len(t) == 0: continue
        c, ls, mk, lab = ST[r]
        if rule == 'abs1e-4':
            a.plot(t, x, ls, marker=mk, ms=3, color=c, label=lab)
        else:
            a.semilogy(t, np.maximum(x, 1e-30), ls, marker=mk, ms=3, color=c, label=lab)
    a.set_title(name, fontsize=10); a.set_xlabel('t [M]'); a.grid(alpha=0.3); a.set_xlim(-2, 202)
    if rule != 'abs1e-4': a.set_ylim(bottom=1e-16)
    a.legend(fontsize=6)
fig.suptitle('c285 test 2: the gas above the ceiling band, WENOZ + ceiling vs WENOZ (and PPMX +/- ceiling), t <= 200')
fig.tight_layout(); fig.savefig(f'{A}/c285_t2.png', dpi=70); plt.close(fig)
