#!/usr/bin/env python3
"""c289 (R-268 arm 1): WENOZ 1/32 (W32) vs PPMX 1/32 (L32), against the PPMX 1/64 reference (L64), from ID to C1 (t_p + 50).
    c289_r268.py ANADIR
Inputs (ANADIR, from c289.pbs and run_wind.sh): outflow_{W32,L32,L64}.csv, envelope_*.csv, coldcore_*.csv (every 12 M),
stream_{W32,L32,L64}_transverse.csv (c253, the dump at t - t_p ~ 49), dmde_*.npz (tde_dmde at C1).
The R-268 measure per quantity and dump: dW = |X(W32) - X(L64)| against dP = |X(L32) - X(L64)|; 'closer' = the smaller.
Writes c289_r268.png and c289_r268_table.txt."""
import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = sys.argv[1]
RUNS = ('W32', 'L32', 'L64')
ST = {'W32': ('#d9731a', '-', 'o', 'WENOZ 1/32 (W32)'), 'L32': ('#1f5fa8', '-', 's', 'PPMX 1/32 (L32)'),
      'L64': ('k', '--', '^', 'PPMX 1/64 (L64, reference)')}


def rd(f):
    return list(csv.DictReader(open(f))) if os.path.exists(f) else []


def series(run, kind, key, shell=None, band=None):
    rows = rd(f'{A}/{kind}_{run}.csv')
    if shell is not None:
        rows = [r for r in rows if r['shell'] == shell and r['band'] == band]
    t, x = [], []
    for r in rows:
        try:
            tv, xv = float(r['t']), float(r[key])
        except (ValueError, KeyError, TypeError):
            continue
        t.append(tv); x.append(xv)
    o = np.argsort(t)
    return np.array(t)[o], np.array(x)[o]


def at(t, x, tm, tol=1.0):
    if len(t) == 0: return np.nan
    i = np.argmin(abs(t - tm))
    return x[i] if abs(t[i] - tm) < tol else np.nan


Q = [('wind M(>1.5R*)', ('outflow', 'M_gt1.5', None, None)), ('wind K/K0 (d>1.5R*)', ('outflow', 'K15', None, None)),
     ('core K_P/K0', ('coldcore', 'core_KP', None, None)), ('shell 0.5-0.8 R* K_P/K0', ('coldcore', 's58_KP', None, None))]
for sh in ('1-1.5', '1.5-3', '3-6'):
    for b in ('ge1e-7', '1e-12_1e-7'):
        Q.append((f'M d={sh} {b}', ('envelope', 'M_over_Mstar', sh, b)))

lines = ['quantity                         t      W32          L32          L64          |W32-L64|    |L32-L64|    closer']
summ = []
for name, (kind, key, sh, b) in Q:
    S = {r: series(r, kind, key, sh, b) for r in RUNS}
    t6, x6 = S['L64']
    nw = npp = 0
    for i, tm in enumerate(t6):
        if tm < 11.0: continue
        w = at(*S['W32'], tm); p = at(*S['L32'], tm); r = x6[i]
        if not (np.isfinite(w) and np.isfinite(p) and np.isfinite(r)): continue
        dw, dp = abs(w - r), abs(p - r)
        c = 'WENOZ' if dw < dp else ('PPMX' if dp < dw else 'tie')
        nw += c == 'WENOZ'; npp += c == 'PPMX'
        lines.append(f'{name:32s} {tm:6.1f} {w:12.5e} {p:12.5e} {r:12.5e} {dw:12.4e} {dp:12.4e} {c}')
    summ.append(f'{name:32s} dumps closer: WENOZ {nw}, PPMX {npp}')

# the early stream at C1 (one dump each, t - t_p ~ 49)
st = {r: rd(f'{A}/stream3_{r}_transverse.csv') for r in RUNS}   # c253 at C1: radii 18, 21, excl 2 (all four runs have the stream there)
for R in ('18.0', '21.0'):
    for key in ('M', 'fwhm_z', 'fwhm_ip', 'rho_pk', 'K'):
        v = {}
        for r in RUNS:
            rows = [x for x in st[r] if x.get('R') == R]
            try:
                v[r] = float(rows[-1][key]) if rows else np.nan
            except (KeyError, ValueError):
                v[r] = np.nan
        if all(np.isfinite(list(v.values()))):
            dw, dp = abs(v['W32'] - v['L64']), abs(v['L32'] - v['L64'])
            c = 'WENOZ' if dw < dp else ('PPMX' if dp < dw else 'tie')
            lines.append(f'stream R={R} {key:28s} C1 {v["W32"]:12.5e} {v["L32"]:12.5e} {v["L64"]:12.5e} {dw:12.4e} {dp:12.4e} {c}')
            summ.append(f'stream R={R} {key}: closer {c}')

# dM/dE at C1
def dmde(r):
    f = f'{A}/dmde_{r}.npz'
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}
D = {r: dmde(r) for r in RUNS}
if all(D.values()):
    def quant(d):
        # tde_dmde.py's npz: edges (E/dE bin edges), H_0 (mass per bin), fb_0 (bound fraction, E < 0)
        if 'H_0' not in d or 'edges' not in d: return None
        ed = np.asarray(d['edges']); h = np.asarray(d['H_0'], float); c = np.cumsum(h)/h.sum()
        q = [float(np.interp(p, c, ed[1:])) for p in (0.1, 0.5, 0.9)]
        return dict(f_bound=float(d['fb_0']), q10=q[0], q50=q[1], q90=q[2])
    Qd = {r: quant(D[r]) for r in RUNS}
    if all(Qd.values()):
        for key in ('f_bound', 'q10', 'q50', 'q90'):
            w, p, r = Qd['W32'][key], Qd['L32'][key], Qd['L64'][key]
            dw, dp = abs(w - r), abs(p - r); c = 'WENOZ' if dw < dp else ('PPMX' if dp < dw else 'tie')
            lines.append(f'dM/dE at C1 {key:30s} {w:12.5e} {p:12.5e} {r:12.5e} {dw:12.4e} {dp:12.4e} {c}')
            summ.append(f'dM/dE {key}: closer {c}')
    else:
        summ.append('dM/dE: npz keys not recognised: ' + ', '.join(sorted(D['W32'].keys())))
open(f'{A}/c289_r268_table.txt', 'w').write('SUMMARY\n' + '\n'.join(summ) + '\n\n' + '\n'.join(lines) + '\n')
print('\n'.join(summ))

fig, ax = plt.subplots(3, 4, figsize=(20, 13))
for k, (name, (kind, key, sh, b)) in enumerate(Q[:12]):
    a = ax.flat[k]
    for r in RUNS:
        t, x = series(r, kind, key, sh, b)
        if len(t) == 0: continue
        c, ls, mk, lab = ST[r]
        if 'K_P' in name or 'K/K0' in name:
            a.plot(t, x, ls, marker=mk, ms=3, color=c, label=lab)
        else:
            a.semilogy(t, np.maximum(x, 1e-30), ls, marker=mk, ms=3, color=c, label=lab)
    a.set_title(name, fontsize=10); a.set_xlabel('t [M]'); a.grid(alpha=0.3); a.axvline(491.1, color='0.6', lw=0.8)
    if not ('K_P' in name or 'K/K0' in name): a.set_ylim(bottom=1e-14)
    a.legend(fontsize=7)
fig.suptitle('c289 (R-268 arm 1): WENOZ vs PPMX at dx 1/32 against the PPMX 1/64 reference, ID -> C1 (grey: t_p = 491.1)')
fig.tight_layout(); fig.savefig(f'{A}/c289_r268.png', dpi=70); plt.close(fig)
