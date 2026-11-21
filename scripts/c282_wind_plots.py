#!/usr/bin/env python3
"""c282 (R-268 arm 1 to t = 92; user question 2026-11-21): the wind and the envelope around the star, WENOZ W32 vs PPMX L32 / L64.
    c282_wind_plots.py ANADIR
Inputs (ANADIR): outflow_{W32,L32}.csv (c206_outflow.py), envelope_{W32,L32,L64}.csv (c282_envelope.py),
coldcore_{W32,L32,L64}.csv (c247_coldcore.py); references c249_ana/outflow_L64_P18.csv and the 1/16 pair c276_wenoz/ana/outflow_A1{P,W}.csv.
Writes c282_wind.png and c282_wind_table.txt (per dump: the R-268 measure |X(W32) - X(L64)| vs |X(L32) - X(L64)|)."""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = sys.argv[1]
C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
ST = {'W32': ('#d9731a', '-', 'o', 'WENOZ 1/32 (W32)'), 'L32': ('#1f5fa8', '-', 's', 'PPMX 1/32 (L32)'),
      'L64': ('k', '--', '^', 'PPMX 1/64 (L64, reference)'), 'A1W': ('#d9731a', ':', None, 'WENOZ 1/16'), 'A1P': ('#1f5fa8', ':', None, 'PPMX 1/16')}
TMAX = 93.0


def rd(f):
    return list(csv.DictReader(open(f))) if os.path.exists(f) else []


def col(rows, k):
    out = []
    for r in rows:
        try: out.append(float(r[k]))
        except (ValueError, KeyError, TypeError): out.append(np.nan)
    return np.array(out)


OUT = {'W32': rd(f'{A}/outflow_W32.csv'), 'L32': rd(f'{A}/outflow_L32.csv'), 'L64': rd(f'{C}/c249_ana/outflow_L64_P18.csv'),
       'A1W': rd(f'{C}/c276_wenoz/ana/outflow_A1W.csv'), 'A1P': rd(f'{C}/c276_wenoz/ana/outflow_A1P.csv')}
ENV = {k: rd(f'{A}/envelope_{k}.csv') for k in ('W32', 'L32', 'L64')}
CC = {k: rd(f'{A}/coldcore_{k}.csv') for k in ('W32', 'L32', 'L64')}


def env(k, shell, band, q='M_over_Mstar'):
    r = [x for x in ENV[k] if x['shell'] == shell and x['band'] == band]
    return col(r, 't'), col(r, q)


fig, ax = plt.subplots(3, 3, figsize=(17, 14))
for k, rows in OUT.items():
    if not rows: continue
    c, ls, mk, lab = ST[k]; t = col(rows, 't'); m = t <= TMAX
    for j, q in enumerate(('M_gt1.5', 'K15')):
        ax[0, j].plot(t[m], col(rows, q)[m], ls, marker=mk, ms=4, color=c, label=lab)
for k, rows in CC.items():
    if not rows: continue
    c, ls, mk, lab = ST[k]; t = col(rows, 't'); m = t <= TMAX
    ax[0, 2].plot(t[m], col(rows, 'core_KP')[m], ls, marker=mk, ms=4, color=c, label=f'{lab} core < 0.2 R*')
    ax[0, 2].plot(t[m], col(rows, 's58_KP')[m], ls, marker=mk, ms=4, mfc='none', color=c, alpha=0.7, label=f'{lab} shell 0.5-0.8 R*')
BANDS = [('ge1e-7', '>= 1e-7 rho_c (wind cut)'), ('1e-12_1e-7', '1e-12..1e-7'), ('1e-18_1e-12', '1e-18..1e-12'), ('lt1e-18', '< 1e-18 (floor)')]
for i, (shell, row) in enumerate((('1-1.5', 1), ('1.5-3', 1), ('3-6', 1))):
    a = ax[1, i]
    for k in ('W32', 'L32', 'L64'):
        if not ENV[k]: continue
        c, ls, mk, lab = ST[k]
        for (b, bl), bm in zip(BANDS, ('o', 's', '^', 'x')):
            t, M = env(k, shell, b)
            a.semilogy(t, np.maximum(M, 1e-30), ls, marker=bm, ms=4, color=c, alpha=0.9, label=f'{k} {bl}')
    a.set_title(f'mass by density band, d = {shell} R*', fontsize=10); a.set_ylim(1e-14, 1e-1)
for i, shell in enumerate(('1-1.5', '1.5-3', '3-6')):
    a = ax[2, i]
    for k in ('W32', 'L32', 'L64'):
        if not ENV[k]: continue
        c, ls, mk, lab = ST[k]
        t, M = env(k, shell, 'hot_Tgt1e-3')
        a.semilogy(t, np.maximum(M, 1e-30), ls, marker=mk, ms=4, color=c, label=f'{lab}: hot mass (P/rho > 1e-3)')
    a.set_title(f'hot gas (P/rho > 1e-3, any density), d = {shell} R*', fontsize=10); a.set_ylim(1e-22, 1e-6)
for (i, j), (xl, yl, ti, lg) in {(0, 0): ('t [M]', 'M(d > 1.5 R*, rho > 1e-7 rho_c) / M*', 'R-268 wind: lifted surface mass', True),
                                 (0, 1): ('t [M]', 'mass-weighted K/K0 at d > 1.5 R*', 'wind heat', True),
                                 (0, 2): ('t [M]', 'K_P / K0', 'core and outer shell heat', False),
                                 (1, 0): ('t [M]', 'M / M*', None, True), (1, 1): ('t [M]', 'M / M*', None, True),
                                 (1, 2): ('t [M]', 'M / M*', None, True),
                                 (2, 0): ('t [M]', 'M / M*', None, True), (2, 1): ('t [M]', 'M / M*', None, True),
                                 (2, 2): ('t [M]', 'M / M*', None, True)}.items():
    a = ax[i, j]; a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.3); a.set_xlim(-2, TMAX)
    if ti: a.set_title(ti, fontsize=10)
    if lg and i == 0: a.set_yscale('log')
    a.legend(fontsize=6, ncol=1 if i != 1 else 3)
fig.suptitle('c282: wind and envelope around the star, WENOZ W32 vs PPMX L32 and L64 (1/16 pair dotted), t <= 92')
fig.tight_layout(); fig.savefig(f'{A}/c282_wind.png', dpi=75); plt.close(fig)

# the R-268 measure per matched dump time
lines = ['quantity  t  W32  L32  L64  |W32-L64|  |L32-L64|  closer']
def meas(name, kind, getter):
    tw, xw = getter('W32'); tl, xl = getter('L32'); t6, x6 = getter('L64')
    for t, w in zip(tw, xw):
        if t > TMAX or not np.isfinite(w): continue
        i = np.argmin(abs(tl - t)) if len(tl) else None; j = np.argmin(abs(t6 - t)) if len(t6) else None
        if i is None or j is None or abs(tl[i] - t) > 1.0 or abs(t6[j] - t) > 1.0: continue
        p, r = xl[i], x6[j]; dw, dp = abs(w - r), abs(p - r)
        lines.append(f'{name:28s} {t:6.1f} {w:10.3e} {p:10.3e} {r:10.3e} {dw:10.3e} {dp:10.3e}  {"WENOZ" if dw < dp else "PPMX"}')
g_out = lambda q: (lambda k: (col(OUT[k], 't'), col(OUT[k], q)))
g_cc = lambda q: (lambda k: (col(CC[k], 't'), col(CC[k], q)))
g_env = lambda s, b, q='M_over_Mstar': (lambda k: env(k, s, b, q))
meas('wind M(>1.5R*)', 0, g_out('M_gt1.5')); meas('wind K15', 0, g_out('K15'))
meas('core K_P', 0, g_cc('core_KP')); meas('shell0.5-0.8 K_P', 0, g_cc('s58_KP'))
for s in ('1-1.5', '1.5-3', '3-6'):
    meas(f'hot M d={s}', 0, g_env(s, 'hot_Tgt1e-3'))
    for b, _ in BANDS:
        meas(f'M d={s} {b}', 0, g_env(s, b))
open(f'{A}/c282_wind_table.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
