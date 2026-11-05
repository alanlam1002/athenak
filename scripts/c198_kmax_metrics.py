#!/usr/bin/env python3
"""c198 (user item 5): 1D check of a higher dual_energy_kmax (c198_mk_kmax.py decks; run dirs <grp>/<case>_<tag>).
Per case, against k1 (kmax = 1, the current production value): over every tab dump (mhd_w_bcc), max |rho - rho_k1|/rho_k1
and max |P - P_k1|/P_k1 for k100 / k1000 / k10000 ('bitwise' if all dumps are byte-identical), and the run's
max K/kmax in k1 (the ceiling margin; K = P/rho^gamma). Shock cases (cold, c107, col): f_RH of the shocked slab
(c190_dent_metrics' definition) for each variant, kc10 included (the cold set's binding ceiling, kmax = 10 K0).
    c198_kmax_metrics.py DIR"""
import glob, os, re, sys
import numpy as np
GAM = 5.0/3.0
D = sys.argv[1]
TAGS = ('k1', 'k100', 'k1000', 'k10000', 'kc10')


def tabs(d, kind='mhd_w_bcc'):
    out = []
    for fn in sorted(glob.glob(f'{d}/tab/*{kind}*.tab')):
        out.append((fn, np.loadtxt(fn, ndmin=2)))
    return out


def krh(M):
    p = (2*GAM*M**2 - (GAM - 1))/(GAM + 1); r = (GAM + 1)*M**2/((GAM - 1)*M**2 + 2)
    return p/r**GAM, r


def fRH(d, M):
    T = tabs(d)
    if len(T) < 2:
        return np.nan
    a0, a = T[0][1], T[-1][1]
    rho, P, r0, P0 = a[:, 3], a[:, 7], a0[:, 3], a0[:, 7]
    K0 = np.median(P0/r0**GAM); rr = krh(M)[1]
    idx = np.nonzero(rho > np.median(r0)*(1 + 0.5*(rr - 1)))[0]
    if len(idx) < 5:
        return np.nan
    lo, hi = idx.min(), idx.max(); w = hi - lo; sel = np.arange(lo + int(0.2*w), hi - int(0.2*w) + 1)
    return (np.median(P[sel]/rho[sel]**GAM)/K0 - 1.0)/(krh(M)[0] - 1)


def kmax_of(deck):
    return float(re.search(r'(?m)^dual_energy_kmax\s*=\s*(\S+)', open(deck).read()).group(1))


cases = sorted({re.sub(r'_(k1|k100|k1000|k10000|kc10)$', '', d) for d in glob.glob(f'{D}/*/*_k*') if os.path.isdir(d)})
nb = 0; worst = {}
for c in cases:
    nm = c[len(D) + 1:]
    base = tabs(f'{c}_k1')
    if not base:
        print(f'{nm}: k1 missing'); continue
    kmx = max(np.max(a[:, 7]/a[:, 3]**GAM) for _, a in base)/kmax_of(f'{c}_k1.athinput')
    parts = [f'max K/kmax(k1) {kmx:.2e}']
    for t in ('k100', 'k1000', 'k10000'):
        T = tabs(f'{c}_{t}')
        if len(T) != len(base):
            parts.append(f'{t} dumps {len(T)}/{len(base)}'); continue
        same = all(open(f1, 'rb').read() == open(f2, 'rb').read() for (f1, _), (f2, _) in zip(base, T))
        dr = max(np.max(np.abs(b[:, 3] - a[:, 3])/np.abs(a[:, 3])) for (_, a), (_, b) in zip(base, T))
        dp = max(np.max(np.abs(b[:, 7] - a[:, 7])/np.abs(a[:, 7])) for (_, a), (_, b) in zip(base, T))
        worst[t] = max(worst.get(t, 0.0), dp, dr); nb += same
        parts.append(f'{t} ' + ('bitwise' if same else f'drho {dr:.1e} dP {dp:.1e}'))
    m = re.search(r'_M([\d.]+)', nm)
    if m and ('cold' in nm or 'c107' in nm or 'col_M' in nm):
        M = float(m.group(1))
        parts.append('f_RH ' + ' '.join(f'{t} {fRH(f"{c}_{t}", M):.4f}' for t in TAGS if os.path.isdir(f'{c}_{t}')))
    print(f'{nm:40s} ' + ' | '.join(parts))
print(f'bitwise variant runs: {nb}; worst max relative difference vs k1: ' + ', '.join(f'{t} {v:.2e}' for t, v in worst.items()))
