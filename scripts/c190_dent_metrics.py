#!/usr/bin/env python3
"""c190 (R-225 item 1): D-ent 1D regression metrics (mk190c.py decks; arms off / leg (production DE) / cnd (conditioning)).
  shocks (cold/, flat/col_M*, flat/c107_*): the shocked slab (rho > 1 + 0.5 (r_RH - 1) times upstream, its middle 60%),
    median K/K0 - 1 at t_end; f_RH = (K - 1)/(K_RH - 1); capture vs off; the slab's branch shares (de_diag 0) at t_end
    (cnd: 11 tracer, 12 safeguard, 13 tracer-floor, 14 resync, 15 energy passive).
  hom (flat/hom_c*): max/min K/K0 - 1 over rho > 0.3 rho_max at t_end.   sod: max |P - P_off|/P_off.
  build (build/lw_*): the entropy excess X(t) = sum rho (K/K0 - 1) dx per dump, arms vs off; the resync share (code 14).
    c190_dent_metrics.py CPUDIR"""
import glob, math, re, sys
import numpy as np
GAM = 5.0/3.0
D = sys.argv[1]


def tabs(d, kind):
    out = []
    for fn in sorted(glob.glob(f'{d}/tab/*{kind}*.tab')):
        t = float(open(fn).readline().split('time=')[1].split()[0])
        out.append((t, np.loadtxt(fn, ndmin=2)))
    return out


def prim(d, k=-1):
    t, a = tabs(d, 'mhd_w_bcc')[k]
    return t, a[:, 2], a[:, 3], a[:, 7]          # t, x, rho, P


def branch(d, k=-1):
    T = tabs(d, 'mhd_de_diag')
    return T[k][1][:, 3] if T else None


def krh(M):
    p = (2*GAM*M**2 - (GAM - 1))/(GAM + 1)
    r = (GAM + 1)*M**2/((GAM - 1)*M**2 + 2)
    return p/r**GAM, r


def slab_K(d, M):
    t, x, rho, P = prim(d)
    t0, x0, rho0, P0 = prim(d, 0)
    K0 = np.median(P0/rho0**GAM); r0 = np.median(rho0)
    _, rr = krh(M)
    idx = np.nonzero(rho > r0*(1 + 0.5*(rr - 1)))[0]
    if len(idx) < 5:
        return np.nan, None
    lo, hi = idx.min(), idx.max(); w = hi - lo
    sel = np.arange(lo + int(0.2*w), hi - int(0.2*w) + 1)
    return np.median(P[sel]/rho[sel]**GAM)/K0 - 1.0, sel


def shares(d, sel):
    b = branch(d)
    if b is None or sel is None:
        return ''
    c = {int(v): int(n) for v, n in zip(*np.unique(b[sel], return_counts=True))}
    return ' '.join(f'{k}:{v}' for k, v in sorted(c.items()))


def shock_rows(pat, getM):
    print(f'{"case":34s} {"K_RH-1":>9s} | {"off":>9s} {"leg":>9s} {"cnd":>9s} | f_RH off/leg/cnd       | cnd slab branches')
    for d_off in sorted(glob.glob(f'{D}/{pat}_off_dent')):
        base = d_off[:-len('_off_dent')]; nm = base.split('/')[-1]; M = getM(nm)
        kr = krh(M)[0] - 1
        v = {}; sl = None
        for a in ('off', 'leg', 'cnd'):
            dd = f'{base}_{a}_dent'
            try:
                v[a], s = slab_K(dd, M)
                if a == 'cnd':
                    sl = shares(dd, s)
            except Exception as e:   # noqa
                v[a] = np.nan
        f = ' / '.join(f'{v[a]/kr:6.3f}' for a in ('off', 'leg', 'cnd'))
        print(f'{nm:34s} {kr:9.3e} | {v["off"]:9.3e} {v["leg"]:9.3e} {v["cnd"]:9.3e} | {f} | {sl}')


print('== cold boosted colliding flows (upstream rho 1, P0; q_ad ~ 1e-4 (P0 3.2e-6) / ~5e-4 (1.6e-5) at vb 0.3)')
shock_rows('cold/cold_*', lambda nm: float(re.search(r'_M([\d.]+)_', nm).group(1)))
print('== C-107 moving shocks (warm, P 0.01)')
shock_rows('flat/c107_*', lambda nm: float(re.search(r'_M([\d.]+)_', nm).group(1)))
print('== flat unit tests: colliding flows')
shock_rows('flat/col_M*', lambda nm: float(re.search(r'_M([\d.]+)', nm).group(1)))
print('== flat unit tests: homologous compression (K/K0 - 1 over rho > 0.3 rho_max at t_end) and Sod')
for c in ('0.05', '0.1', '0.2', '0.4'):
    out = []
    for a in ('off', 'leg', 'cnd'):
        t, x, rho, P = prim(f'{D}/flat/hom_c{c}_{a}_dent'); _, _, r0, P0 = prim(f'{D}/flat/hom_c{c}_{a}_dent', 0)
        K0 = np.median(P0/r0**GAM); dn = rho > 0.3*rho.max(); K = P[dn]/rho[dn]**GAM/K0 - 1
        out.append(f'{a} [{K.min():+.2e}, {K.max():+.2e}]')
    print(f'hom c={c:4s}: ' + '  '.join(out))
_, _, ro, Po = prim(f'{D}/flat/sod_off_dent')
for a in ('leg', 'cnd'):
    _, _, r, P = prim(f'{D}/flat/sod_{a}_dent')
    print(f'sod {a}: max |P - P_off|/P_off {np.max(np.abs(P - Po)/Po):.2e}, max |rho - rho_off|/rho_off {np.max(np.abs(r - ro)/ro):.2e}')
print('== building shock: entropy excess X = sum rho (K/K0 - 1) dx (K0 = the initial median) at t = 100/200/300/400; '
      'cnd resync cells (code 14) / tracer cells (11) at those dumps')
for d_off in sorted(glob.glob(f'{D}/build/lw_*_off_dent')):
    base = d_off[:-len('_off_dent')]; nm = base.split('/')[-1]
    row = {}
    for a in ('off', 'leg', 'cnd'):
        T = tabs(f'{base}_{a}_dent', 'mhd_w_bcc')
        if not T:
            row[a] = None; continue
        a0 = T[0][1]; K0 = np.median(a0[:, 7]/a0[:, 3]**GAM); dx = a0[1, 2] - a0[0, 2]
        row[a] = {round(t): (a1[:, 3]*(a1[:, 7]/a1[:, 3]**GAM/K0 - 1)).sum()*dx for t, a1 in T}
    br = {round(t): (int((a1[:, 3] == 14).sum()), int((a1[:, 3] == 11).sum()))
          for t, a1 in tabs(f'{base}_cnd_dent', 'mhd_de_diag')}
    for tt in (100, 200, 300, 400):
        s = ' '.join(f'{a} {row[a].get(tt, np.nan):+.3e}' if row[a] else f'{a} n/a' for a in ('off', 'leg', 'cnd'))
        print(f'{nm:18s} t={tt:3d}: {s} | cnd 14/11 {br.get(tt)}')
