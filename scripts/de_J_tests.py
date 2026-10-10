#!/usr/bin/env python3
"""T-13 J: dual-energy shock-flag sensors (research R-031 J; src/cfc/DEVELOPMENT.md item 72).
CPU 1D tests at WD-centre conditions (rho0 = 3.17e-4, P0 = 1.82e-8, c_s = 9.8e-3), F
defaults, ppmx + HLLE + FOFC; directory suffix = the sensor variant:

  off       energy only
  jump      item 70: div v <= 0 and |dP|/min(P) > 0.3 to a face neighbour
  jameson   div v <= 0 and psi_P = |P+ - 2P + P-|/(P+ + 2P + P-) > 0.25 (R-031)
  vcurv     div v <= 0 and psi_v = |v+ - 2v + v-|/c_s > 0.05 (plus psi_P > 0.25 clause)
  default   item 72: vcurv 0.05, no psi_P clause, one hysteresis dilation pass (a face
            neighbour of a flagged cell joins if its own sensor > 0.5 x threshold)

Tests:
  hom_c<c>   homologous slab compression, rho = rho0 (1 - (x/R*)^2)^1.5, v = -x/tau, 11
             cells per R* (L4), dx/(tau c_s) = c (R-031's periapsis estimate: 0.10 / 0.15
             / 0.20 at r_p = 16 / 12 / 10), to t = tau/2. This profile is an exact
             homologous (shock-free) solution, so K must stay at K0 where the flag is off.
  col_M<M>   two colliding flows -> two shocks of Mach M; the flag must fire, i.e.
             post-shock K (between the collision point and the shocks) matches energy-only.
  sod        V1 Sod (post-shock K = 3.30).

    de_J_tests.py [--dir DIR]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

GAM = 5.0 / 3.0
VAR = ['off', 'jump', 'jameson', 'vcurv', 'default']
LAB = {'off': 'energy only', 'jump': 'jump |ΔP|/P>0.3 (item 70)', 'jameson': 'Jameson ψ_P>0.25',
       'vcurv': 'ψ_v>0.05 (+ψ_P clause)', 'default': 'ψ_v>0.05 + hysteresis (item 72 default)'}
COL = {'off': 'k', 'jump': '#d95f02', 'jameson': '#7570b3', 'vcurv': '#66a61e', 'default': '#e7298a'}
K0 = 1.8236526547804246e-08 / 3.174214651708447e-4**GAM
MACH = ['1.2', '1.5', '2', '3']
HOM = ['0.05', '0.1', '0.2', '0.4']


def tabs(d):
    out = []
    for fn in sorted(glob.glob(f'{d}/tab/*.tab')):
        with open(fn) as f:
            t = float(f.readline().split('time=')[1].split()[0])
        a = np.loadtxt(fn, ndmin=2)
        out.append(dict(t=t, x=a[:, 2], rho=a[:, 3], u=a[:, 4], P=a[:, 7]))
    return out


def sensors(s):
    rho, P = s['rho'], s['P']
    v = s['u'] / np.sqrt(1 + s['u']**2)
    h = 1 + GAM / (GAM - 1) * P / rho
    cs = np.sqrt(GAM * P / (rho * h))
    n = len(rho)
    c = slice(1, n - 1)
    psip = np.zeros(n); psip[c] = np.abs(P[2:] - 2 * P[c] + P[:-2]) / (P[2:] + 2 * P[c] + P[:-2])
    psiv = np.zeros(n); psiv[c] = np.abs(v[2:] - 2 * v[c] + v[:-2]) / cs[c]
    comp = np.zeros(n); comp[c] = -(v[2:] - v[:-2]) / 2 / cs[c]
    return psip, psiv, comp


def post_shock_K(d):
    st = tabs(d)[-1]
    dd = np.abs(st['x'] - 0.5)
    m = (dd > 0.04) & (dd < 0.10)        # away from the wall heating at x = 0.5
    return np.median(st['P'][m] / st['rho'][m]**GAM) / K0


def k_rh(M):
    p = (2 * GAM * M**2 - (GAM - 1)) / (GAM + 1)
    r = (GAM + 1) * M**2 / ((GAM - 1) * M**2 + 2)
    return p / r**GAM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/lus/flare/projects/CompactBinaryMerger/tlam/'
                    'athenak_run/cfc/dual_energy_tests_J')
    a = ap.parse_args()
    D = a.dir
    out = f'{D}/plots'
    os.makedirs(out, exist_ok=True)
    rows = []
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
    # shocks
    ref = {M: post_shock_K(f'{D}/col_M{M}_off') for M in MACH}
    for M in MACH:
        rows.append(f'col M={M:4s} energy-only post-shock K/K0 {ref[M]:.5f} (Rankine-Hugoniot '
                    f'{k_rh(float(M)):.5f})')
    for v in VAR[1:]:
        cap = [(post_shock_K(f'{D}/col_M{M}_{v}') - 1) / (ref[M] - 1) for M in MACH]
        rows.append(f'col {v:8s} captured fraction of the entropy jump at M = '
                    + ' / '.join(MACH) + ': ' + ' / '.join(f'{x:.2f}' for x in cap))
        ax[0, 0].plot([float(M) for M in MACH], cap, 'o-', color=COL[v], label=LAB[v])
    ax[0, 0].axhline(1, color='0.5', lw=0.8)
    ax[0, 0].set_title('shocks: captured fraction of the entropy jump'); ax[0, 0].set_xlabel('Mach')
    ax[0, 0].legend(fontsize=8)
    # homologous
    for v in VAR:
        for thr, axx in ((0.3, ax[0, 1]), (0.1, ax[0, 2])):
            kmax, kmin = [], []
            for c in HOM:
                fin = tabs(f'{D}/hom_c{c}_{v}')[-1]
                dn = fin['rho'] > thr * fin['rho'].max()
                K = fin['P'][dn] / fin['rho'][dn]**GAM / K0
                kmax.append(K.max() - 1); kmin.append(K.min() - 1)
            axx.plot([float(c) for c in HOM], kmax, 'o-', color=COL[v], label=LAB[v])
            axx.plot([float(c) for c in HOM], kmin, 'v--', color=COL[v])
            rows.append(f'hom {v:8s} rho>{thr}rho_max: K/K0-1 max ' + ' / '.join(f'{x:+.4f}' for x in kmax)
                        + '  min ' + ' / '.join(f'{x:+.4f}' for x in kmin) + '  at c = ' + ' / '.join(HOM))
    for thr, axx in ((0.3, ax[0, 1]), (0.1, ax[0, 2])):
        axx.set_title(f'homologous, t = τ/2: K/K0 − 1 over ρ > {thr} ρ_max (max ●, min ▼)')
        axx.set_xlabel('dx/(τ c_s)'); axx.set_yscale('symlog', linthresh=1e-4)
        axx.axvspan(0.1, 0.2, color='0.9'); axx.legend(fontsize=7)
    # sensor distributions in the homologous runs (interior) and at the M = 1.2 shock
    for c in HOM:
        T = tabs(f'{D}/hom_c{c}_off')
        mx = np.zeros(3)
        for st in T:
            dn = st['rho'] > 0.3 * st['rho'].max()
            mx = np.maximum(mx, [x[dn].max() for x in sensors(st)])
        rows.append(f'hom c={c:4s} energy-only, rho>0.3rho_max, max over t: psi_P {mx[0]:.3f}'
                    f'  psi_v {mx[1]:.3f}  -div v dx/c_s {mx[2]:.3f}')
    for M in MACH:
        st = tabs(f'{D}/col_M{M}_off')[-1]
        ps = sensors(st)
        rows.append(f'col M={M:4s} energy-only, max at the shocks: psi_P {ps[0].max():.3f}'
                    f'  psi_v {ps[1].max():.3f}  -div v dx/c_s {ps[2].max():.3f}')
    for s, axx, lab in ((f'{D}/hom_c0.2_off', ax[1, 0], 'homologous c = 0.2, t = τ/4'),
                        (f'{D}/col_M1.5_off', ax[1, 1], 'collision M = 1.5, end')):
        T = tabs(s)
        st = T[len(T) // 2] if 'hom' in s else T[-1]
        ps = sensors(st)
        for y, l in zip(ps, ('ψ_P', 'ψ_v = |Δ²v|/c_s', '−∇·v dx/c_s')):
            axx.semilogy(st['x'], np.maximum(y, 1e-8), label=l)
        axx2 = axx.twinx(); axx2.plot(st['x'], st['rho'] / st['rho'].max(), 'k:', lw=0.8)
        axx.axhline(0.05, color='0.6', ls=':', lw=0.8); axx.axhline(0.25, color='0.6', ls='--', lw=0.8)
        axx.set_title(lab + ' (energy only; dotted ρ/ρ_max)'); axx.legend(fontsize=8)
        axx.set_ylim(1e-6, 2)
    for v in VAR:
        st = tabs(f'{D}/sod_{v}')[-1]
        ax[1, 2].plot(st['x'], st['P'] / st['rho']**GAM, color=COL[v], label=LAB[v], lw=1.2)
        ps = (st['rho'] > 0.2) & (st['rho'] < 0.3) & (st['x'] > 0.7)
        rows.append(f'sod {v:8s} post-shock K {np.median(st["P"][ps] / st["rho"][ps]**GAM):.4f}')
    ax[1, 2].set_title('Sod t = 0.4: K = P/ρ^Γ'); ax[1, 2].legend(fontsize=7); ax[1, 2].set_ylim(0.8, 3.6)
    for x in ax.flat:
        x.grid(alpha=0.3)
    fig.suptitle('T-13 J: dual-energy shock flag (CPU 1D, WD-centre state, 11 cells/R*)')
    fig.savefig(f'{out}/de_J_tests.png', dpi=100)
    with open(f'{out}/de_J_table.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
