#!/usr/bin/env python3
"""C-047 (R-057 item 1): offline two-tier extension of the C-046 S0 grid-rest what-if.

Variants (r_s = min(6, 0.5 r_p); every modified cell gets v = beta'/alpha capped at |v| <= 0.9
along beta', cold P, as in c046_s0.py):
  single : r < r_s, rho < 1e-10 rho_c, rho reset (C-046's zone; mass removed).
  two-tier (r_in, rho_shell):
    sink   r < r_in in {1.9, 2.9}: rho < 1e-10 rho_c, rho reset (mass removed);
    shell  r_in <= r < r_s: rho < rho_shell in {1e-16, 1e-14, 1e-12} rho_c, velocity only
           (mass kept, velocity modified);
    tier3  (optional, '+t3') r_s <= r < 2 r_s: as the shell, excluding cells within 3 R* of
           the star's centroid (mass-weighted centroid of rho > 1e-3 rho_max).
  shell-only reference (r_in = 0), not requested: does the sink matter at all?

Per restart and variant: dt (gain over the baseline), the limiting cell (where, r_BH, rho,
|lambda| = |alpha-term - shift|), and the removed / velocity-modified mass split by radius
(r < 1.9 / 1.9-2.9 / >= 2.9 M; areal R = 3 M / 4 M sit at isotropic r = 1.87 / 2.92, measured)
and binned in the BH-frame Killing E/dE and in |L| (specific, about x_BH), each E bin also
as a fraction of the restart's total dM/dE mass in that bin (r_BH >= 3, as tde_dmde.py).

    c047_s0b.py [--out DIR] [--only R6]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import c046_s0 as S                               # noqa: E402
from c045_dt import rst_at, logged_dt, RHOC, CFL  # noqa: E402

MSTAR, RSTAR, DE = 7.47e-5, 0.677, 0.677 / 16.4**2
CASES = (('R6', 'R6_rp10_comov_L4_fix,R6x_rp10_comov_L4_t300', 5.0, (129.6, 144.8, 200.0, 300.0)),
         ('R7', 'R7_rp18_comov_L4_fix', 6.0, (137.4, 300.0)))
THR, VMAX = 1e-10, 0.9
EB = np.array([-np.inf, -6, -3, -2, -1, 0, 1, 2, 3, 6, np.inf])
LB = np.array([0, 2, 4, 6, 10, np.inf])
RB = np.array([0, 1.9, 2.9, np.inf])
S.RCAND = 12.0                                    # tier 3 reaches 2 r_s = 12 for rp18


def variants(rs):
    out = [('single', dict(sink=rs, shell=None, t3=False))]
    for rin in (1.9, 2.9):
        for rsh in (1e-16, 1e-14, 1e-12):
            for t3 in (False, True):
                out.append((f'rin {rin} shell {rsh:.0e}' + (' +t3' if t3 else ''), dict(sink=rin, shell=rsh, t3=t3)))
    for rsh in (1e-16, 1e-12):
        out.append((f'ref: shell only {rsh:.0e} +t3', dict(sink=0.0, shell=rsh, t3=True)))
    return out


def masks(Cc, rs, v, xstar):
    r, rho, exc = Cc['r'], Cc['rho'], Cc['exc']
    sink = (r < v['sink']) & (rho < THR * RHOC) & ~exc
    shell = np.zeros_like(sink)
    if v['shell'] is not None:
        shell = (r >= v['sink']) & (r < rs) & (rho < v['shell'] * RHOC) & ~exc
        if v['t3']:
            far = np.sqrt((Cc['X'] - xstar[0])**2 + (Cc['Y'] - xstar[1])**2 + (Cc['Z'] - xstar[2])**2) > 3 * RSTAR
            shell |= (r >= rs) & (r < 2 * rs) & (rho < v['shell'] * RHOC) & far & ~exc
    return sink, shell


def dt_with(B, Cc, mod):
    """CFL dt with the grid-rest velocity (capped, cold) in the cells `mod`; the limiting cell"""
    z = {k: val[mod] for k, val in Cc.items()}
    beta = [z['b0'], z['b1'], z['b2']]
    gd = [[z['g00'], z['g01'], z['g02']], [z['g01'], z['g11'], z['g12']], [z['g02'], z['g12'], z['g22']]]
    bn = np.sqrt(np.maximum(sum(gd[i][j] * beta[i] * beta[j] for i in range(3) for j in range(3)), 0))
    vb = bn / z['alpha']
    f = np.minimum(1.0, VMAX / np.maximum(vb, 1e-300))
    vv = [f * beta[i] / z['alpha'] for i in range(3)]
    lam, _, at, bt = S.lam_hydro(z['alpha'], beta, vv, [z['gi0'], z['gi1'], z['gi2']], np.minimum((f * vb)**2, 0.99),
                                 np.zeros_like(vb))
    dtz = CFL * z['dx'] / np.maximum(lam, 1e-300)
    dtc = Cc['dt'].copy(); dtc[mod] = dtz
    ncb = np.where(B['r'] >= S.RCAND, B['dt'], np.inf)
    i_nc = int(np.argmin(ncb))
    j = int(np.argmin(dtc))
    if dtc[j] < ncb[i_nc]:
        k = Cc['bidx'][j]
        if mod[j]:
            jj = int(np.nonzero(mod)[0].searchsorted(j))
            L = dict(where='modified (capped)' if f[jj] < 1 else 'modified', lam=lam[jj], at=at[jj], bt=bt[jj])
        else:
            L = dict(where='near BH, unmodified', lam=B['lam'][k], at=B['at'][k], bt=B['bt'][k])
        L.update(r=Cc['r'][j], rho=Cc['rho'][j] / RHOC, lev=Cc['lev'][j], dt=dtc[j])
    else:
        L = dict(where=f'r_BH >= {S.RCAND:g}', r=B['r'][i_nc], rho=B['rho'][i_nc] / RHOC, lev=B['lev'][i_nc],
                 lam=B['lam'][i_nc], at=B['at'][i_nc], bt=B['bt'][i_nc], dt=B['dt'][i_nc])
    return L


def binned(m, Cc, ref):
    """mass [M*] split by radius; E/dE bins (with the fraction of the grid's bin mass); |L| bins"""
    dM = Cc['dM'] * m
    rr = np.histogram(Cc['r'], RB, weights=dM)[0] / MSTAR
    eb = np.histogram(Cc['E'] / DE, EB, weights=dM)[0] / MSTAR
    lb = np.histogram(Cc['L'], LB, weights=dM)[0] / MSTAR
    fr = np.where(ref > 0, eb / np.maximum(ref, 1e-300), np.nan)
    return rr, eb, fr, lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{S.C}/plots/phase2_m1')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    rows = ['R-057 item 1: two-tier S0 extension (c047_s0b.py). Masses in M*. Radius split r < 1.9 / 1.9-2.9 / >= 2.9 M.',
            'E bins (E/dE) edges ' + ' '.join(f'{e:g}' for e in EB) + '; |L| bins edges ' + ' '.join(f'{e:g}' for e in LB) + ' [M].',
            '"frac" = modified mass / total grid mass (r_BH >= 3) in the same E bin.']
    for lab, d, rs, ts in CASES:
        if a.only and lab not in a.only.split(','):
            continue
        for tt in ts:
            fn = rst_at(d, tt)
            t, B, Cc = S.load(fn)
            c = D.cells(fn, rcut=0.0)
            assert c['dM'].size == B['r'].size
            cand = B['r'] < S.RCAND
            assert cand.sum() == Cc['r'].size and np.allclose(c['dM'][cand], Cc['dM'])
            xb = np.asarray(c['xbh'])
            X, Y, Z = c['x'] - xb[0], c['y'] - xb[1], c['z'] - xb[2]
            Lv = np.stack([Y * c['uz'] - Z * c['uy'], Z * c['ux'] - X * c['uz'], X * c['uy'] - Y * c['ux']])
            Cc.update(E=c['E'][cand], L=np.linalg.norm(Lv, axis=0)[cand], X=c['x'][cand], Y=c['y'][cand],
                      Z=c['z'][cand], bidx=np.nonzero(cand)[0])
            dense = c['rho'] > 1e-3 * c['rho'].max()
            xstar = np.array([np.average(q[dense], weights=c['dM'][dense]) for q in (c['x'], c['y'], c['z'])])
            far3 = np.sqrt(X**2 + Y**2 + Z**2) >= 3.0
            ref = np.histogram(c['E'][far3] / DE, EB, weights=c['dM'][far3])[0] / MSTAR
            dt0 = B['dt'].min()
            rows.append(f'== {lab} t = {t:.1f} (r_s {rs:g}): dt0 {dt0:.4f} (code {logged_dt(d, tt):.4f}); star centroid '
                        f'r_BH {np.linalg.norm(xstar - xb):.1f}; grid mass per E bin (r_BH >= 3) '
                        + ' '.join(f'{x:.1e}' for x in ref))
            print(rows[-1], flush=True)
            for name, v in variants(rs):
                sink, shell = masks(Cc, rs, v, xstar)
                L = dt_with(B, Cc, sink | shell)
                rows.append(f'  {name:28s} dt {L["dt"]:.4f} (x{L["dt"] / dt0:.2f}) lim [{L["where"]}, L{L["lev"]}, '
                            f'r_BH {L["r"]:.2f}, rho {L["rho"]:.1e}, |lam| {L["lam"]:.3f} = |a-term {L["at"]:+.3f} - '
                            f'shift {L["bt"]:+.3f}|]')
                for tag, m in (('removed', sink), ('v-modified', shell)):
                    if not m.any():
                        continue
                    rr, eb, fr, lb = binned(m, Cc, ref)
                    rows.append(f'      {tag:10s} {m.sum():7d} cells, M {rr.sum():.1e} (r: ' + ' / '.join(f'{x:.1e}' for x in rr)
                                + f'); max rho {Cc["rho"][m].max() / RHOC:.1e}')
                    rows.append('        E bins ' + ' '.join(f'{x:.1e}' for x in eb))
                    rows.append('        frac   ' + ' '.join(f'{x:.1e}' for x in fr))
                    rows.append('        |L|    ' + ' '.join(f'{x:.1e}' for x in lb))
                print('\n'.join(rows[-1 - 4 * (int(sink.any()) + int(shell.any())):]), flush=True)
    if not a.only:
        with open(f'{a.out}/c047_s0b.txt', 'w') as f:
            f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
