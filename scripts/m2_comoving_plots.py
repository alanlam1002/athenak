#!/usr/bin/env python3
"""Phase 2 Milestone 2 (R-040 item 4/6): the comoving-gauge TDE runs against the BH-frame ones.

  controller: |V'| (grid velocity of the tracked gas), |x'_c - x'_tgt| and xidot from the
              history (fc_* columns); restart segments overplotted
  orbit:      mass-weighted BH-frame Killing E = -u_t - 1 and L_z = x u_y - y u_x (relative to
              x_BH) of all gas outside r_BH < 3, from restarts (scripts/tde_dmde.cells)
  star:       puffed mass and dense-gas mean K/K0 from z = 0 slices
              (scripts/tde_t11_surface_diag.measure)

    m2_comoving_plots.py COMOV=DIR[,DIR...] [...] --ref Q1=DIR [...] [--out f.png]
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import tde_t11_surface_diag as T                  # noqa: E402

TDE = 'cfc_tde_wd_imbh_parabolic_tracker'


def orbit(dirs):
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(f'{d}/rst/*.rst')):
            c = D.cells(f)
            m, xb = c['dM'], c['xbh']
            L = (c['x'] - xb[0]) * c['uy'] - (c['y'] - xb[1]) * c['ux']
            rows.append((c['t'], (c['E'] * m).sum() / m.sum(), (L * m).sum() / m.sum()))
    rows.sort()
    return np.array(rows)


def slices(lab, dirs, cache):
    out = []
    for i, d in enumerate(dirs):
        T.RUNS[f'{lab}_{i}'] = (d, TDE, 10, 'k', '-')
        s = T.series(f'{lab}_{i}', cache)
        out.append(np.column_stack([s['t'], s['puff'], s['Kw01'], s['Kmax']]))
    a = np.concatenate(out)
    return a[np.argsort(a[:, 0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='LABEL=DIR[,DIR...] comoving runs')
    ap.add_argument('--ref', nargs='*', default=[], help='LABEL=DIR[,DIR...] BH-frame runs')
    ap.add_argument('--tmax', type=float, default=1e9)
    ap.add_argument('--cache', default=f'{T.RUN}/plots/dual_energy')
    ap.add_argument('--out', default=f'{T.RUN}/plots/phase2_m1/m2_comoving.png')
    a = ap.parse_args()
    fig, ax = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for spec in a.runs:
        lab, ds = spec.split('='); dirs = ds.split(',')
        for d in dirs:
            h = np.loadtxt(glob.glob(f'{d}/*.user.hst')[0])
            h = h[h[:, 0] <= a.tmax]
            ax[0, 0].semilogy(h[:, 0], np.hypot(np.hypot(h[:, 10], h[:, 11]), h[:, 12]) + 1e-12,
                              lw=0.8, label=f'{lab} |V\'| ({os.path.basename(d)})')
            ax[0, 1].semilogy(h[:, 0], h[:, 14] + 1e-12, lw=0.8, label=f'{lab} ({os.path.basename(d)})')
            ax[0, 2].plot(h[:, 0], h[:, 4], lw=1, label=f'{lab} ξ̇x'); ax[0, 2].plot(h[:, 0], h[:, 5], lw=1, label=f'{lab} ξ̇y')
    for kind, specs, ls in (('comoving', a.runs, '-'), ('BH frame', a.ref, '--')):
        for spec in specs:
            lab, ds = spec.split('='); dirs = ds.split(',')
            o = orbit(dirs); o = o[o[:, 0] <= a.tmax]
            ax[1, 0].plot(o[:, 0], (o[:, 1] - o[0, 1]) / (0.677 / 16.4**2), 'o' + ls, ms=3, label=f'{lab} ({kind})')
            ax[1, 1].plot(o[:, 0], o[:, 2] - o[0, 2], 'o' + ls, ms=3, label=f'{lab} ({kind})')
            s = slices(lab, dirs, a.cache); s = s[s[:, 0] <= a.tmax]
            ax[1, 2].semilogy(s[:, 0], s[:, 1], ls, lw=1, label=f'{lab} puffed mass ({kind})')
            print(f'{lab} ({kind}): E drift {o[-1, 1] - o[0, 1]:+.3e} (= {(o[-1, 1] - o[0, 1]) / (0.677 / 16.4**2):+.4f} ΔE), '
                  f'L drift {o[-1, 2] - o[0, 2]:+.5f} by t = {o[-1, 0]:.0f}; puffed mass {s[-1, 1]:.2e}, '
                  f'dense mean K/K0 {s[-1, 2]:.4f}, core K/K0 {s[-1, 3]:.4f} at t = {s[-1, 0]:.0f}')
    ax[0, 0].set_title("controller: grid speed of the tracked gas |V'|")
    ax[0, 1].set_title("controller: centroid offset |x'_c − x'_tgt| [M]")
    ax[0, 2].set_title('ξ̇ (frame velocity)')
    ax[1, 0].set_title('orbit: Δ⟨E⟩ (mass-weighted, BH-frame Killing) / ΔE')
    ax[1, 1].set_title('orbit: Δ⟨L_z⟩ about the BH')
    ax[1, 2].set_title('puffed mass (slice; mass outside R* of the core)')
    for x in ax.flat:
        x.set_xlabel('t [M]'); x.grid(alpha=0.3); x.legend(fontsize=7)
    fig.savefig(a.out, dpi=90)


if __name__ == '__main__':
    main()
