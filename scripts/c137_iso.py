#!/usr/bin/env python3
"""c137 (R-150 item 4, user-approved 2026-10-28): the T-11 follow-ups, from the 3D primitive dumps.
Per dump, over the dense bands rho >= 0.01 and >= 0.1 rho_c: the mass-weighted mean of K/K0 - 1 (K = P/rho^Gamma;
weights rho dV_coord, as c122_b1/C-118), the warm (> +5%) and cold (< -5%) mass fractions, rho_max/rho_c, K/K0 at
rho_max and the density-weighted centre x_cm (grid coordinates).
  star mode (ISO_A3, ISO_A2; C-118's ISO_A/ISO_B): K0 = kappa, rho_c = <problem> rhoc (the deck);
  bump mode (ISO_A4p, ISO_B4p): the gaussian entropy bump has uniform P, so K0 is each cell's own K at t = 0
      (the first dump; the bump sits at grid rest, a uniform mesh, so a cell holds the same gas to O(drift)),
      rho_c = <problem> rho0.
    c137_iso.py OUT.csv RUNDIR[:star|:bump] [...]
RUNDIR is an arm directory (bin/*.bin; deck <dir>/../<basename>.par)."""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402

GAM = 5.0 / 3.0
KEYS = ['t', 'd01_mean', 'd01_warm5', 'd01_cold5', 'd1_mean', 'd1_warm5', 'd1_cold5', 'rhomax', 'K_at_rhomax',
        'xcm']


def par(d, key):
    s = open(f'{d}/../{os.path.basename(d)}.par').read()
    return float(re.search(r'(?m)^%s\s*=\s*([-+.\deE]+)' % key, s).group(1))


def load(f):
    P = bc.read_binary(f)
    g = np.asarray(P['mb_geometry'])
    n = P['nx1_out_mb']
    q = P['mb_data']
    pk = 'press' if 'press' in q else 'eint'
    rho = np.concatenate([np.asarray(q['dens'][b]).ravel() for b in range(len(g))])
    p = np.concatenate([np.asarray(q[pk][b]).ravel() for b in range(len(g))])
    if pk == 'eint':
        p = (GAM - 1.0) * p
    dv = np.concatenate([np.full(n**3, ((g[b, 1] - g[b, 0]) / n)**3) for b in range(len(g))])
    x = np.concatenate([np.broadcast_to(g[b, 0] + (np.arange(n) + 0.5) * (g[b, 1] - g[b, 0]) / n,
                                        (n, n, n)).ravel() for b in range(len(g))])
    return float(P['time']), rho, p, dv, x


def stats(t, rho, p, dv, x, k0, rhoc):
    k = p / rho**GAM / k0 - 1.0
    out = {'t': t}
    for lab, thr in (('d01', 0.01), ('d1', 0.1)):
        m = rho >= thr * rhoc
        w = rho[m] * dv[m]
        W = w.sum()
        out[f'{lab}_mean'] = (w * k[m]).sum() / W
        out[f'{lab}_warm5'] = w[k[m] > 0.05].sum() / W
        out[f'{lab}_cold5'] = w[k[m] < -0.05].sum() / W
    i = np.argmax(rho)
    out['rhomax'] = rho[i] / rhoc
    out['K_at_rhomax'] = k[i] + 1.0
    m = rho >= 0.01 * rhoc
    out['xcm'] = (rho[m] * dv[m] * x[m]).sum() / (rho[m] * dv[m]).sum()
    return out


def main():
    out, rows = sys.argv[1], []
    for s in sys.argv[2:]:
        d, mode = (s.rsplit(':', 1) + ['star'])[:2] if ':' in s else (s, 'star')
        a = os.path.basename(d.rstrip('/'))
        fs = sorted(glob.glob(f'{d}/bin/*.prim_3d.*.bin')) or sorted(glob.glob(f'{d}/bin/*.bin'))   # 3D dumps only
        if mode == 'bump':
            rhoc = par(d, 'rho0')
            _, r0, p0, _, _ = load(fs[0])
            k0 = p0 / r0**GAM
        else:
            rhoc, k0 = par(d, 'rhoc'), par(d, 'kappa')
        for f in fs:
            t, rho, p, dv, x = load(f)
            rows.append((a, stats(t, rho, p, dv, x, k0, rhoc)))
    with open(out, 'w') as fo:
        fo.write('arm,' + ','.join(KEYS) + '\n')
        for a, o in rows:
            fo.write(a + ',' + ','.join(f'{o[k]:.6g}' for k in KEYS) + '\n')
    print(open(out).read())


if __name__ == '__main__':
    main()
