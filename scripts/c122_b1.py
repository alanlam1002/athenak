#!/usr/bin/env python3
"""c122 B.1 (R-140 item 1, PHASE1_SPECS S-B B.1): the smooth-flow T-11 test, rp10 comoving L4,
metric_terms cell / analytic_bg / wellbalanced, dual energy off (B1N_*) and on (B1D_*).
From the 3D primitive dumps (every 6 M): over the dense bands rho >= 0.01 and >= 0.1 rho_c,
the mass-weighted mean of K/K0 - 1 (K = P/rho^Gamma, K0 = kappa; weights rho dV_coord, which
omits W psi^6 ~ 1 + O(1e-2) across the star: argued adequate for comparing arms), the mass
fractions with K/K0 - 1 > +5% / < -5%, rho_max/rho_c and K/K0 at rho_max. From the user
history: rho_max(t). From out.log (dual energy on): the last DESHK / CRECOV lines' counters.
    c122_b1.py OUTDIR RUNROOT"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
GAM = 5.0/3.0
ARMS = ['B1N_cell', 'B1N_abg', 'B1N_wb', 'B1D_cell', 'B1D_abg', 'B1D_wb']


def par(d, key):
    s = open(f'{d}/../{os.path.basename(d)}.par').read()
    return float(re.search(r'(?m)^%s\s*=\s*([-+.\deE]+)' % key, s).group(1))


def dump(f, rhoc, k0):
    P = bc.read_binary(f)
    g = np.asarray(P['mb_geometry']); n = P['nx1_out_mb']
    q = P['mb_data']
    pk = 'press' if 'press' in q else 'eint'
    rho = np.concatenate([np.asarray(q['dens'][b]).ravel() for b in range(len(g))])
    p = np.concatenate([np.asarray(q[pk][b]).ravel() for b in range(len(g))])
    if pk == 'eint':
        p = (GAM - 1.0)*p
    dv = np.concatenate([np.full(n**3, ((g[b, 1] - g[b, 0])/n)**3) for b in range(len(g))])
    x = p/rho**GAM/k0 - 1.0
    out = {'t': float(P['time'])}
    for lab, thr in (('d01', 0.01), ('d1', 0.1)):
        m = rho >= thr*rhoc
        w = rho[m]*dv[m]; W = w.sum()
        out[f'{lab}_mean'] = (w*x[m]).sum()/W
        out[f'{lab}_warm5'] = w[x[m] > 0.05].sum()/W
        out[f'{lab}_cold5'] = w[x[m] < -0.05].sum()/W
    i = np.argmax(rho)
    out['rhomax'] = rho[i]/rhoc; out['K_at_rhomax'] = x[i] + 1.0
    return out


def main():
    od, root = sys.argv[1], sys.argv[2]
    rows = []
    for a in ARMS:
        d = f'{root}/{a}'
        rhoc, k0 = par(d, 'rhoc'), par(d, 'kappa')
        for f in sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin')):
            o = dump(f, rhoc, k0)
            if o['t'] < 41.0 or o['t'] > 120.0 + 1e-6:
                continue
            rows.append((a, o))
    keys = ['t', 'd01_mean', 'd01_warm5', 'd01_cold5', 'd1_mean', 'd1_warm5', 'd1_cold5', 'rhomax',
            'K_at_rhomax']
    with open(f'{od}/c122_b1_dense.csv', 'w') as fo:
        fo.write('arm,' + ','.join(keys) + '\n')
        for a, o in rows:
            fo.write(a + ',' + ','.join(f'{o[k]:.6g}' for k in keys) + '\n')
    # dual-energy counters (last DESHK / CRECOV line of each DE-on arm)
    with open(f'{od}/c122_b1_de.txt', 'w') as fo:
        for a in ARMS:
            if not a.startswith('B1D'):
                continue
            L = open(f'{root}/{a}/out.log', errors='replace').read().split('\n')
            for tag in ('DESHK', 'CRECOV'):
                ll = [l for l in L if l.startswith(tag)]
                fo.write(f'{a} {ll[-1] if ll else "(none)"}\n')
    print(open(f'{od}/c122_b1_dense.csv').read())


if __name__ == '__main__':
    main()
