#!/usr/bin/env python3
"""c143 (R-152 item 6 test (2), user-approved 2026-10-28): the isolated-star checks of <cfc> solve_every_stages = N
against the N = 1 arm, from the 3D dumps (prim_3d and adm_3d, every 6 M).
  per arm and dump: the dense (rho >= 0.01, 0.1 rho_c) mass-weighted <K/K0 - 1> and warm/cold fractions, rho_max/rho_c
      (c137_iso.stats, star mode: K0 = kappa, rho_c = rhoc of the deck);
  per arm != REF and dump index: the held-metric error against REF at the same dump index (blocks matched by their
      geometry; the dump times are printed, they differ by < 1 dt): over the dense cells (rho_REF >= 0.01 rho_c) the max
      and mass-weighted |d psi4|/psi4, |d alpha|/alpha and |d beta|_coord, and the same maxima over all cells;
  the wall time per arm from DIR/../runs.log (start .. rc lines) and the cycle count.
    c143_cadence.py OUT.csv DECK REF ARM [ARM ...]
REF and ARM are run directories (bin/*.prim_3d.*.bin, bin/*.adm_3d.*.bin)."""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402
import c137_iso as I                              # noqa: E402


def blocks(f):
    P = bc.read_binary(f)
    g = np.asarray(P['mb_geometry'])
    return float(P['time']), {tuple(np.round(g[b], 9)): b for b in range(len(g))}, P['mb_data']


def adm_err(fr, fa, fpr, rhoc):
    tr, br, qr = blocks(fr)
    ta, ba, qa = blocks(fa)
    _, bp, qp = blocks(fpr)
    out = dict(t_ref=tr, t_arm=ta)
    mx = {'psi4': 0.0, 'alpha': 0.0, 'beta': 0.0}
    mxd = dict(mx)
    sw = {k: 0.0 for k in mx}
    w = 0.0
    n = 0
    for key, b in br.items():
        if key not in ba or key not in bp:
            continue
        n += 1
        c = ba[key]
        d = {'psi4': np.abs(np.asarray(qa['adm_psi4'][c]) / np.asarray(qr['adm_psi4'][b]) - 1.0),
             'alpha': np.abs(np.asarray(qa['adm_alpha'][c]) / np.asarray(qr['adm_alpha'][b]) - 1.0),
             'beta': np.sqrt(sum((np.asarray(qa[f'adm_beta{x}'][c]) - np.asarray(qr[f'adm_beta{x}'][b]))**2
                                 for x in 'xyz'))}
        rho = np.asarray(qp['dens'][bp[key]])
        dv = np.prod(np.asarray(key)[[1, 3, 5]] - np.asarray(key)[[0, 2, 4]]) / rho.size
        m = rho >= 0.01 * rhoc
        for k, v in d.items():
            mx[k] = max(mx[k], float(v.max()))
            if m.any():
                mxd[k] = max(mxd[k], float(v[m].max()))
                sw[k] += float((v[m] * rho[m]).sum() * dv)
        w += float(rho[m].sum() * dv)
    for k in mx:
        out[f'max_{k}'] = mx[k]
        out[f'dense_max_{k}'] = mxd[k]
        out[f'dense_mw_{k}'] = sw[k] / w if w > 0 else np.nan
    out['nblocks_matched'] = n
    return out


def wall(d):
    a = os.path.basename(d.rstrip('/'))
    L = open(f'{os.path.dirname(d.rstrip("/"))}/runs.log').read().splitlines()
    import datetime as dt
    tot = 0.0
    st = None
    for l in L:
        if l.startswith(f'{a} start'):
            st = dt.datetime.strptime(l.split()[2], '%Y-%m-%dT%H:%M:%SZ')
        elif l.startswith(f'{a} rc=') and st is not None:
            en = dt.datetime.strptime(re.search(r'end (\S+)', l).group(1), '%Y-%m-%dT%H:%M:%SZ')
            tot += (en - st).total_seconds()
            st = None
    cyc = re.findall(r'cycle=(\d+) time=', open(f'{d}/out.log').read())
    return tot, int(cyc[-1]) if cyc else -1


def main():
    out, deck, ref, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    s = open(deck).read()
    rhoc = float(re.search(r'(?m)^rhoc\s*=\s*([-+.\deE]+)', s).group(1))
    kap = float(re.search(r'(?m)^kappa\s*=\s*([-+.\deE]+)', s).group(1))
    rows = []
    for d in [ref] + arms:
        a = os.path.basename(d.rstrip('/'))
        wt, cyc = wall(d)
        fp = sorted(glob.glob(f'{d}/bin/*.prim_3d.*.bin'))
        for f in fp:
            idx = f.split('.')[-2]
            t, rho, p, dv, x = I.load(f)
            r = dict(arm=a, idx=idx, wall_s=wt, cycles=cyc)
            r.update(I.stats(t, rho, p, dv, x, kap, rhoc))
            if d != ref:
                fr = f'{ref}/bin/' + os.path.basename(f).replace('prim_3d', 'adm_3d')
                fa = f.replace('prim_3d', 'adm_3d')
                fpr = f'{ref}/bin/' + os.path.basename(f)
                if os.path.exists(fr) and os.path.exists(fa) and os.path.exists(fpr):
                    r.update(adm_err(fr, fa, fpr, rhoc))
            rows.append(r)
            print(r, flush=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out, 'w') as fo:
        fo.write(','.join(keys) + '\n')
        for r in rows:
            fo.write(','.join(str(r.get(k, '')) for k in keys) + '\n')


if __name__ == '__main__':
    main()
