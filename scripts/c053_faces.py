#!/usr/bin/env python3
"""C-053 (R-066 item 2): locate the flux update's rest-mass non-conservation from the MLFACE
dump (<mhd> mass_ledger_faces): per stage, every MeshBlock's beta_s dt sum F(IDN) dA through
its six faces. Neighbouring faces are matched by geometry; per shared face the mass created is
  dM = sum_B F_B(-) - F_A(+)      (A on the low side, B the block(s) on the high side; F in +x),
which is zero if both sides used the same flux. Combined over stages with w_s = prod_{j>s}
gam0_j (rk3: 1/6, 2/3, 1) and summed by r_BH of the face centre, by level relation (same
level / coarse-fine) and by axis; plus the largest faces. x_BH(t) = puncture_x - xi(t), with xi
from the user.hst frame_disp columns.

    c053_faces.py RUNDIR [--par PARFILE]
"""
import argparse
import collections
import re

import numpy as np

W = {1: 1.0 / 6.0, 2: 2.0 / 3.0, 3: 1.0}
RB = np.array([0, 1.88, 2.92, 6, 12, 24, np.inf])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--par', default=None)
    a = ap.parse_args()
    par = open(a.par or f'{a.run}/../T.par').read()
    px = np.array([float(re.search(rf'^puncture_x{n}\s*=\s*([-+.\deE]+)', par, re.M).group(1))
                   if re.search(rf'^puncture_x{n}\s*=', par, re.M) else 0.0 for n in (1, 2, 3)])
    uh = np.loadtxt(f'{a.run}/cfc_tde_wd_imbh_parabolic_tracker.user.hst')
    xi = uh[:, 7:10]
    rows = collections.defaultdict(list)
    for ln in open(f'{a.run}/mlface.txt'):
        p = ln.split()
        rows[(int(p[1]), int(p[2]))].append([float(v) for v in p[3:]])
    led = {}
    for ln in open(f'{a.run}/out.log'):
        if ln.startswith('MLEDGER '):
            d = dict(re.findall(r'(\w+)=([^ ]+)', ln))
            led[int(d['cycle'])] = (float(d['time']), float(d['Y']), float(d['Fout']))
    tot = collections.Counter()
    big = []
    per_cyc = collections.Counter()
    for (cyc, st), r in sorted(rows.items()):
        b = np.array(r)            # gid lev x1min x1max x2min x2max x3min x3max f1- f1+ f2- f2+ f3- f3+
        t = led.get(cyc, (np.nan,))[0]
        k = int(np.argmin(np.abs(uh[:, 0] - t))) if np.isfinite(t) else -1
        xbh = px - xi[k]
        w = W[st]
        for ax in range(3):
            lo, hi = 2 + 2 * ax, 3 + 2 * ax
            tr = [q for q in range(3) if q != ax]
            # index high-side faces (x_min of each block) by coordinate
            idx = collections.defaultdict(list)
            for j in range(len(b)):
                idx[round(b[j, lo], 9)].append(j)
            for i in range(len(b)):
                X = round(b[i, hi], 9)
                cand = idx.get(X, [])
                # blocks whose transverse box overlaps A's
                nb = [j for j in cand if all(b[j, 2 + 2 * q] < b[i, 3 + 2 * q] - 1e-12 and
                                             b[j, 3 + 2 * q] > b[i, 2 + 2 * q] + 1e-12 for q in tr)]
                if not nb:
                    continue
                if len(nb) == 1 and b[nb[0], 1] < b[i, 1]:
                    continue        # A fine, B coarse: counted from the coarse side's group
                fa = b[i, 9 + 2 * ax]                      # A's + face
                fb = sum(b[j, 8 + 2 * ax] for j in nb)     # B's - faces
                rel = 'same' if (len(nb) == 1 and b[nb[0], 1] == b[i, 1]) else 'coarse|fine'
                dm = w * (fb - fa)
                c = np.array([(b[i, 2 + 2 * q] + b[i, 3 + 2 * q]) / 2 for q in range(3)])
                c[ax] = b[i, hi]
                r = float(np.linalg.norm(c - xbh))
                ir = int(np.searchsorted(RB, r, side='right') - 1)
                tot[(rel, ir, ax)] += dm
                per_cyc[cyc] += dm
                big.append((abs(dm), dm, cyc, st, rel, ax, r, int(b[i, 1]), c.round(3).tolist()))
    print('mass created at shared faces, by level relation x r_BH bin (sum over axes), M_BH units:')
    rels = sorted(set(k[0] for k in tot))
    print(' ' * 14 + ''.join(f'{f"{RB[i]:g}-{RB[i + 1]:g}":>13s}' for i in range(len(RB) - 1)) + '        total')
    for rel in rels:
        v = [sum(tot[(rel, i, ax)] for ax in range(3)) for i in range(len(RB) - 1)]
        print(f'{rel:14s}' + ''.join(f'{x:13.3e}' for x in v) + f'{sum(v):13.3e}')
    print('by axis: ' + ', '.join(f'x{ax + 1} {sum(v for k, v in tot.items() if k[2] == ax):+.3e}' for ax in range(3)))
    print('per cycle (faces) vs ledger Y+Fout:')
    for cyc in sorted(per_cyc):
        y = led.get(cyc)
        print(f'  c{cyc}: faces {per_cyc[cyc]:+.4e}  ledger Y+Fout {y[1] + y[2] if y else np.nan:+.4e}')
    big.sort(reverse=True)
    print('largest faces: |dM| dM cycle stage rel axis r_BH level centre')
    for x in big[:25]:
        print('  ' + ' '.join(str(v) if not isinstance(v, float) else f'{v:.3e}' for v in x[1:]))


if __name__ == '__main__':
    main()
