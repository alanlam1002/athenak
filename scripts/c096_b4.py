#!/usr/bin/env python3
"""C-096 (R-099 B4): excision / accretion / zone booking totals of paired runs (old vs fixed trumpet).

From each run's out.log (rank-0 diagnostics of proj/tde, src/eos/primitive_solver_hyd.hpp, src/cfc/cfc.cpp):
  MLEDGER_X  per cycle: the C2P's rest-mass change X by class (0 excision floor+flux mask, 1 floor mask
             only, 2 zone/tier 3, 3 C2P-floored, 4 other) x r_BH bin (< 1.88, 1.88-2.92, 2.92-6, 6-12,
             >= 12) x (removed, added) (primitive_solver_hyd.hpp l. 495-511, 797); summed over the window;
  MLEDGER    per cycle: Fexc (mass flux into excision-flux cells), Fout; summed;
  ATMZONE    the last line's cumulative zone dM, dE, dM_exc; ATMZONE_BINS the last cumulative zone mass
             in r_BH (< 1.88, 1.88-2.92, >= 2.92; since R-107 item 5 also split at 0.59 / 0.78) summed
             over E and |L| (layout [r][E][L], 3 or 5 x 10 x 5);
  CFC accretion  the last M_accreted and the summed dM_raw (cfc.cpp l. 2029-2036).
The r_BH bins are fixed in the code at 1.88 / 2.92 (areal 3 M / 4 M). With the fixed background the
excision is r_iso < 0.59 (was < 1.50), so '< 1.88' now holds excised cells (< 0.59) AND evolved cells
(0.59-1.88); before 30c52c1d it held excised (< 1.50) and evolved (1.50-1.88) cells.

    c096_b4.py LABEL=RUNDIR [...]
"""
import re
import sys

import numpy as np

CLS = ('excision', 'floor-mask', 'zone/t3', 'C2P-floored', 'other')
RB5 = ('<1.88', '1.88-2.92', '2.92-6', '6-12', '>=12')
# since wip/r107-hardening (R-107 item 5) the inner bin is split at the lapse excision and horizon
RB7 = ('<0.59', '0.59-0.78', '0.78-1.88', '1.88-2.92', '2.92-6', '6-12', '>=12')
ZR3 = ('<1.88', '1.88-2.92', '>=2.92')
ZR5 = ('<0.59', '0.59-0.78', '0.78-1.88', '1.88-2.92', '>=2.92')


def parse(d):
    X = None
    fexc = fout = draw = 0.0
    macc = np.nan
    zone = bins = None
    ncyc = 0
    for line in open(f'{d}/out.log'):
        if line.startswith('MLEDGER_X '):
            v = np.array([float(q) for q in line.split()[2:]])
            X = v if X is None else X + v
        elif line.startswith('MLEDGER '):
            ncyc += 1
            fexc += float(re.search(r'Fexc=(\S+)', line).group(1))
            fout += float(re.search(r'Fout=(\S+)', line).group(1))
        elif line.startswith('ATMZONE '):
            zone = {k: float(re.search(rf' {k}=(\S+)', line).group(1)) for k in ('cum_dM', 'cum_dE', 'cum_dM_exc')}
        elif line.startswith('ATMZONE_BINS '):
            bins = np.array([float(v) for v in line.split(':', 1)[1].split()])
        elif line.startswith('### CFC accretion'):
            draw += float(re.search(r'dM_raw=(\S+)', line).group(1))
            macc = float(re.search(r'M_accreted=(\S+)', line).group(1))
    nr = (0 if X is None else X.size) // 10
    return dict(X=(np.zeros((5, 5, 2)) if X is None else X.reshape(5, nr, 2)), fexc=fexc, fout=fout, draw=draw, macc=macc, zone=zone, bins=bins, n=ncyc)


def main():
    runs = [a.split('=', 1) for a in sys.argv[1:]]
    R = {lab: parse(d) for lab, d in runs}
    labs = [lab for lab, _ in runs]
    print('cycles: ' + ', '.join(f'{lab} {R[lab]["n"]}' for lab in labs))
    print('C2P rest-mass change X summed over the window, removed / added [M_sun-units of the code]:')
    print('  class        r_BH       ' + ''.join(f'{lab + " rem":>14s}{lab + " add":>14s}' for lab in labs))
    nr = R[labs[0]]['X'].shape[1]
    if any(R[lab]['X'].shape[1] != nr for lab in labs):
        raise SystemExit('runs have different MLEDGER_X binnings (5 vs 7 r bins): compare like with like')
    RB = RB7 if nr == 7 else RB5
    for c, cn in enumerate(CLS):
        for r, rn in enumerate(RB):
            vals = [R[lab]['X'][c, r] for lab in labs]
            if all(abs(v).sum() == 0 for v in vals):
                continue
            print(f'  {cn:12s} {rn:10s} ' + ''.join(f'{v[0]:14.3e}{v[1]:14.3e}' for v in vals))
    for k, txt in (('fexc', 'excision inflow sum Fexc'), ('fout', 'outer outflow sum Fout'),
                   ('draw', 'CFC accretion sum dM_raw'), ('macc', 'CFC M_accreted (last)')):
        print(f'  {txt:28s}' + ''.join(f'{R[lab][k]:16.4e}' for lab in labs))
    for k in ('cum_dM', 'cum_dE', 'cum_dM_exc'):
        print(f'  zone {k:23s}' + ''.join(f'{(R[lab]["zone"] or {}).get(k, np.nan):16.4e}' for lab in labs))
    for lab in labs:
        b = R[lab]['bins']
        if b is not None and b.size in (150, 250):
            nz = b.size // 50
            rb = b.reshape(nz, 10, 5).sum(axis=(1, 2))
            names = ZR5 if nz == 5 else ZR3
            print(f'  zone cumulative dM by r_BH ({lab}): ' + '  '.join(f'{n} {v:.3e}' for n, v in zip(names, rb)))


if __name__ == '__main__':
    main()
