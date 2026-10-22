#!/usr/bin/env python3
"""C-098 (R-099 A1): the cold-recovery attribution, exact deterministic differences on the fixed binary.

Arms (c096_a1/<arm>, RC_R6X11's deck from R6 00006 -> t = 170, proj/tde 30c52c1d, 7 x 12 ranks each):
  on        X11 deck (cold_recovery, fofc_cold_recovery, fofc_scalar_consistent all true)
  fcr_off   fofc_cold_recovery = false
  fsc_off   fofc_scalar_consistent = false
  crec_off  cold_recovery = false AND fofc_cold_recovery = false (required: primitive_solver_hyd.hpp:1317)
The 13 physics rows of c087_x11.py (c051_s2 / c044_mdot on each arm's t = 170 restart). Same binary, restart,
deck and layout, so each difference is the key's exact effect:
  fofc_cold_recovery      = on - fcr_off
  fofc_scalar_consistent  = on - fsc_off
  cold_recovery itself    = fcr_off - crec_off   (both without the FOFC rule)
  cold_recovery + its rule = on - crec_off
For scale only: sigma_diff of the OLD-binary noise ensemble (C-081/C-087; pre-769a8efb, pinned background), which
is not this binary's spread (R-102: E4 measures that).

    c098_a1.py [--out FILE]
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c051_s2 as S          # noqa: E402
import c044_mdot as M        # noqa: E402
import c087_x11 as J         # noqa: E402

ARMS = ['on', 'fcr_off', 'fsc_off', 'crec_off']
PAIRS = (('fofc_cold_recovery (on - fcr_off)', 'on', 'fcr_off'),
         ('fofc_scalar_consistent (on - fsc_off)', 'on', 'fsc_off'),
         ('cold_recovery itself (fcr_off - crec_off)', 'fcr_off', 'crec_off'),
         ('cold_recovery + its FOFC rule (on - crec_off)', 'on', 'crec_off'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    a = ap.parse_args()
    cfg = S.RUNS['R6']
    tp = M.t_peri(cfg['src'], cfg['rp'])
    V, out = {}, []
    for r in ARMS:
        d = f'{S.C}/c096_a1/{r}'
        if not os.path.exists(f'{d}/DONE'):
            out.append(f'{r}: not done, skipped')
            continue
        fn, V[r] = J.rows(f'c096_a1/{r}', tp)
        out.append(f'{r}: {os.path.basename(fn)} t = {V[r]["t"]:.2f}')
    E = {}
    for r in J.ENS:
        _, E[r] = J.rows(r, tp)
    for lab, x, y in PAIRS:
        if x not in V or y not in V:
            continue
        out.append(f'-- {lab}')
        for key, name, kind in J.ROWS:
            dd = J.diff(V[x][key], V[y][key], kind)
            sig = float(np.sqrt(np.mean(np.square([J.diff(E[p][key], E[q][key], kind)
                                                   for p, q in itertools.combinations(J.ENS, 2)]))))
            xs = f'{float(V[x][key]):.6g}' if kind != 'q' else f'{np.max(V[x][key]):.4f}'
            out.append(f'   {name:20s} {xs:>12s}  diff {dd:+11.3e}   old-ensemble sig_diff {sig:9.3e}  ({abs(dd) / sig:5.2f} sig)')
    for r in V:
        led, cre, err, dts = J.logs(f'c096_a1/{r}')
        st = [k for k in cre if k['path'] == 'steps']
        g = lambda k: sum(float(c.get(k, 0)) for c in st)
        X = sum(float(k['X']) for k in led)
        out.append(f'{r}: cycles {len(led)} median dt {np.median(list(dts.values())):.5g} C2P errors {err} stage X {X:.3e}'
                   + (f' | CRECOV n {g("n"):.3e} DSchanged {g("DSchanged"):.0f} m_erased {g("m_erased"):.3e} '
                      f'fofc_rescued {g("fofc_rescued"):.3e}' if st else ' | no CRECOV steps lines'))
    txt = '\n'.join(out)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
