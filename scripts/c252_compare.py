#!/usr/bin/env python3
"""c252 (R-250 item 3): remnant measures at matched t - t_p from existing c142_B.csv tables (scripts/c142_remnant.py) and
rho_max tables (c160_census.py 'rhomax' or c252_rhomax.py), nearest row within +-TOL M of each target.
    c252_compare.py TARGETS(comma) RUN=c142csv[+c142csv...]@rhomaxcsv[:tcol:rcol:tp] ..."""
import sys
import pandas as pd
TOL = 20.0
tg = [float(x) for x in sys.argv[1].split(',')]
print('run,target,dt_c142,M_rem,M_Hill,M_Hsc,K_Hill,K_rem,dt_rho,rhomax')
for spec in sys.argv[2:]:
    name, rest = spec.split('=', 1)
    cs, rs = rest.split('@')
    c = pd.concat([pd.read_csv(f) for f in cs.split('+')]).sort_values('t_minus_tp').drop_duplicates('t_minus_tp').reset_index(drop=True)
    rf, tcol, rcol, tp = rs.split(':'); r = pd.read_csv(rf).sort_values(tcol).reset_index(drop=True); rt = r[tcol] - float(tp) if tcol == 't' else r[tcol]
    for x in tg:
        row = c.loc[(c.t_minus_tp - x).abs().idxmin()] if len(c) else None
        j = (rt - x).abs().idxmin()
        ok = row is not None and abs(row.t_minus_tp - x) <= TOL
        okr = abs(rt[j] - x) <= TOL
        f = lambda k: f'{row[k]:.3f}' if ok else ''
        print(f"{name},{x:.0f},{(row.t_minus_tp - x) if ok else float('nan'):+.1f},{f('M_rem')},{f('M_Hill')},{f('M_Hsc')},{f('K_Hill')},{f('K_rem')},"
              f"{(rt[j] - x) if okr else float('nan'):+.1f},{(r[rcol][j] if okr else float('nan')):.4f}")
