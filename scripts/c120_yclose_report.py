#!/usr/bin/env python3
"""R-131 item 3 (c120): the YCLOSE tracer ledger of the CD/BD reruns, tabulated.
Reads the LAST segment's cumulative YCLOSE_PH_RUN (per phase, all items) and YCLOSE_BAND_RUN (per phase and
density band: C2P items, flagged up/down in 7 |V.n|_coord bins) lines of each run's out.log (mass_closure.cpp,
athenak wip/yclose 9aab7e69), and the YCLOSE closure lines (R, PASS) of every segment. Items (R-131 item 3):
 (a) flagged-dense resyncs (dbr 4) up / down, by band and |V.n| bin; (b) non-dense resyncs (dbr 0);
 (c) cold recovery -- NOTE: booked inside 'floor' (a cold-recovered cell always has prim_floor, ledger class 3,
     which mass_closure.cpp tests before bit 256), so (c) + the floor part of (d) are one item here;
 (d) floor_stage + floor_init, zone_rem/add, exc*, rf; (e) out; (f) regrid; plus source, rk_coef, restart.
    c120_yclose_report.py OUTDIR LABEL=RUNDIR [...]
"""
import sys

BANDS = ['>=0.1', '0.01-0.1', '0.001-0.01', '<0.001']
VB = ['<0.003', '0.003-0.01', '0.01-0.03', '0.03-0.1', '0.1-0.2', '0.2-0.3', '>=0.3']
PH = ['pre', 'peri', 'early', 'late']


def kv(line):
    return dict(t.rsplit('=', 1) for t in line.split() if '=' in t and not t.startswith('t=['))


def parse(d):
    L = open(f'{d}/out.log', errors='replace').read().splitlines()
    clos = [kv(l) for l in L if l.startswith('YCLOSE cycles=')]
    last = max(int(kv(l)['cycle']) for l in L if l.startswith('YCLOSE_PH_RUN'))
    ph = {int(kv(l)['ph']): kv(l) for l in L if l.startswith('YCLOSE_PH_RUN') and int(kv(l)['cycle']) == last}
    bd = {}
    for l in L:
        if l.startswith('YCLOSE_BAND_RUN') and int(kv(l)['cycle']) == last:
            k = kv(l)
            bd[(int(k['ph']), int(k['band']))] = {**{x: float(k[x]) for x in k if x not in ('up', 'dn', 'cycle')},
                                                 'up': [float(x) for x in k['up'].split(',')],
                                                 'dn': [float(x) for x in k['dn'].split(',')]}
    return clos, ph, bd


def main():
    out = sys.argv[1]
    R = {s.split('=')[0]: parse(s.split('=', 1)[1]) for s in sys.argv[2:]}
    rows = ['label,phase,band,item,value']
    summ = ['label,phase,S0,a_up,a_dn,a_net,b_nondense,cd_floor_crec,d_zone,d_exc_rf,e_out,f_regrid,other_items,total_dM']
    for lab, (clos, ph, bd) in R.items():
        S0 = float(clos[0]['M_start'])
        for p in sorted(ph):
            k = ph[p]
            f = lambda n: float(k[n])
            up = sum(sum(bd[(p, b)]['up']) for b in range(4))
            dn = sum(sum(bd[(p, b)]['dn']) for b in range(4))
            nd = sum(bd[(p, b)]['nondense'] for b in range(4))
            fl = f('floor_stage') + f('floor_init')
            zn = f('zone_rem') + f('zone_add') + f('unmask')
            ex = sum(f(n) for n in k if n.startswith('exc')) + f('rf')
            oth = f('source') + f('rk_coef') + f('restart') + sum(bd[(p, b)]['dense_chk'] for b in range(4))
            tot = up + dn + nd + fl + zn + ex + f('out') + f('regrid') + oth
            summ.append(f'{lab},{PH[p]},{S0:.6e},{up:.4e},{dn:.4e},{up + dn:.4e},{nd:.4e},{fl:.4e},{zn:.4e},{ex:.4e},'
                        f'{f("out"):.4e},{f("regrid"):.4e},{oth:.4e},{tot:.4e}')
            for b in range(4):
                for v in range(7):
                    rows.append(f'{lab},{PH[p]},{BANDS[b]},up{VB[v]},{bd[(p, b)]["up"][v]:.6e}')
                    rows.append(f'{lab},{PH[p]},{BANDS[b]},dn{VB[v]},{bd[(p, b)]["dn"][v]:.6e}')
                for it in ('nondense', 'floor', 'zone_rem', 'zone_add', 'exc', 'rf', 'crec', 'dense_chk'):
                    rows.append(f'{lab},{PH[p]},{BANDS[b]},{it},{bd[(p, b)][it]:.6e}')
        for c in clos:
            print(lab, 'segment', c['cycles'], 't_end', c['time'], 'dM', c['dM'], 'R', c['R'], 'R/M_grid', c['R/M_grid'])
    open(f'{out}/c120_yclose_items.csv', 'w').write('\n'.join(rows) + '\n')
    open(f'{out}/c120_yclose_summary.csv', 'w').write('\n'.join(summ) + '\n')
    print('\n'.join(summ))


if __name__ == '__main__':
    main()
