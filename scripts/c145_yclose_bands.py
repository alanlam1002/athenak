#!/usr/bin/env python3
"""c145 (R-152 item 3(ii), R-153: the YCLOSE attribution of the envelope heating, by density band and phase).
From a run's out.log, the last cycle's cumulative YCLOSE_BAND_RUN lines (mass_closure.cpp; per phase and band
rho/rho_c >= 0.1, [0.01, 0.1), [0.001, 0.01), < 0.001 with rho_c = <mhd> tracer_closure_rhoc) and YCLOSE_PH_RUN
lines (per phase, all items, including the init/regrid path's floor_init and regrid): the change of the dual-energy
tracer DY booked to
  up / dn   flagged-dense resyncs (dbr 4: the energy solution, DY raised / lowered; summed over the |V.n| bins),
  nondense  non-dense resyncs (dbr 0),  floor (class 3), crec (cold recovery), zone (removed + added),
  exc       lapse excision + r_f,       dense_chk (dbr 1/2/3/5, DY unchanged: a check),
per band (C2P items only) and per phase (all items: + out, floor_init, regrid, source, other = the C2P items).
    c145_yclose_bands.py OUT.csv LABEL=RUNDIR [...]"""
import sys


def kv(line):
    return dict(t.split('=', 1) for t in line.split() if '=' in t and not t.startswith('t=['))


rows = []
for spec in sys.argv[2:]:
    lab, d = spec.split('=', 1)
    L = open(f'{d}/out.log', errors='replace').read().splitlines()
    last = max(int(kv(l)['cycle']) for l in L if l.startswith('YCLOSE_BAND_RUN'))
    ph = {}
    for l in L:
        if l.startswith('YCLOSE_PH_RUN') and int(kv(l)['cycle']) == last:
            k = kv(l)
            ph[int(k['ph'])] = (l.split('t=[')[1].split(')')[0], k)
    for l in L:
        if not (l.startswith('YCLOSE_BAND_RUN') and int(kv(l)['cycle']) == last):
            continue
        k = kv(l)
        p, b = int(k['ph']), int(k['band'])
        r = dict(label=lab, phase=p, t_range=ph[p][0] if p in ph else '', band=b,
                 up=sum(float(x) for x in k['up'].split(',')), dn=sum(float(x) for x in k['dn'].split(',')),
                 nondense=float(k['nondense']), floor=float(k['floor']), crec=float(k['crec']),
                 zone=float(k['zone_rem']) + float(k['zone_add']), exc=float(k['exc']) + float(k['rf']),
                 dense_chk=float(k['dense_chk']))
        rows.append(r)
    for p, (tr, k) in sorted(ph.items()):
        rows.append(dict(label=lab, phase=p, t_range=tr, band='all',
                         **{x: float(k[x]) for x in ('out', 'floor_stage', 'floor_init', 'regrid', 'crec', 'source',
                                                     'other', 'zone_rem', 'zone_add', 'rf')}))
keys = []
for r in rows:
    keys += [x for x in r if x not in keys]
with open(sys.argv[1], 'w') as f:
    f.write(','.join(keys) + '\n')
    for r in rows:
        f.write(','.join(f'{r[x]:.4e}' if isinstance(r.get(x), float) else str(r.get(x, '')) for x in keys) + '\n')
for r in rows:
    if r['band'] in (1, 2, 3, 'all'):
        print({x: (f'{v:.3e}' if isinstance(v, float) else v) for x, v in r.items()})
