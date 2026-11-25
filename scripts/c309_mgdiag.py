#!/usr/bin/env python3
"""c309 (R-280 items 2-4): parse wip/mg-diag MGDIAG lines (multigrid_driver.cpp DiagPrint) of replay runs. A solve = the lines from
one 'it 0' to the next for a field. Per solve and channel: n (V-cycles), at the stopping point (last it) the defect shares f_hole
(within 5 M of the hole), f_star (within 2 R*), f_else, the rms defect sqrt(sum d^2 dV / V_ref) (V_ref = 128^3, mg_norm_ref_volume),
eps (last correction, mass-weighted), eps_str (stream class), dumax_star. Cycles < --cmin are skipped (warm starts from another mesh).
Item 4: for each solve of REF (default M) the history eps_k and rms_k; for a candidate criterion "stop at the first k >= 1 with
eps_k < tol and rms_k < 1e-8" the V-cycles it needs, tol = the median final eps of REF (= L32M's accuracy).
    c309_mgdiag.py DIR RUN [RUN ...] [--cmin 505] [--ref M]"""
import argparse, collections, re, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('runs', nargs='+'); ap.add_argument('--cmin', type=int, default=505)
ap.add_argument('--ref', default='M'); ap.add_argument('--vref', type=float, default=128.0**3)
a = ap.parse_args()
def parse(run):
    solves = []; cur = {}
    for l in open(f'{a.dir}/{run}/out.log'):
        if not l.startswith('MGDIAG '): continue
        t = l.split(); d = dict(zip(t[2::2], t[3::2])); name = t[1]
        it, v, cyc = int(d['it']), int(d['v']), int(d['cyc'])
        rec = {k: float(d[k]) for k in ('eps', 'eps_str', 'm_str', 'dumax_star', 'd2_hole', 'd2_star', 'd2_else')}
        rec['rms'] = ((rec['d2_hole'] + rec['d2_star'] + rec['d2_else'])/a.vref)**0.5
        key = (name, v)
        if it == 0:
            if key in cur: solves.append(cur.pop(key))
            cur[key] = dict(name=name, v=v, cyc=cyc, hist=[rec])
        elif key in cur: cur[key]['hist'].append(rec)
    solves += list(cur.values())
    return [s for s in solves if s['cyc'] >= a.cmin]
R = {r: parse(r) for r in a.runs}
med = lambda x: st.median(x) if x else float('nan')
print(f'{"run":5s} {"field":10s} v  nsolve  Vcyc(med/mean)  f_hole@stop f_star@stop f_else@stop  rms@stop    eps@stop   eps_str@stop dumax_star@stop  f_hole@it0')
for r, S in R.items():
    g = collections.defaultdict(list)
    for s in S: g[(s['name'], s['v'])].append(s)
    for (n, v), L in sorted(g.items()):
        fin = [s['hist'][-1] for s in L]; tot = lambda h: h['d2_hole'] + h['d2_star'] + h['d2_else']
        fh = med([h['d2_hole']/tot(h) for h in fin if tot(h) > 0]); fs = med([h['d2_star']/tot(h) for h in fin if tot(h) > 0])
        f0 = med([s['hist'][0]['d2_hole']/tot(s['hist'][0]) for s in L if tot(s['hist'][0]) > 0])
        print(f'{r:5s} {n:10s} {v}  {len(L):5d}  {med([len(s["hist"]) - 1 for s in L]):4.1f}/{sum(len(s["hist"]) - 1 for s in L)/len(L):5.2f}  {fh:10.3e}  {fs:10.3e}  {1 - fh - fs:10.3e}  '
              f'{med([h["rms"] for h in fin]):10.3e} {med([h["eps"] for h in fin]):10.3e} {med([h["eps_str"] for h in fin]):10.3e}  '
              f'{med([h["dumax_star"] for h in fin]):10.3e}  {f0:10.3e}')
if a.ref in R:
    print(f'\nitem 4: criterion "first k >= 1 with eps_k < tol and rms_k < 1e-8" on {a.ref}\'s histories (tol per field/channel = median final eps of {a.ref})')
    g = collections.defaultdict(list)
    for s in R[a.ref]: g[(s['name'], s['v'])].append(s)
    for (n, v), L in sorted(g.items()):
        tol = med([s['hist'][-1]['eps'] for s in L])
        for lab, tl in (('median final eps', tol), ('max final eps', max(s['hist'][-1]['eps'] for s in L))):
            need, used, miss = [], [], 0
            for s in L:
                k = next((k for k, h in enumerate(s['hist']) if k >= 1 and h['eps'] <= tl and h['rms'] < 1e-8), None)
                if k is None: miss += 1; k = len(s['hist']) - 1
                need.append(k); used.append(len(s['hist']) - 1)
            print(f'  {n:10s} v{v}: tol = {lab} {tl:.3e}: eps-criterion V-cycles mean {sum(need)/len(need):.2f} (not reached in {miss}/{len(L)}), '
                  f'rms 1e-10 used mean {sum(used)/len(used):.2f}')
