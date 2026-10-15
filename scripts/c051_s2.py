#!/usr/bin/env python3
"""C-051 (R-058 §2 / R-063 §3, §5 / R-064 step 2): evaluation of the S2 restart tests.

Arms (one 16-node bundle each; R6 restart 00006 t = 113.7 -> 170 on 7 nodes, R7 restart
00013 t = 253.7 -> 300 on 9 nodes; binary build_gpu_pe2626_zone d21072a5):
  T = zone on + dual_energy_div covariant (+ tier 3, rule-gated)
  A = zone off + coord      (atmosphere_zone_book: booking only)
  B = zone off + covariant  (atmosphere_zone_book)
The zone effect is T - B, the divergence effect B - A.

Criteria (numbers per R-058 §2): dt gain; capture check on the zone-removed mass; wings;
Mdot (frozen-in, final restart); rest-mass booking closure (hst mass + excision + zone);
the gate; physics within control noise; tier 3 (T_R7); B - A (switch census); cost.

    c051_s2.py [--nproc 6] [--out DIR]
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import remnant as RM                              # noqa: E402
import rst_tree as RT                             # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
P = f'{C}/plots/phase2_m1'
B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC, K0, MSTAR, GAM = 3.24e-4, 1.2346626297e-2, 7.47e-5, 5.0 / 3.0
DE = 0.677 / 16.4**2
THR = 1e-10 * RHOC
EB = np.array([-np.inf, -6, -3, -2, -1, 0, 1, 2, 3, 6, np.inf])
ELAB = ['<-6', '-6..-3', '-3..-2', '-2..-1', '-1..0', '0..1', '1..2', '2..3', '3..6', '>6']
RUNS = {'R6': dict(src='R6_rp10_comov_L4_fix', rst0='00006', rp=10.0, nodes=7, tend=170.0,
                   win=((113.7, 136.0), (136.0, 150.0), (150.0, 170.0)), gain=1.4),
        'R7': dict(src='R7_rp18_comov_L4_fix', rst0='00013', rp=18.0, nodes=9, tend=300.0,
                   win=((253.7, 275.0), (275.0, 300.0)), gain=2.0)}
ARMS = ('T', 'A', 'B')
CACHE = f'{P}/c051_cache'


# ---------------------------------------------------------------- logs
def parse(d):
    """ensemble.out: dt series, ATMZONE rows, last bins, T3 rows/bins, timers, C2P errors"""
    o = dict(dt=[], z=[], bins=None, t3=[], t3bins=None, timers=None, c2p=0, elapsed=0.0)
    for ln in open(f'{d}/ensemble.out'):
        m = re.match(r'elapsed=(\S+) cycle=\d+ time=(\S+) dt=(\S+)', ln)
        if m:
            o['dt'].append((float(m.group(2)), float(m.group(3)))); o['elapsed'] = float(m.group(1))
            continue
        if ln.startswith('ATMZONE '):
            kv = dict(re.findall(r'(\w+)=([^\s(]+)', ln))
            o['z'].append([float(kv[k]) for k in ('time', 'dM', 'n_mod', 'max_rho_mod', 'max_rho_zone',
                                                    'n_viol', 'cum_dM', 'cum_dE', 'dM_exc', 'cum_dM_exc')])
        elif ln.startswith('ATMZONE_BINS'):
            o['bins'] = (float(re.search(r'time=(\S+)', ln).group(1)),
                         np.array(ln.split(':')[-1].split(), float).reshape(3, 10, 5))
        elif ln.startswith('ATMZONE_T3 '):
            kv = dict(re.findall(r'(\w+)=([^\s(]+)', ln))
            o['t3'].append([float(kv[k]) for k in ('time', 'active', 'd', 'R90', 'M_mod', 'n_mod', 'n_viol')]
                           + [float(x) for x in re.search(r'cum_dS=\(([^)]*)\)', ln).group(1).split(',')]
                           + [float(kv['cum_dE'])])
        elif ln.startswith('ATMZONE_T3_BINS'):
            o['t3bins'] = (float(re.search(r'time=(\S+)', ln).group(1)),
                           np.array(ln.split(':')[-1].split(), float).reshape(3, 10, 5))
        elif ln.startswith('PHASE_TIMERS_FINAL'):
            o['timers'] = {k: float(v) for k, v in re.findall(r'(\w+)=(\S+)', ln)}
        elif 'An error occurred during the primitive solve' in ln:
            o['c2p'] += 1
    for k in ('dt', 'z', 't3'):
        o[k] = np.array(o[k])
    return o


# ---------------------------------------------------------------- 3D restart pass
def rst_pass(fn):
    tag = fn.split('/')[-3] + '_' + fn.split('.')[-2]
    cf = f'{CACHE}/{tag}.npz'
    if os.path.exists(cf):
        z = np.load(cf)
        return {k: z[k] for k in z.files}
    c = D.cells(fn, rcut=3.0)
    Mb, xc, vc, rem = RM.bound_mass(c)
    xb = c['xbh']
    Lz = (c['x'] - xb[0]) * c['uy'] - (c['y'] - xb[1]) * c['ux']
    dense = c['rho'] > 0.1 * RHOC
    k = c['P'][dense] / c['rho'][dense]**GAM / K0 - 1
    out = dict(t=c['t'], E=c['E'], dM=c['dM'], Lz=Lz, rem=rem, Mb=Mb / MSTAR, kd=k, md=c['dM'][dense])
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cf, **out)
    return out


def quant(E, m, q):
    o = np.argsort(E); c = np.cumsum(m[o]) / m.sum()
    return np.interp(q, c, E[o])


def physics(z):
    """the R-058 §2 physics rows from one restart pass"""
    deb = ~z['rem'].astype(bool)
    md, kd = z['md'], z['kd']
    return dict(t=float(z['t']), K=float(np.average(kd, weights=md)) + 1.0,
                M3=md[kd > 0.05].sum() / md.sum(), M10=md[kd > 0.10].sum() / md.sum(),
                E=(z['E'] * z['dM']).sum() / z['dM'].sum() / DE, Lz=(z['Lz'] * z['dM']).sum() / z['dM'].sum(),
                bound=z['dM'][z['E'] < 0].sum() / z['dM'].sum(), rem=float(z['Mb']),
                q=quant(z['E'][deb], z['dM'][deb], np.array([0.1, 0.25, 0.5, 0.75, 0.9])) / DE,
                qx=quant(z['E'][deb], z['dM'][deb], np.array([0.05, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 0.95])) / DE)


# ---------------------------------------------------------------- slices (matched times)
def slice_K(f):
    """dense (rho > 0.1 rho_c) area- and density-weighted mean K/K0 and warm fractions, z = 0"""
    from tde_ensemble_plots import bc
    q = bc.read_binary(f)
    g = np.asarray(q['mb_geometry']); nx = q['nx1_out_mb']
    rho = np.asarray(q['mb_data']['dens'])[:, 0]; Pr = np.asarray(q['mb_data']['press'])[:, 0]
    h2 = ((g[:, 1] - g[:, 0]) / nx)**2
    w = rho * h2[:, None, None]
    dn = rho > 0.1 * RHOC
    k = Pr / np.maximum(rho, 1e-300)**GAM / K0
    return (q['time'], np.average(k[dn], weights=w[dn]), w[dn][k[dn] > 1.05].sum() / w[dn].sum(),
            w[dn][k[dn] > 1.10].sum() / w[dn].sum())


def _census(d, n):
    import d_r043_maps as R
    return R.census(R.load(d, 'xy', n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=6)
    ap.add_argument('--out', default=P)
    a = ap.parse_args()
    rows = ['C-051: S2 evaluation (c051_s2.py). T = zone on + covariant (+ tier 3), A = zone off + coord, '
            'B = zone off + covariant. Measured unless marked argued. Masses in M_BH units unless "M*".']
    L = {(r, x): parse(f'{C}/S2{x}_{r}') for r in RUNS for x in ARMS}
    # 3D passes: start restart (shared), final restart of each arm
    jobs = []
    for r, cfg in RUNS.items():
        jobs.append(f'{C}/{cfg["src"]}/rst/{B}.{cfg["rst0"]}.rst')
        for x in ARMS:
            fs = sorted(glob.glob(f'{C}/S2{x}_{r}/rst/*.rst'))
            jobs.append(fs[-1])
    with Pool(min(a.nproc, len(jobs))) as p:
        Z = dict(zip(jobs, p.map(rst_pass, jobs)))
    fin = {}
    for r, cfg in RUNS.items():
        for x in ARMS:
            fin[(r, x)] = Z[sorted(glob.glob(f'{C}/S2{x}_{r}/rst/*.rst'))[-1]]
        fin[(r, '0')] = Z[f'{C}/{cfg["src"]}/rst/{B}.{cfg["rst0"]}.rst']

    # 1. dt
    rows.append('== 1. dt (code dt per cycle, median over windows; the final clipped step to tlim excluded)')
    verdict = {}
    for r, cfg in RUNS.items():
        for lo, hi in cfg['win']:
            med = {}
            for x in ARMS:
                a2 = L[(r, x)]['dt'][:-2]                    # the last two steps are clipped to tlim
                k = (a2[:, 0] >= lo) & (a2[:, 0] < hi)
                med[x] = np.median(a2[k, 1])
            g1, g2 = med['T'] / med['A'], med['T'] / med['B']
            verdict.setdefault(r, []).append(min(g1, g2))
            rows.append(f'  {r} t {lo:5.1f}-{hi:5.1f}: T {med["T"]:.4f}  A {med["A"]:.4f}  B {med["B"]:.4f}  '
                        f'T/A x{g1:.2f}  T/B x{g2:.2f}')
        for x in ARMS:
            d = L[(r, x)]['dt']
            v = d[:-2, 1]
            k = (d[:-2, 0] >= cfg['win'][0][0])
            rows.append(f'  {r} {x}: min dt {v.min():.4f} at t {d[np.argmin(v), 0]:.1f} (last two steps, clipped to '
                        f'tlim, excluded: {d[-2, 1]:.2e}, {d[-1, 1]:.2e}); cycles {len(d)}; cycles with dt < 0.6 x '
                        f'median {(v[k] < 0.6 * np.median(v[k])).mean():.3f}')
        nc = {x: len(L[(r, x)]['dt']) for x in ARMS}
        rows.append(f'  {r}: whole-window gain in cycles: A/T x{nc["A"] / nc["T"]:.2f}, B/T x{nc["B"] / nc["T"]:.2f}')
        ok = min(verdict[r]) >= cfg['gain']
        rows.append(f'  -> {r}: min window gain x{min(verdict[r]):.2f} vs required x{cfg["gain"]:.1f}: '
                    f'{"PASS" if ok else "FAIL"}')

    # 2/3. capture and wings (T)
    rows.append('== 2. capture check / 3. wings (T, zone-removed rest mass from ATMZONE_BINS, last bins line; '
                'net per bin = removal - addition: the W increase at fixed rho in floor cells adds rest mass)')
    for r in RUNS:
        tb, bins = L[(r, 'T')]['bins']
        tot_abs = np.abs(bins).sum()
        cap = np.abs(bins[:, :, :2]).sum() + np.abs(bins[0, :, 2:]).sum()
        pos, neg = bins[bins > 0].sum(), bins[bins < 0].sum()
        rows.append(f'  {r} (bins at t {tb:.1f}): net {bins.sum():+.3e}, positive bins {pos:+.3e}, negative bins '
                    f'{neg:+.3e}; capture fraction (|L| < 4 M or r < 1.88, by |bin|) {cap / max(tot_abs, 1e-300):.4f} '
                    f'(>= 0.99: {"PASS" if cap >= 0.99 * tot_abs else "FAIL"})')
        nc = (bins[1:, :, 2:]).sum(axis=(0, 2))
        rows.append('    non-capture remainder by E bin: ' + ' '.join(f'{ELAB[i]}:{v:+.1e}' for i, v in enumerate(nc)))
        eb = bins.sum(axis=(0, 2))
        for ref in ('B', 'A'):
            z = fin[(r, ref)]
            gm = np.histogram(z['E'] / DE, EB, weights=z['dM'])[0]
            fr = np.abs(eb) / np.maximum(gm, 1e-300)
            core = [i for i in range(10) if -3 <= EB[i] and EB[i + 1] <= 3]
            mid = [1, 8]
            ok = all(fr[i] <= 1e-4 for i in core) and all(fr[i] <= 1e-3 for i in mid)
            rows.append(f'    wings vs {ref} grid mass at t_end: |zone| / grid per E bin '
                        + ' '.join(f'{ELAB[i]}:{fr[i]:.1e}' for i in range(10))
                        + f'  (|E| <= 3dE <= 1e-4, 3-6dE <= 1e-3: {"PASS" if ok else "FAIL"})')

    # 4. Mdot
    import c044_mdot as M
    rows.append('== 4. Mdot (frozen-in from the final restart; remnant excluded, r_BH >= 3; Keplerian return '
                'from the release geodesic periapsis)')
    tc = np.sqrt(M.TB[1:] * M.TB[:-1])
    for r, cfg in RUNS.items():
        tp = M.t_peri(cfg['src'], cfg['rp'])
        md = {}
        for x in ARMS:
            z = fin[(r, x)]; deb = ~z['rem'].astype(bool)
            md[x] = M.mdot(z['E'][deb], z['dM'][deb], tp)
        for a1, a2, lab in (('T', 'B', 'zone effect T-B'), ('B', 'A', 'divergence effect B-A')):
            p1, p2 = md[a1].max(), md[a2].max()
            k = md[a2] >= 0.01 * p2
            dev = np.max(np.abs(md[a1][k] / md[a2][k] - 1))
            ok = abs(p1 / p2 - 1) <= 0.03 and dev <= 0.10
            rows.append(f'  {r} {lab}: peak {p1:.4e} vs {p2:.4e} ({p1 / p2 - 1:+.4f}); max |ratio - 1| where >= 1% '
                        f'of peak {dev:.4f} ({k.sum()} bins): {"PASS" if ok else "FAIL"}')

    # 5. closure
    rows.append('== 5. rest-mass booking closure: Delta(hst mass) + cum_dM_exc + zone cum_dM (cumulative since the '
                'restart; the final 1-<50 cycles after the last bins line are in cum_dM, which is per cycle)')
    for r in RUNS:
        for x in ARMS:
            h = np.loadtxt(glob.glob(f'{C}/S2{x}_{r}/*.mhd.hst')[0])
            z = L[(r, x)]['z']
            dm = h[-1, 2] - h[0, 2]
            ex, zn = z[-1, 9], z[-1, 6]
            res = dm + ex + zn
            rows.append(f'  {r} {x}: hst mass {h[0, 2]:.6e} -> {h[-1, 2]:.6e} (t {h[0, 0]:.2f}-{h[-1, 0]:.2f}), '
                        f'Delta {dm:+.3e}; excised {ex:+.3e}; zone {zn:+.3e}; residual {res:+.3e} '
                        f'({abs(res) / max(abs(dm), 1e-300):.2e} of |Delta|)')
    rows.append('  argued: the hst mass starts at the first history row after the restart (one hst dt later), and the '
                'excision/zone tallies at the first cycle; outflow through the outer boundary is not booked.')

    # 6. gate
    rows.append('== 6. gate (T)')
    for r in RUNS:
        z = L[(r, 'T')]['z']
        rows.append(f'  {r}: max n_viol {z[:, 5].max():.0f}; max over steps of max_rho_mod {z[:, 3].max():.3e} = '
                    f'{z[:, 3].max() / THR:.2e} rho_thr; max rho in the zone (incl. unmodified) {z[:, 4].max():.3e}; '
                    f'modified cells per step {np.median(z[:, 2]):.0f} (median): '
                    f'{"PASS" if z[:, 5].max() == 0 and z[:, 3].max() < THR else "FAIL"}')

    # 7. physics (final restarts) + slice K series at matched times
    rows.append('== 7. physics at the matched final restart (3D; dense = rho > 0.1 rho_c, r_BH >= 3)')
    lim = dict(K=0.003, M3=0.02, bound=0.003)
    for r in RUNS:
        ph = {x: physics(fin[(r, x)]) for x in ('0',) + ARMS}
        for x in ('0',) + ARMS:
            p_ = ph[x]
            rows.append(f'  {r} {"start" if x == "0" else x} t {p_["t"]:.1f}: dense K/K0 {p_["K"]:.5f} M3 {p_["M3"]:.4f} '
                        f'(>10% {p_["M10"]:.4f}) <E> {p_["E"]:+.4f} dE <Lz> {p_["Lz"]:.6f} bound {p_["bound"]:.5f} '
                        f'remnant {p_["rem"]:.4f} M* q10/25/50/75/90 ' + '/'.join(f'{v:+.3f}' for v in p_['q']) + ' dE'
                        f'; C2P errors {L[(r, x)]["c2p"] if x != "0" else "-"}')
        for a1, a2, lab in (('T', 'B', 'T-B'), ('B', 'A', 'B-A')):
            u, v = ph[a1], ph[a2]
            dq = np.max(np.abs(u['q'] - v['q']))
            dqx = np.max(np.abs(u['qx'] - v['qx']))      # q50 excluded: it sits in the E ~ 0 gap
            dl = abs(u['Lz'] - v['Lz']) / abs(v['Lz'])
            res = dict(K=abs(u['K'] / v['K'] - 1) <= lim['K'], M3=abs(u['M3'] - v['M3']) <= lim['M3'],
                       q=dq <= 0.01, q_no50=dqx <= 0.01, bound=abs(u['bound'] - v['bound']) <= lim['bound'],
                       rem=abs(u['rem'] / v['rem'] - 1) <= 0.005, Lz=dl <= 1e-5)
            rows.append(f'  {r} {lab}: dK/K {u["K"] / v["K"] - 1:+.2e} dM3 {u["M3"] - v["M3"]:+.4f} d<E> '
                        f'{u["E"] - v["E"]:+.4f} dE dLz/Lz {dl:.1e} dq max {dq:.4f} (q5-q40,q60-q95 {dqx:.4f}) dE dbound {u["bound"] - v["bound"]:+.5f} '
                        f'drem/rem {u["rem"] / v["rem"] - 1:+.4f} -> ' + ' '.join(f'{k}:{"ok" if ok else "OUT"}'
                                                                                   for k, ok in res.items()))
    rows.append('  slice series (prim_xy z = 0, matched integer times): dense K/K0 and warm area-mass fractions')
    for r, cfg in RUNS.items():
        ser = {}
        for x in ARMS:
            fs = sorted(glob.glob(f'{C}/S2{x}_{r}/bin/{B}.prim_xy.*.bin'))
            with Pool(min(a.nproc, 8)) as p:
                ser[x] = {round(t): (k, w5, w10) for t, k, w5, w10 in p.map(slice_K, fs)}
        ts = sorted(set(ser['T']) & set(ser['A']) & set(ser['B']))
        dTB = np.array([ser['T'][t][0] / ser['B'][t][0] - 1 for t in ts])
        dBA = np.array([ser['B'][t][0] / ser['A'][t][0] - 1 for t in ts])
        w5TB = np.array([ser['T'][t][1] - ser['B'][t][1] for t in ts])
        w5BA = np.array([ser['B'][t][1] - ser['A'][t][1] for t in ts])
        for t in ts[::max(1, len(ts) // 8)] + [ts[-1]]:
            rows.append(f'    {r} t {t:4d}: K/K0 T {ser["T"][t][0]:.5f} A {ser["A"][t][0]:.5f} B {ser["B"][t][0]:.5f};'
                        f' warm>5% T {ser["T"][t][1]:.4f} A {ser["A"][t][1]:.4f} B {ser["B"][t][1]:.4f}')
        rows.append(f'    {r} over {len(ts)} matched dumps: max |K_T/K_B - 1| {np.abs(dTB).max():.2e}, '
                    f'max |K_B/K_A - 1| {np.abs(dBA).max():.2e}; max |warm5 T-B| {np.abs(w5TB).max():.4f}, '
                    f'B-A {np.abs(w5BA).max():.4f}')

    # 8. tier 3
    rows.append('== 8. tier 3 (T)')
    for r in RUNS:
        t3 = L[(r, 'T')]['t3']
        if not len(t3):
            continue
        act = t3[:, 1] > 0.5
        rows.append(f'  {r}: active {act.sum()} of {len(t3)} cycles (first active t {t3[act, 0].min() if act.any() else np.nan:.1f}); '
                    f'd {t3[-1, 2]:.2f} R90 {t3[-1, 3]:.2f}; modified cells per active step (median) '
                    f'{np.median(t3[act, 5]) if act.any() else 0:.0f}; M_mod per step max {t3[:, 4].max():.3e}; '
                    f'max n_viol {t3[:, 6].max():.0f}; cum dS ({t3[-1, 7]:+.2e},{t3[-1, 8]:+.2e},{t3[-1, 9]:+.2e}) '
                    f'cum dE {t3[-1, 10]:+.2e}')
        if L[(r, 'T')]['t3bins'] is not None and act.any():
            tb, bins = L[(r, 'T')]['t3bins']
            eb = bins.sum(axis=(0, 2))
            z = fin[(r, 'B')]
            gm = np.histogram(z['E'] / DE, EB, weights=z['dM'])[0]
            fr_c = eb / np.maximum(gm, 1e-300); fr_s = fr_c / act.sum()
            ok = all(fr_s[i] <= 1e-6 for i in range(5, 10))
            rows.append('    modified mass / B grid mass per E bin, cumulative over steps: '
                        + ' '.join(f'{ELAB[i]}:{fr_c[i]:.1e}' for i in range(10)))
            rows.append('    ... per active step (cumulative / n_active; the same cells recur every step): '
                        + ' '.join(f'{ELAB[i]}:{fr_s[i]:.1e}' for i in range(10))
                        + f'  (E > 0 <= 1e-6 per step: {"PASS" if ok else "FAIL"}; cumulative, a strict upper bound: '
                        f'{"PASS" if all(fr_c[i] <= 1e-6 for i in range(5, 10)) else "FAIL"})')

    # 9. B - A: switch census
    import d_r043_maps as R
    rows.append('== 9. divergence effect B - A (and T): switch census on the dediag_xy slices '
                '(fractions of dense cells, mean over dumps)')
    for r, cfg in RUNS.items():
        cen = {}
        for x in ARMS:
            d = f'{C}/S2{x}_{r}'
            ns = sorted(int(f.split('.')[-2]) for f in glob.glob(f'{d}/bin/{B}.dediag_xy.*.bin'))
            with Pool(min(a.nproc, 8)) as p:
                cen[x] = np.array(p.starmap(_census, [(d, n) for n in ns]))
        wins = ((113.0, 136.0), (136.0, 170.1)) if r == 'R6' else ((253.0, 300.1),)
        for lo, hi in wins:
            line = f'  {r} t {lo:.0f}-{hi:.0f}:'
            for x in ARMS:
                A_ = cen[x]; k = (A_[:, 0] >= lo) & (A_[:, 0] < hi) & (A_[:, 1] > 0)
                nd = A_[k, 1]
                line += (f'  {x} flagged {np.nanmean(A_[k, 18]):.4f} energy-branch {np.mean((A_[k, 5] + A_[k, 6]) / nd):.4f}'
                         f' guard {np.mean(A_[k, 3] / nd):.4f} <Ken/Ktr-1>ent {np.nanmean(A_[k, 8]):+.1e} |')
            rows.append(line)

    # 10. cost
    rows.append('== 10. cost: PHASE_TIMERS_FINAL (rank 0, fenced) and node-h (ensemble.out elapsed x nodes)')
    for r, cfg in RUNS.items():
        for x in ARMS:
            tm = L[(r, x)]['timers']; w = tm['wall']
            rows.append(f'  {r} {x}: wall {w:.0f} s, node-h {L[(r, x)]["elapsed"] * cfg["nodes"] / 3600:.1f}, cycles '
                        f'{tm["cycles"]:.0f}; mg {tm["mg"] / w:.1%} hydro_etc {tm["hydro_etc"] / w:.1%} io {tm["io"] / w:.1%} '
                        f'regrid {tm["regrid"] / w:.1%} newdt {tm["newdt"] / w:.2%} untimed {(w - tm["timed"]) / w:.1%}')
    open(f'{a.out}/c051_s2.txt', 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
