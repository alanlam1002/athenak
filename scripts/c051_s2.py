#!/usr/bin/env python3
"""C-051 (R-058 §2 / R-063 §3, §5 / R-064 step 2): evaluation of the S2 restart tests.

Arms (one 16-node bundle each; R6 restart 00006 t = 113.7 -> 170 on 7 nodes, R7 restart
00013 t = 253.7 -> 300 on 9 nodes; binary build_gpu_pe2626_zone d21072a5):
  T = zone on + dual_energy_div covariant (+ tier 3, rule-gated)
  A = zone off + coord      (atmosphere_zone_book: booking only)
  B = zone off + covariant  (atmosphere_zone_book)
The zone effect is T - B, the divergence effect B - A.

Criteria (numbers per R-058 §2, revised per R-065 §3 / R-067 Q2 / R-069 item 6): dt gain;
capture check on the zone-removed mass; wings; Mdot (frozen-in, final restart): cumulative
R-076 item 2 criteria (R-065 §3 as revised by R-070 Q6): cumulative F(E) over ALL bound gas (E < 0,
remnant included), each arm normalised by its own bound mass, KS max|F_1 - F_2| <= 0.01; the debris
fallback times (remnant excluded; Keplerian return) at which M_fb reaches 1% / 10% of the debris bound
mass within 5% and 50% (t_half) within 3%; peak Mdot within 3% (t_peak dropped); bound and unbound
quantile sets q10-q90 each within 0.01 dE; T-B tolerances max(tol, 2 |B - A|); rest-mass closure
<= 1e-10 M* per segment, from the run's MLEDGER lines when present (update non-conservation
sum(Y + Fout) + between-cycle changes), else from hst + booked tallies.
C-081 / R-084 working tolerances (2 sigma_diff of the three R6 noise samples) replace these for t_half
(9.7%), peak Mdot (3.6%), remnant (1.7%), quantiles (0.027 bound / 0.033 unbound) and add debris bound
mass (8.3%); the thermal rows are unchanged.

    c051_s2.py [--runs 'R6:T=S2T_R6,A=S2A_R6,B=S2B_R6;R7:T=S2T_R7,A=S2A_R7,B=S2B_R7'] [--nproc 6]
               [--out DIR] [--name c051_s2_r076.txt]
    (run labels R6, R7, L5, Q1c; arm dirs relative to the cfc run dir or absolute; any subset of T/A/B)
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
CLOSE_TOL = 1e-10 * MSTAR                         # R-076 item 2: 1e-10 M* per segment, in M_BH units
QS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
EB = np.array([-np.inf, -6, -3, -2, -1, 0, 1, 2, 3, 6, np.inf])
ELAB = ['<-6', '-6..-3', '-3..-2', '-2..-1', '-1..0', '0..1', '1..2', '2..3', '3..6', '>6']
RUNS = {'R6': dict(src='R6_rp10_comov_L4_fix', rst0='00006', rp=10.0, nodes=7, tend=170.0,
                   win=((113.7, 136.0), (136.0, 150.0), (150.0, 170.0)), gain=1.4),
        'R7': dict(src='R7_rp18_comov_L4_fix', rst0='00013', rp=18.0, nodes=9, tend=300.0,
                   win=((253.7, 275.0), (275.0, 300.0)), gain=2.0),
        'L5': dict(src='L5c_rp10_comov_t200', rst0='00006', rp=10.0, nodes=10, tend=170.0,
                   win=((108.8, 136.0), (136.0, 150.0), (150.0, 170.0)), gain=1.4),
        'Q1c': dict(src='RQ1_bh_L4_t300_from_De', rst0='00006', rp=10.0, nodes=16, tend=170.0,
                    win=((123.8, 136.0), (136.0, 150.0), (150.0, 170.0)), gain=1.4)}
CWIN = {'R6': ((113.0, 136.0), (136.0, 170.1)), 'L5': ((108.0, 136.0), (136.0, 170.1)),
        'Q1c': ((123.0, 136.0), (136.0, 170.1)), 'R7': ((253.0, 300.1),)}
DEFAULT_RUNS = 'R6:T=S2T_R6,A=S2A_R6,B=S2B_R6;R7:T=S2T_R7,A=S2A_R7,B=S2B_R7'
CACHE = f'{P}/c051_cache'


# ---------------------------------------------------------------- logs
def parse(d):
    """ensemble.out: dt series, ATMZONE rows, last bins, T3 rows/bins, timers, C2P errors"""
    o = dict(dt=[], z=[], bins=None, t3=[], t3bins=None, timers=None, c2p=0, elapsed=0.0, led=[])
    fn = f'{d}/ensemble.out' if os.path.exists(f'{d}/ensemble.out') else f'{d}/out.log'
    for ln in open(fn):
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
        elif ln.startswith('MLEDGER '):
            kv = dict(re.findall(r'(\w+)=([^\s,]+)', ln))
            o['led'].append([float(kv[k]) for k in ('cycle', 'time', 'Mu1', 'Mend', 'Y', 'Fout', 'X')])
        elif 'An error occurred during the primitive solve' in ln:
            o['c2p'] += 1
    for k in ('dt', 'z', 't3', 'led'):
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
                qb=quant(z['E'][deb & (z['E'] < 0)], z['dM'][deb & (z['E'] < 0)], QS) / DE,
                qu=quant(z['E'][deb & (z['E'] > 0)], z['dM'][deb & (z['E'] > 0)], QS) / DE)


def arm_pairs(arms):
    """the compared pairs: T-B (zone) and B-A (gate) when present, else T-A"""
    p = [q for q in (('T', 'B'), ('B', 'A')) if q[0] in arms and q[1] in arms]
    if not p and 'T' in arms and 'A' in arms:
        p = [('T', 'A')]
    return p


def scaled(tol, ba):
    """R-065 §3: a zone-effect tolerance scaled by the control pair, max(tol, 2 |B - A|)"""
    return max(tol, 2.0 * abs(ba))


def fallback(z, tp):
    """R-076 item 2: F(E) over ALL bound gas (E < 0, remnant included), E-sorted and normalised by
    this arm's own bound mass; the debris (remnant excluded) bound mass, and the Keplerian return
    times (as c044_mdot.mdot; per cell, unbinned) at which the debris M_fb(< t) reaches 1% / 10% /
    50% (t_half) of it. Returns (E, F/M_b_all, M_b_all, times[3], M_b_debris)"""
    b = z['E'] < 0
    E, m = z['E'][b], z['dM'][b]
    o = np.argsort(E)
    Mall = m.sum()
    deb = ~z['rem'].astype(bool) & b
    Ed, md = z['E'][deb], z['dM'][deb]
    Mb = md.sum()
    tr = tp + 2 * np.pi * (-2 * Ed)**-1.5
    ot = np.argsort(tr)
    cf = np.cumsum(md[ot]) / Mb
    return E[o], np.cumsum(m[o]) / Mall, Mall, np.interp([0.01, 0.10, 0.50], cf, tr[ot]), Mb


def ks(f1, f2):
    """max |F_1(E) - F_2(E)| on the union of both E sets, each F normalised by its own bound mass"""
    E1, F1 = f1[0], f1[1]
    E2, F2 = f2[0], f2[1]
    g = np.union1d(E1, E2)
    a = np.interp(g, E1, F1, left=0.0, right=1.0)
    b = np.interp(g, E2, F2, left=0.0, right=1.0)
    return float(np.max(np.abs(a - b)))


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
    ap.add_argument('--name', default='c051_s2_r076.txt')
    ap.add_argument('--runs', default=DEFAULT_RUNS)
    a = ap.parse_args()
    rows = ['C-051 / R-069 item 6: S2 evaluation with the revised criteria (c051_s2.py). T = zone on + covariant (+ tier 3), A = zone off + coord, '
            'B = zone off + covariant. Measured unless marked argued. Masses in M_BH units unless "M*".']
    RS, AR, DIR = {}, {}, {}
    for spec in a.runs.split(';'):
        lab, arms = spec.split(':')
        RS[lab] = RUNS[lab]
        got = dict(kv.split('=') for kv in arms.split(','))
        AR[lab] = tuple(x for x in ('T', 'A', 'B') if x in got)
        for x in AR[lab]:
            DIR[(lab, x)] = got[x] if got[x].startswith('/') else f'{C}/{got[x]}'
    rows.append('runs: ' + '; '.join(f'{r} ' + ', '.join(f'{x}={DIR[(r, x)]}' for x in AR[r]) for r in RS))
    L = {(r, x): parse(DIR[(r, x)]) for r in RS for x in AR[r]}
    # 3D passes: start restart (shared), final restart of each arm
    jobs = []
    for r, cfg in RS.items():
        jobs.append(f'{C}/{cfg["src"]}/rst/{B}.{cfg["rst0"]}.rst')
        for x in AR[r]:
            fs = sorted(glob.glob(f'{DIR[(r, x)]}/rst/*.rst'))
            jobs.append(fs[-1])
    with Pool(min(a.nproc, len(jobs))) as p:
        Z = dict(zip(jobs, p.map(rst_pass, jobs)))
    fin = {}
    for r, cfg in RS.items():
        for x in AR[r]:
            fin[(r, x)] = Z[sorted(glob.glob(f'{DIR[(r, x)]}/rst/*.rst'))[-1]]
        fin[(r, '0')] = Z[f'{C}/{cfg["src"]}/rst/{B}.{cfg["rst0"]}.rst']

    # 1. dt
    rows.append('== 1. dt (code dt per cycle, median over windows; the final clipped step to tlim excluded)')
    verdict = {}
    for r, cfg in RS.items():
        for lo, hi in cfg['win']:
            med = {}
            for x in AR[r]:
                a2 = L[(r, x)]['dt'][:-2]                    # the last two steps are clipped to tlim
                k = (a2[:, 0] >= lo) & (a2[:, 0] < hi)
                med[x] = np.median(a2[k, 1]) if k.any() else np.nan
            line = f'  {r} t {lo:5.1f}-{hi:5.1f}: ' + '  '.join(f'{x} {med[x]:.4f}' for x in AR[r])
            if 'T' in med:
                gs = [med['T'] / med[c] for c in ('A', 'B') if c in med]
                if gs:
                    verdict.setdefault(r, []).append(min(gs))
                    line += '  ' + '  '.join(f'T/{c} x{med["T"] / med[c]:.2f}' for c in ('A', 'B') if c in med)
            rows.append(line)
        for x in AR[r]:
            d = L[(r, x)]['dt']
            v = d[:-2, 1]
            k = (d[:-2, 0] >= cfg['win'][0][0])
            rows.append(f'  {r} {x}: min dt {v.min():.4f} at t {d[np.argmin(v), 0]:.1f} (last two steps, clipped to '
                        f'tlim, excluded: {d[-2, 1]:.2e}, {d[-1, 1]:.2e}); cycles {len(d)}; cycles with dt < 0.6 x '
                        f'median {(v[k] < 0.6 * np.median(v[k])).mean():.3f}')
        nc = {x: len(L[(r, x)]['dt']) for x in AR[r]}
        if r in verdict:
            rows.append(f'  {r}: whole-window gain in cycles: ' + ', '.join(f'{c}/T x{nc[c] / nc["T"]:.2f}'
                                                                           for c in ('A', 'B') if c in nc))
            ok = min(verdict[r]) >= cfg['gain']
            rows.append(f'  -> {r}: min window gain x{min(verdict[r]):.2f} vs required x{cfg["gain"]:.1f}: '
                        f'{"PASS" if ok else "FAIL"}')

    # 2/3. capture and wings (T)
    rows.append('== 2. capture check / 3. wings (T, zone-removed rest mass from ATMZONE_BINS, last bins line; '
                'net per bin = removal - addition: the W increase at fixed rho in floor cells adds rest mass)')
    for r in RS:
        if 'T' not in AR[r] or L[(r, 'T')]['bins'] is None:
            continue
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
        for ref in [c for c in ('B', 'A') if c in AR[r]]:
            z = fin[(r, ref)]
            gm = np.histogram(z['E'] / DE, EB, weights=z['dM'])[0]
            fr = np.abs(eb) / np.maximum(gm, 1e-300)
            core = [i for i in range(10) if -3 <= EB[i] and EB[i + 1] <= 3]
            mid = [1, 8]
            ok = all(fr[i] <= 1e-4 for i in core) and all(fr[i] <= 1e-3 for i in mid)
            rows.append(f'    wings vs {ref} grid mass at t_end: |zone| / grid per E bin '
                        + ' '.join(f'{ELAB[i]}:{fr[i]:.1e}' for i in range(10))
                        + f'  (|E| <= 3dE <= 1e-4, 3-6dE <= 1e-3: {"PASS" if ok else "FAIL"})')

    # 4. Mdot / fallback (R-076 item 2)
    import c044_mdot as M
    rows.append('== 4. fallback (frozen-in from the final restart; r_BH >= 3; Keplerian return from the release geodesic '
                'periapsis). R-076 item 2: F(E) over ALL bound gas (remnant included), each arm normalised by its own '
                'bound mass, KS <= 0.01; debris (remnant excluded) M_fb at 1% / 10% of its bound mass within 5%, t_half '
                '(50%) within 9.7%; peak Mdot within 3.6% (t_peak dropped); debris bound mass within 8.3% '
                '(C-081 / R-084 working tolerances); T-B tolerances max(tol, 2|B-A|)')
    tc = np.sqrt(M.TB[1:] * M.TB[:-1])
    for r, cfg in RS.items():
        tp = M.t_peri(cfg['src'], cfg['rp'])
        md, fb = {}, {}
        for x in AR[r]:
            z = fin[(r, x)]; deb = ~z['rem'].astype(bool)
            md[x] = M.mdot(z['E'][deb], z['dM'][deb], tp)
            fb[x] = fallback(z, tp)
        pairs = arm_pairs(AR[r])
        met = {}
        for a1, a2 in pairs:
            met[(a1, a2)] = dict(ks=ks(fb[a1], fb[a2]), f1=fb[a1][3][0] / fb[a2][3][0] - 1,
                                 f10=fb[a1][3][1] / fb[a2][3][1] - 1, thalf=fb[a1][3][2] / fb[a2][3][2] - 1,
                                 pk=md[a1].max() / md[a2].max() - 1, mb=fb[a1][2] / fb[a2][2] - 1,
                                 mbd=fb[a1][4] / fb[a2][4] - 1)
        tol = dict(ks=0.01, f1=0.05, f10=0.05,
                   thalf=0.097,  # C-081 / R-084 working tolerance (2σ_diff, n = 3, ±50%)
                   pk=0.036,  # C-081 / R-084 working tolerance (2σ_diff, n = 3, ±50%)
                   mbd=0.083)  # C-081 / R-084 working tolerance (2σ_diff, n = 3, ±50%)
        for a1, a2 in pairs:
            u = met[(a1, a2)]
            ba = met.get(('B', 'A'))
            res = {}
            for k_, t_ in tol.items():
                tt = scaled(t_, ba[k_]) if (a1 == 'T' and ba is not None) else t_
                res[k_] = (abs(u[k_]) <= tt, tt)
            lab = {'T': 'zone', 'B': 'gate'}.get(a1, '') + f' effect {a1}-{a2}' if (a1, a2) != ('T', 'A') else 'T-A'
            rows.append(f'  {r} {lab}: KS max|dF| {u["ks"]:.4f}; t(M_fb 1%) {u["f1"]:+.4f}, t(10%) {u["f10"]:+.4f}, '
                        f't_half {u["thalf"]:+.4f}; peak {u["pk"]:+.4f}; M_bound(all) {u["mb"]:+.4f}, '
                        f'(debris) {u["mbd"]:+.4f} -> '
                        + ' '.join(f'{k_}:{"ok" if ok else "OUT"}(tol {tt:.3g})' for k_, (ok, tt) in res.items())
                        + f'  => {"PASS" if all(ok for ok, _ in res.values()) else "FAIL"}')
        for x in AR[r]:
            rows.append(f'    {r} {x}: M_bound all {fb[x][2] / MSTAR:.5f} M*, debris {fb[x][4] / MSTAR:.5f} M*; '
                        f't(M_fb 1%) {fb[x][3][0]:.4g} M, t(10%) {fb[x][3][1]:.4g} M, t_half {fb[x][3][2]:.4g} M; '
                        f'peak Mdot {md[x].max():.4e} M*/M at t {tc[np.argmax(md[x])]:.4g} M')

    # 5. closure
    rows.append('== 5. rest-mass closure, R-076 item 2: <= 1e-10 M* per segment. From MLEDGER when present: unbooked = '
                'sum(Y + Fout) (the update\'s non-conservation) + between-cycle changes (M(u1) - previous M_end); '
                'booked = outer outflow + C2P (floors, zone, excision). Else hst + booked tallies (excision, zone).')
    for r in RS:
        for x in AR[r]:
            led = L[(r, x)]['led']
            if len(led):
                Y, Fo, X = led[:, 4], led[:, 5], led[:, 6]
                G = led[1:, 2] - led[:-1, 3]
                dm = led[-1, 3] - led[0, 2]
                res = (Y + Fo).sum() + G.sum()
                rows.append(f'  {r} {x} (ledger, {len(led)} cycles, t {led[0, 1]:.2f}-{led[-1, 1]:.2f}): grid dM {dm:+.3e} = '
                            f'-outflow {-Fo.sum():+.3e} + C2P {X.sum():+.3e} + unbooked {res:+.3e} '
                            f'[update {(Y + Fo).sum():+.2e}, between cycles {G.sum():+.2e}] = {res / MSTAR:+.2e} M* '
                            f'(tol 1e-10 M* = {CLOSE_TOL:.2e}): {"PASS" if abs(res) <= CLOSE_TOL else "FAIL"}')
                continue
            hs = glob.glob(f'{DIR[(r, x)]}/*.mhd.hst')
            z = L[(r, x)]['z']
            if not hs or not len(z):
                rows.append(f'  {r} {x}: no ledger and no ATMZONE booking -- closure not evaluated')
                continue
            h = np.loadtxt(hs[0])
            dm = h[-1, 2] - h[0, 2]
            ex, zn = z[-1, 9], z[-1, 6]
            res = dm + ex + zn
            rows.append(f'  {r} {x} (hst + tallies): hst mass {h[0, 2]:.6e} -> {h[-1, 2]:.6e} (t {h[0, 0]:.2f}-{h[-1, 0]:.2f}), '
                        f'Delta {dm:+.3e}; excised {ex:+.3e}; zone {zn:+.3e}; residual {res:+.3e} '
                        f'= {res / MSTAR:+.2e} M* (tol 1e-10 M* = {CLOSE_TOL:.2e} per segment): '
                        f'{"PASS" if abs(res) <= CLOSE_TOL else "FAIL"}')
    rows.append('  argued (hst path): the hst mass starts at the first history row after the restart, the tallies at the '
                'first cycle, and outflow through the outer boundary is not booked.')

    # 6. gate
    rows.append('== 6. gate (T)')
    for r in RS:
        if 'T' not in AR[r] or not len(L[(r, 'T')]['z']):
            continue
        z = L[(r, 'T')]['z']
        rows.append(f'  {r}: max n_viol {z[:, 5].max():.0f}; max over steps of max_rho_mod {z[:, 3].max():.3e} = '
                    f'{z[:, 3].max() / THR:.2e} rho_thr; max rho in the zone (incl. unmodified) {z[:, 4].max():.3e}; '
                    f'modified cells per step {np.median(z[:, 2]):.0f} (median): '
                    f'{"PASS" if z[:, 5].max() == 0 and z[:, 3].max() < THR else "FAIL"}')

    # 7. physics (final restarts) + slice K series at matched times
    rows.append('== 7. physics at the matched final restart (3D; dense = rho > 0.1 rho_c, r_BH >= 3)')
    lim = dict(K=0.003, M3=0.02, bound=0.003)
    for r in RS:
        ph = {x: physics(fin[(r, x)]) for x in ('0',) + AR[r]}
        for x in ('0',) + AR[r]:
            p_ = ph[x]
            rows.append(f'  {r} {"start" if x == "0" else x} t {p_["t"]:.1f}: dense K/K0 {p_["K"]:.5f} M3 {p_["M3"]:.4f} '
                        f'(>10% {p_["M10"]:.4f}) <E> {p_["E"]:+.4f} dE <Lz> {p_["Lz"]:.6f} bound {p_["bound"]:.5f} '
                        f'remnant {p_["rem"]:.4f} M*'
                        f'; C2P errors {L[(r, x)]["c2p"] if x != "0" else "-"}')
        dif = {}
        for a1, a2 in arm_pairs(AR[r]):
            u, v = ph[a1], ph[a2]
            dif[(a1, a2)] = dict(K=u['K'] / v['K'] - 1, M3=u['M3'] - v['M3'], bound=u['bound'] - v['bound'],
                                 rem=u['rem'] / v['rem'] - 1, Lz=abs(u['Lz'] - v['Lz']) / abs(v['Lz']),
                                 qb=np.max(np.abs(u['qb'] - v['qb'])), qu=np.max(np.abs(u['qu'] - v['qu'])),
                                 E=u['E'] - v['E'])
        tol = dict(K=lim['K'], M3=lim['M3'], bound=lim['bound'],
                   rem=0.017,  # C-081 / R-084 working tolerance (2σ_diff, n = 3, ±50%)
                   Lz=1e-5,
                   qb=0.027,  # C-081 / R-084 working tolerance (2σ_diff, n = 3, ±50%)
                   qu=0.033)  # C-081 / R-084 working tolerance (2σ_diff, n = 3, ±50%)
        for a1, a2 in arm_pairs(AR[r]):
            lab = f'{a1}-{a2}'
            u, ba = dif[(a1, a2)], dif.get(('B', 'A'))
            res = {}
            for k_, t_ in tol.items():
                tt = scaled(t_, ba[k_]) if (a1 == 'T' and ba is not None) else t_
                res[k_] = (abs(u[k_]) <= tt, tt)
            rows.append(f'  {r} {lab}: dK/K {u["K"]:+.2e} dM3 {u["M3"]:+.4f} d<E> {u["E"]:+.4f} dE dLz/Lz {u["Lz"]:.1e} '
                        f'dq bound max {u["qb"]:.4f} unbound max {u["qu"]:.4f} dE dbound {u["bound"]:+.5f} '
                        f'drem/rem {u["rem"]:+.4f} -> ' + ' '.join(f'{k_}:{"ok" if ok else "OUT"}(tol {tt:.3g})'
                                                                   for k_, (ok, tt) in res.items())
                        + (f'  => {"PASS" if all(ok for ok, _ in res.values()) else "FAIL"}' if a1 == 'T' else ''))
        for x in AR[r]:
            rows.append(f'    {r} {x} bound q10/25/50/75/90 ' + '/'.join(f'{v:+.3f}' for v in ph[x]['qb'])
                        + ' unbound ' + '/'.join(f'{v:+.3f}' for v in ph[x]['qu']) + ' dE')
    rows.append('  slice series (prim_xy z = 0, matched integer times): dense K/K0 and warm area-mass fractions')
    for r, cfg in RS.items():
        ser = {}
        for x in AR[r]:
            fs = sorted(glob.glob(f'{DIR[(r, x)]}/bin/{B}.prim_xy.*.bin'))
            with Pool(min(a.nproc, 8)) as p:
                ser[x] = {round(t): (k, w5, w10) for t, k, w5, w10 in p.map(slice_K, fs)}
        arms = AR[r]
        ts = sorted(set.intersection(*[set(ser[x]) for x in arms]))
        if not ts:
            continue
        for t in ts[::max(1, len(ts) // 8)] + [ts[-1]]:
            rows.append(f'    {r} t {t:4d}: K/K0 ' + ' '.join(f'{x} {ser[x][t][0]:.5f}' for x in arms)
                        + '; warm>5% ' + ' '.join(f'{x} {ser[x][t][1]:.4f}' for x in arms))
        line = f'    {r} over {len(ts)} matched dumps:'
        for a1, a2 in arm_pairs(arms):
            dK = np.array([ser[a1][t][0] / ser[a2][t][0] - 1 for t in ts])
            dw = np.array([ser[a1][t][1] - ser[a2][t][1] for t in ts])
            line += f' max |K_{a1}/K_{a2} - 1| {np.abs(dK).max():.2e}, max |warm5 {a1}-{a2}| {np.abs(dw).max():.4f};'
        rows.append(line)

    # 8. tier 3
    rows.append('== 8. tier 3 (T)')
    for r in RS:
        if 'T' not in AR[r]:
            continue
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
            z = fin[(r, 'B' if 'B' in AR[r] else AR[r][-1])]
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
    for r, cfg in RS.items():
        cen = {}
        for x in AR[r]:
            d = f'{DIR[(r, x)]}'
            ns = sorted(int(f.split('.')[-2]) for f in glob.glob(f'{d}/bin/{B}.dediag_xy.*.bin'))
            with Pool(min(a.nproc, 8)) as p:
                cen[x] = np.array(p.starmap(_census, [(d, n) for n in ns]))
        wins = CWIN[r]
        for lo, hi in wins:
            line = f'  {r} t {lo:.0f}-{hi:.0f}:'
            for x in AR[r]:
                A_ = cen[x]; k = (A_[:, 0] >= lo) & (A_[:, 0] < hi) & (A_[:, 1] > 0)
                nd = A_[k, 1]
                line += (f'  {x} flagged {np.nanmean(A_[k, 18]):.4f} energy-branch {np.mean((A_[k, 5] + A_[k, 6]) / nd):.4f}'
                         f' guard {np.mean(A_[k, 3] / nd):.4f} <Ken/Ktr-1>ent {np.nanmean(A_[k, 8]):+.1e} |')
            rows.append(line)

    # 10. cost
    rows.append('== 10. cost: PHASE_TIMERS_FINAL (rank 0, fenced) and node-h (ensemble.out elapsed x nodes)')
    for r, cfg in RS.items():
        for x in AR[r]:
            tm = L[(r, x)]['timers']
            if tm is None:
                rows.append(f'  {r} {x}: no PHASE_TIMERS_FINAL'); continue
            w = tm['wall']
            rows.append(f'  {r} {x}: wall {w:.0f} s, node-h {L[(r, x)]["elapsed"] * cfg["nodes"] / 3600:.1f}, cycles '
                        f'{tm["cycles"]:.0f}; mg {tm["mg"] / w:.1%} hydro_etc {tm["hydro_etc"] / w:.1%} io {tm["io"] / w:.1%} '
                        f'regrid {tm["regrid"] / w:.1%} newdt {tm["newdt"] / w:.2%} untimed {(w - tm["timed"]) / w:.1%}')
    open(f'{a.out}/{a.name}', 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
