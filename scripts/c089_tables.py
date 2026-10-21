#!/usr/bin/env python3
"""C-089 (R-092 item 4): tables from the c089_hot.py 3D caches and the c089_slices.py slice cache.

Band = C-086's sub-switch gas: 1e-9 rho_c < D <= dual_energy_rho_switch (3.2e-8), r_BH >= 3 (c086_rst.py:177-185).
Hot = K/K0 > 100 with K = P/rho^Gamma from the restart's conserved state (tde_dmde.c2p).
Orbit: bound = BH-frame Killing E < 0 (tde_dmde.py:164-167, as C-086 item 9 / remnant split);
capture (project criterion) = |L| < 4 M, the |L| edge the zone's capture check uses (c051_s2.py:276-278 with the
zone's |L| bins, primitive_solver_hyd.hpp:2263-2265, 2286); capture (E-dependent, ARGUED) = L < L_c(eps) of a test
particle on Schwarzschild with eps = 1 + E: L_c from max_r V_eff(L) = eps^2, V_eff = (1 - 2/r)(1 + L^2/r^2), and
L_c = sqrt(12) for eps^2 < 8/9. L, E are coordinate (CFC) quantities, not Schwarzschild ones.
Floor (c): P/rho < 1e-14 in 3D (TFL_PROXY; validated on the slices against the stored temperature == tfloor).

    c089_tables.py --cache DIR --slices FILE [--out DIR]
"""
import argparse
import glob
import os
import pickle

import numpy as np

MSTAR, RHOC = 7.47e-5, 3.24e-4
REDG = [3, 5, 8, 15, 30, np.inf]
RLAB = ['3-5', '5-8', '8-15', '15-30', '>30']
KEDG = [-np.inf, 1e-6, 0.5, 2, 10, 1e2, 1e3, 1e4, 1e5, 1e6, np.inf]
KLAB = ['<1e-6(floor)', '1e-6-0.5', '0.5-2', '2-10', '10-1e2', '1e2-1e3', '1e3-1e4', '1e4-1e5', '1e5-1e6', '>1e6']
DEDG = [0, 1.01, 2.01, 4.01, 8.01, np.inf]
DLAB = ['<=1', '1-2', '2-4', '4-8', '>8/none']
TFL = 1e-14


def Lcrit(eps):
    L = np.linspace(np.sqrt(12) + 1e-9, 40, 20000)
    rm = (L * L - L * np.sqrt(L * L - 12)) / 2                 # the potential maximum (inner root)
    Vm = (1 - 2 / rm) * (1 + L * L / rm**2)
    e2 = np.asarray(eps, float)**2
    out = np.interp(e2, Vm, L, left=np.sqrt(12), right=40.0)
    return np.where(e2 < 8 / 9, np.sqrt(12), out)


def load(cache):
    R = {}
    for f in sorted(glob.glob(f'{cache}/*.npz')):
        z = np.load(f)
        R[(str(z['lab']), str(z['rdir']), str(z['num']))] = z
    return R


def pct(x, w, q=(10, 50, 90)):
    if x.size == 0:
        return 'n/a'
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return '/'.join(f'{x[o][min(np.searchsorted(c, qq / 100), x.size - 1)]:.2e}' for qq in q)


def fr(a, b):
    return a / b if b > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--slices', required=True)
    ap.add_argument('--out', default='/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/plots/phase2_m1')
    a = ap.parse_args()
    R = load(a.cache)
    S = pickle.load(open(a.slices, 'rb'))
    L = ['C-089 (R-092 item 4): localisation of the hot sub-switch mass, the matched L4/L5 test, the distinct floor mass.',
         'scripts/c089_hot.py (3D extraction) + c089_slices.py (z = 0 slices) -> c089_tables.py. MEASURED unless marked.',
         'Band = 1e-9 rho_c < D <= 3.2e-8, r_BH >= 3; hot = K/K0 > 100; masses in M* unless stated. "f" = share of the',
         'band mass (or of the stated parent). Slice masses are the AREA proxy sqrt(gamma) D dx^2 (shares only).', '']

    def prep(z):
        d = {k: z[k] for k in ('dM', 'KK0', 'PoR', 'rbh', 'E', 'L', 'vr', 'psiv', 'dv2', 'C', 'glnr', 'd_atm', 'd_flr',
                               'd_tfl', 'd_tflb', 'd_lev', 'flag', 'lev', 'D')}
        d['hot'] = d['KK0'] > 100
        d['flr'] = d['PoR'] < TFL
        d['bound'] = d['E'] < 0
        d['cap4'] = d['L'] < 4
        d['capE'] = d['L'] < Lcrit(1 + d['E'])
        return d

    # ---------------- (a) localisation, per restart
    order = [('R6', 'R6_rp10_comov_L4_fix', n) for n in ('00006', '00008', '00010', '00013')] + \
            [('R6', 'R6x_rp10_comov_L4_t300', n) for n in ('00017', '00022')] + \
            [('R7', 'R7_rp18_comov_L4_fix', n) for n in ('00007', '00016')] + \
            [('R7', 'R7x_rp18_comov_L4_t500', n) for n in ('00022', '00028')] + \
            [('L5', 'L5c_rp10_comov_t200', n) for n in ('00008', '00010', '00013')]
    order = [k for k in order if k in R]
    P = {k: prep(R[k]) for k in R}
    L += ['== (a1) Band and hot mass; K/K0 distribution (mass share of the band per K/K0 bin)',
          f'{"run":4s} {"t":>6s} {"M_band":>9s} {"M_hot":>9s} {"f_hot":>6s} {"f_flag":>6s} {"hot&flag/flag":>13s} | ' + ' '.join(f'{k:>9s}' for k in KLAB)]
    for k in order:
        d, z = P[k], R[k]
        M = d['dM'].sum()
        kh = np.histogram(d['KK0'], KEDG, weights=d['dM'])[0] / M
        L.append(f'{k[0]:4s} {float(z["t"]):6.1f} {M / MSTAR:9.2e} {d["dM"][d["hot"]].sum() / MSTAR:9.2e} '
                 f'{fr(d["dM"][d["hot"]].sum(), M):6.3f} {fr(d["dM"][d["flag"]].sum(), M):6.3f} '
                 f'{fr(d["dM"][d["hot"] & d["flag"]].sum(), d["dM"][d["flag"]].sum()):13.3f} | ' + ' '.join(f'{x:9.3f}' for x in kh))
    L.append('')
    L += ['== (a2) by r_BH: hot mass share per bin (of all hot mass) | hot fraction of the band in that bin | median log10 K/K0 of hot',
          f'{"run":4s} {"t":>6s} | ' + ' '.join(f'{r:>17s}' for r in RLAB)]
    for k in order:
        d, z = P[k], R[k]
        Mh = d['dM'][d['hot']].sum()
        ib = np.digitize(d['rbh'], REDG) - 1
        cells = []
        for b in range(5):
            s = ib == b
            mh = d['dM'][s & d['hot']].sum(); mb = d['dM'][s].sum()
            kk = np.log10(d['KK0'][s & d['hot']]) if (s & d['hot']).any() else np.array([np.nan])
            cells.append(f'{fr(mh, Mh):5.3f} {fr(mh, mb):5.3f} {np.median(kk):4.1f}')
        L.append(f'{k[0]:4s} {float(z["t"]):6.1f} | ' + ' '.join(f'{c:>17s}' for c in cells))
    L.append('')
    L += ['== (a3) by orbit: share of HOT mass (and hot fraction of the band in the class). cap4 = |L| < 4 (project);',
          '   capE = L < L_c(E) (Schwarzschild test particle, argued). infall = u_i r^i < 0.',
          f'{"run":4s} {"t":>6s} | {"bound&cap4":>13s} {"bound&~cap4":>13s} {"unb&cap4":>13s} {"unb&~cap4":>13s} | {"capE":>13s} | {"infall":>13s} | hot E/dE p10/50/90 (dE=R*/rt^2)']
    dE = 0.677 / 16.4**2
    for k in order:
        d, z = P[k], R[k]
        Mh = d['dM'][d['hot']].sum()
        def c(s):
            return f'{fr(d["dM"][s & d["hot"]].sum(), Mh):5.3f}({fr(d["dM"][s & d["hot"]].sum(), d["dM"][s].sum()):5.3f})'
        b, c4 = d['bound'], d['cap4']
        L.append(f'{k[0]:4s} {float(z["t"]):6.1f} | {c(b & c4):>13s} {c(b & ~c4):>13s} {c(~b & c4):>13s} {c(~b & ~c4):>13s} | '
                 f'{c(d["capE"]):>13s} | {c(d["vr"] < 0):>13s} | {pct(d["E"][d["hot"]] / dE, d["dM"][d["hot"]])}')
    L.append('')
    for name, key, lab in (('(a4) distance to the stream-atmosphere boundary d_atm (cells to the nearest D <= 1e-9 rho_c cell)', 'd_atm', DLAB),
                           ('(a5) distance to an AMR level boundary d_lev (cells, own dx)', 'd_lev', DLAB),
                           ('(a6) FOFC proxy 1: distance to a density-floored cell (rho < 1.02 dfloor; always fails the probe)', 'd_flr', DLAB),
                           ('(a7) FOFC proxy 2: distance to any temperature-floored cell (P/rho < 1e-14; fails unless rescued)', 'd_tfl', DLAB),
                           ('(a8) distance to a temperature-floored cell INSIDE the band (floored stream gas)', 'd_tflb', DLAB)):
        L += [f'== {name}: share of hot mass per bin / hot fraction of the band in the bin',
              f'{"run":4s} {"t":>6s} | ' + ' '.join(f'{x:>12s}' for x in lab)]
        for k in order:
            d, z = P[k], R[k]
            Mh = d['dM'][d['hot']].sum()
            ib = np.digitize(d[key], DEDG) - 1
            ib = np.where(np.isinf(d[key]), len(lab) - 1, ib)
            cells = [f'{fr(d["dM"][(ib == b) & d["hot"]].sum(), Mh):5.3f}/{fr(d["dM"][(ib == b) & d["hot"]].sum(), d["dM"][ib == b].sum()):5.3f}'
                     for b in range(len(lab))]
            L.append(f'{k[0]:4s} {float(z["t"]):6.1f} | ' + ' '.join(f'{c:>12s}' for c in cells))
        L.append('')
    L += ['== (a9) |grad ln rho| dx: hot vs not-hot band cells, mass-weighted p10/50/90; and shares with |grad ln rho| dx > 1',
          f'{"run":4s} {"t":>6s} | {"hot":>26s} | {"not hot":>26s} | hot>1  nothot>1']
    for k in order:
        d, z = P[k], R[k]
        h = d['hot']
        L.append(f'{k[0]:4s} {float(z["t"]):6.1f} | {pct(d["glnr"][h], d["dM"][h]):>26s} | {pct(d["glnr"][~h], d["dM"][~h]):>26s} | '
                 f'{fr(d["dM"][h & (d["glnr"] > 1)].sum(), d["dM"][h].sum()):5.3f}  {fr(d["dM"][~h & (d["glnr"] > 1)].sum(), d["dM"][~h].sum()):5.3f}')
    L.append('')
    # ---------------- zone on/off
    L += ['== (a10) zone on (T) vs off (B), same t: band and hot mass by r_BH - r_s (r_s = atmosphere_zone_radius of the T arm:',
          '   5 for R6 S2T, 6 for R7 RC_R7T; read by primitive_solver_hyd.hpp:1337, centre padm->bc_center = x_BH, cfc.cpp:461/2682).',
          '   M in 1e-9 M_BH units; "hot/band" per bin.']
    ZB = [-np.inf, -1, 0, 1, 3, 6, np.inf]
    ZL = ['<rs-1', 'rs-1..rs', 'rs..rs+1', 'rs+1..+3', 'rs+3..+6', '>rs+6']
    pairs = [(('R6T', 'S2T_R6', '00009'), ('R6B', 'S2B_R6', '00010'), ('R6A', 'S2A_R6', '00010'), 5.0),
             (('R6T', 'RC_R6T', '00009'), ('R6B', 'RC_R6B', '00010'), None, 5.0),
             (('R7T', 'RC_R7T', '00015'), ('R7B', 'RC_R7B', '00016'), None, 6.0),
             (('R7T', 'RC_R7T', '00014'), ('R7B', 'RC_R7B', '00015'), None, 6.0)]
    L.append(f'{"arm":28s} {"t":>6s} | {"M_band":>8s} {"M_hot":>8s} | ' + ' '.join(f'{x:>17s}' for x in ZL))
    for ks in pairs:
        rs = ks[3]
        for kk in ks[:3]:
            if kk is None or kk not in P:
                continue
            d, z = P[kk], R[kk]
            ib = np.digitize(d['rbh'] - rs, ZB) - 1
            cells = [f'{d["dM"][(ib == b) & d["hot"]].sum() * 1e9:7.3f}/{d["dM"][ib == b].sum() * 1e9:7.3f}' for b in range(len(ZL))]
            L.append(f'{kk[1] + "/" + kk[2]:28s} {float(z["t"]):6.1f} | {d["dM"].sum() * 1e9:8.3f} {d["dM"][d["hot"]].sum() * 1e9:8.3f} | '
                     + ' '.join(f'{c:>17s}' for c in cells))
        L.append('')
    # slices: time-pooled hot profile near r_s
    L += ['   z = 0 slices, all matched dumps pooled (area proxy): hot / band per 0.5 M r_BH bin, T vs B']
    fe = np.arange(3, 20.01, 0.5)
    for tl, bl, rs in (('S2T', 'S2B', 5.0), ('R7T', 'R7B', 6.0)):
        sT = {r['num']: r for r in S if r['lab'] == tl}
        sB = {r['num']: r for r in S if r['lab'] == bl}
        com = sorted(set(sT) & set(sB))
        hT = np.sum([sT[n]['rf_hot'] for n in com], 0); bT = np.sum([sT[n]['rf_band'] for n in com], 0)
        hB = np.sum([sB[n]['rf_hot'] for n in com], 0); bB = np.sum([sB[n]['rf_band'] for n in com], 0)
        L.append(f'   {tl} vs {bl}: {len(com)} dumps t {sT[com[0]]["t"]:.0f}-{sT[com[-1]]["t"]:.0f}, r_s = {rs}; total hot T/B = '
                 f'{hT.sum() / max(hB.sum(), 1e-300):.3f}, band T/B = {bT.sum() / max(bB.sum(), 1e-300):.3f}')
        L.append('     r_BH  ' + ' '.join(f'{x:6.1f}' for x in fe[:-1]))
        L.append('     hot T ' + ' '.join(f'{x / max(hT.sum(), 1e-300):6.3f}' for x in hT))
        L.append('     hot B ' + ' '.join(f'{x / max(hB.sum(), 1e-300):6.3f}' for x in hB))
        L.append('     h/b T ' + ' '.join(f'{fr(x, y):6.2f}' for x, y in zip(hT, bT)))
        L.append('     h/b B ' + ' '.join(f'{fr(x, y):6.2f}' for x, y in zip(hB, bB)))
    L.append('')

    # ---------------- (b) L4 vs L5
    L += ['== (b) matched L4 / L5, same t (3D). Flagged = the replica flag (c086_rst.flags) on band cells.',
          '   psi_v = max_a |v^a(+) - 2v^a + v^a(-)| (raw, not / c_s); dv2 = max_a |v^a(+) - v^a(-)|; C = -div v dx.',
          '   Percentiles mass-weighted p10/50/90 over FLAGGED band cells. dx_fin = finest level dx of the band cells.',
          f'{"pair":34s} {"t":>6s} {"M_band":>9s} {"M_flag":>9s} {"M_hot":>9s} {"M_hot&fl":>9s} | {"psi_v flagged":>26s} | {"dv2 flagged":>26s} | {"C flagged":>26s} | {"psi_v hot":>26s}']
    bp = [(('R6B', 'RC_R6B', '00010'), ('L5B', 'RC_L5B', '00011')), (('R6', 'R6_rp10_comov_L4_fix', '00013'), ('L5', 'L5c_rp10_comov_t200', '00013')),
          (('R6B', 'RC_R6B', '00008'), ('L5B', 'RC_L5B', '00008')), (('R6', 'R6_rp10_comov_L4_fix', '00010'), ('L5', 'L5c_rp10_comov_t200', '00010')),
          (('R6', 'R6_rp10_comov_L4_fix', '00008'), ('L5', 'L5c_rp10_comov_t200', '00008'))]
    for pr in bp:
        for kk in pr:
            if kk not in P:
                continue
            d, z = P[kk], R[kk]
            f = d['flag']
            L.append(f'{kk[1] + "/" + kk[2]:34s} {float(z["t"]):6.1f} {d["dM"].sum() / MSTAR:9.2e} {d["dM"][f].sum() / MSTAR:9.2e} '
                     f'{d["dM"][d["hot"]].sum() / MSTAR:9.2e} {d["dM"][d["hot"] & f].sum() / MSTAR:9.2e} | {pct(d["psiv"][f], d["dM"][f]):>26s} | '
                     f'{pct(d["dv2"][f], d["dM"][f]):>26s} | {pct(d["C"][f], d["dM"][f]):>26s} | {pct(d["psiv"][d["hot"]], d["dM"][d["hot"]]):>26s}')
        # by level for this pair
        for kk in pr:
            if kk not in P:
                continue
            d = P[kk]
            rows = []
            for lv in np.unique(d['lev']):
                s = d['lev'] == lv
                rows.append(f'L{lv}: band {d["dM"][s].sum() / MSTAR:.2e} flag {fr(d["dM"][s & d["flag"]].sum(), d["dM"][s].sum()):.3f} '
                            f'hot {fr(d["dM"][s & d["hot"]].sum(), d["dM"][s].sum()):.3f} psi_v_fl50 {pct(d["psiv"][s & d["flag"]], d["dM"][s & d["flag"]], (50,))}')
            L.append(f'     {kk[0]:4s} by level: ' + '; '.join(rows))
        L.append('')
    L += ['   z = 0 slices at EXACTLY matched dumps: stored in-code flag (de_shock) on stored below-switch cells; shares of the',
          '   band area-proxy, and the in-code psi_v = de_sensor x 0.05 x c_s (mass-weighted median over flagged cells).',
          f'   {"pair":10s} {"t":>6s} | {"L4 f_flag":>9s} {"L5 f_flag":>9s} | {"L4 f_hot":>8s} {"L5 f_hot":>8s} | {"L4 psi_v50":>10s} {"L5 psi_v50":>10s} | M_flag L5/L4  M_hot L5/L4  M_band L5/L4']
    for l4, l5 in (('R6', 'L5'), ('R6B', 'L5B')):
        s4 = {r['num']: r for r in S if r['lab'] == l4}
        s5 = {r['num']: r for r in S if r['lab'] == l5}
        com = sorted(set(s4) & set(s5))
        agg = np.zeros(6)
        for n in com:
            a4, a5 = s4[n], s5[n]
            if a4['M_band'] <= 0 or a5['M_band'] <= 0:
                continue
            p4 = pct(*a4['psiv_fl'], (50,)) if a4['psiv_fl'][0].size else 'n/a'
            p5 = pct(*a5['psiv_fl'], (50,)) if a5['psiv_fl'][0].size else 'n/a'
            L.append(f'   {l4 + "/" + l5:10s} {a4["t"]:6.1f} | {a4["M_flag"] / a4["M_band"]:9.3f} {a5["M_flag"] / a5["M_band"]:9.3f} | '
                     f'{a4["M_hot"] / a4["M_band"]:8.3f} {a5["M_hot"] / a5["M_band"]:8.3f} | {p4:>10s} {p5:>10s} | '
                     f'{fr(a5["M_flag"], a4["M_flag"]):11.3f}  {fr(a5["M_hot"], a4["M_hot"]):10.3f}  {fr(a5["M_band"], a4["M_band"]):11.3f}')
            agg += [a4['M_flag'], a5['M_flag'], a4['M_hot'], a5['M_hot'], a4['M_band'], a5['M_band']]
        L.append(f'   {l4 + "/" + l5:10s} pooled over {len(com)} dumps: M_flag L5/L4 {fr(agg[1], agg[0]):.3f}, M_hot L5/L4 {fr(agg[3], agg[2]):.3f}, '
                 f'M_band L5/L4 {fr(agg[5], agg[4]):.3f}; f_flag L4 {fr(agg[0], agg[4]):.3f} L5 {fr(agg[1], agg[5]):.3f}')
    L.append('')

    # ---------------- (c) floor mass
    L += ['== (c) distinct mass AT the floor temperature (P = rho tfloor, tfloor = 1e-22), band cells.',
          '   3D: P/rho < 1e-14 (the restart stores only conserved variables; a T-floored cell re-inverts to P/rho ~ 1e-26..1e-16,',
          '   round-off of tau; physical band gas on the K0 adiabat has P/rho >= ~5e-11). M in M*, f = share of the band mass.',
          f'{"run":10s} {"t":>6s} {"M_band":>9s} {"M_floor":>9s} {"f_floor":>7s} {"M_floor bound":>13s} {"M_floor<1e-10..":>15s}']
    for lab, rd in (('X11', 'RC_R6X11'), ('N1', 'RC_R6N1'), ('N2c', 'RC_R6N2c')):
        for k in sorted([k for k in P if k[1] == rd], key=lambda k: float(R[k]['t'])):
            d, z = P[k], R[k]
            M = d['dM'].sum(); mf = d['dM'][d['flr']].sum()
            L.append(f'{lab:10s} {float(z["t"]):6.1f} {M / MSTAR:9.2e} {mf / MSTAR:9.2e} {fr(mf, M):7.4f} {d["dM"][d["flr"] & d["bound"]].sum() / MSTAR:13.2e} '
                     f'{d["dM"][(d["PoR"] < 1e-10) & ~d["flr"]].sum() / MSTAR:15.2e}')
    L += ['', '   z = 0 slices, every dump: band area-proxy share with the STORED temperature == tfloor (1e-9 rel.), with T < 1e-14,',
          '   and (old path) T-floored cells whose velocity equals the frame velocity (|u~ - W xidot| <= 1e-6 |u~|).',
          f'   {"run":5s} {"t":>6s} {"N_band":>7s} {"N_tfl":>6s} {"f_tfl":>7s} {"f_T<1e-14":>9s} {"f_fvreset":>9s} {"f_hot":>6s}']
    for lab in ('X11', 'N1', 'N2c'):
        rs = sorted([r for r in S if r['lab'] == lab], key=lambda r: r['t'])
        for r in rs[::4]:
            L.append(f'   {lab:5s} {r["t"]:6.1f} {r["N_band"]:7d} {r["N_tfl"]:6d} {fr(r["M_tfl"], r["M_band"]):7.4f} {fr(r["M_tlo"], r["M_band"]):9.4f} '
                     f'{fr(r["M_fvreset"], r["M_band"]):9.4f} {fr(r["M_hot"], r["M_band"]):6.3f}')
        if rs:
            tf = np.array([fr(r['M_tfl'], r['M_band']) for r in rs]); tl = np.array([fr(r['M_tlo'], r['M_band']) for r in rs])
            hh = np.array([fr(r['M_hot'], r['M_band']) for r in rs])
            L.append(f'   {lab:5s} all {len(rs)} dumps: f_tfl median {np.nanmedian(tf):.4f} (max {np.nanmax(tf):.4f}); dumps with any T-floored band cell '
                     f'{sum(r["N_tfl"] > 0 for r in rs)}; f_T<1e-14 == f_tfl in {np.sum(np.isclose(tf, tl))}; f_hot median {np.nanmedian(hh):.3f}')
    L.append('')
    # X11 (cold recovery) vs the old-path runs at the same window
    L += ['== (a11) cold recovery vs the old floor path, 3D band: hot fraction, floor share (P/rho < 1e-14), cold share',
          '   (K/K0 < 10, not floored), flagged share. X11 = RC_R6X11 (cold_recovery + fofc_cold_recovery + fofc_scalar_consistent);',
          '   N1, N2c = old path, same deck otherwise (coord div); R6B/R6T/R6A = S2/RC arms (old path; covariant / zone).',
          f'   {"run":5s} {"dir":10s} {"t":>6s} {"M_band":>9s} {"M_hot":>9s} {"f_hot":>6s} {"f_floor":>7s} {"f_cold":>6s} {"f_flag":>6s}']
    for k in sorted([k for k in P if k[0] in ('X11', 'N1', 'N2c', 'R6B', 'R6T', 'R6A')], key=lambda k: (float(R[k]['t']), k[0])):
        d = P[k]; M = d['dM'].sum()
        L.append(f'   {k[0]:5s} {k[1]:10s} {float(R[k]["t"]):6.1f} {M / MSTAR:9.2e} {d["dM"][d["hot"]].sum() / MSTAR:9.2e} {fr(d["dM"][d["hot"]].sum(), M):6.3f} '
                 f'{fr(d["dM"][d["flr"]].sum(), M):7.4f} {fr(d["dM"][(d["KK0"] < 10) & ~d["flr"]].sum(), M):6.3f} {fr(d["dM"][d["flag"]].sum(), M):6.3f}')
    L.append('')
    L += ['== (a12) temperature profile across the band (3D): mass-weighted median P/rho and K/K0 and hot fraction per D/rho_c decade']
    E = RHOC * np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    for k in order:
        d = P[k]
        cells = []
        for lo_, hi_ in zip(E[:-1], E[1:]):
            s = (d['D'] > lo_) & (d['D'] <= hi_)
            if s.sum() < 10:
                cells.append(f'{"-":>22s}'); continue
            cells.append(f'{pct(d["PoR"][s], d["dM"][s], (50,)):>8s} {pct(d["KK0"][s], d["dM"][s], (50,)):>8s} {fr(d["dM"][s & d["hot"]].sum(), d["dM"][s].sum()):4.2f}')
        L.append(f'   {k[0]:4s} {float(R[k]["t"]):6.1f} | ' + ' | '.join(cells))
    L.append('   (columns: D/rho_c 1e-9..1e-8 | 1e-8..1e-7 | 1e-7..1e-6 | 1e-6..1e-5 | 1e-5..switch; each P/rho, K/K0, f_hot)')
    L.append('')
    # the gas around the band (z = 0 slices, stored primitives)
    L += ['== (a13) P/rho of the gas around the band (z = 0 slice, stored primitives; mass-weighted p10/p50/p90 per rho range)']
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import c086_slices as S86
        for d_, n_ in (('R6_rp10_comov_L4_fix', '00140'), ('R6_rp10_comov_L4_fix', '00170'), ('R6x_rp10_comov_L4_t300', '00300'),
                       ('R7x_rp18_comov_L4_t500', '00500'), ('RC_R6X11', '00150'), ('RC_R6N1', '00150')):
            q = S86.bc.read_binary(f'{S86.C}/{d_}/bin/{S86.B}.prim_xy.{n_}.bin')
            g = np.asarray(q['mb_geometry']); nx = q['nx1_out_mb']; dx = (g[:, 1] - g[:, 0]) / nx
            rho = np.asarray(q['mb_data']['dens'])[:, 0]; pr = np.asarray(q['mb_data']['press'])[:, 0]
            w = rho * (dx**2)[:, None, None]
            cells = []
            for lo_, hi_ in ((1.03e-24, 1e-17), (1e-17, 3.24e-13), (3.24e-13, 3.2e-8), (3.2e-8, 1.0)):
                s = (rho > lo_) & (rho <= hi_)
                cells.append(f'{lo_:.0e}-{hi_:.0e}: {pct(pr[s] / rho[s], w[s])}' if s.any() else f'{lo_:.0e}-{hi_:.0e}: n/a')
            L.append(f'   {d_:24s} t {q["time"]:6.1f} | ' + ' | '.join(cells))
    except Exception as e:                                   # noqa: BLE001
        L.append(f'   (failed: {e})')
    L.append('')
    # cumulative vs distinct erased mass (X11 CRECOV lines)
    import re
    L += ['== (c2) X11 CRECOV census (per cycle = the C2P window from stage 1 to stage 3, primitive_solver_hyd.hpp:661, 761):',
          '   m_erased = recovered mass whose previous P/rho > 1e-10; recovered = m_bound + m_unbound. M*.']
    rows, tc = [], {}
    for f in sorted(glob.glob('/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/RC_R6X11/seg*.out')):
        for ln in open(f):
            m = re.match(r'elapsed=\S+ cycle=(\d+) time=(\S+)', ln)
            if m:
                tc[int(m.group(1))] = float(m.group(2)); continue
            if ln.startswith('CRECOV') and 'path=steps' in ln:
                kv = dict(re.findall(r'(\w+)=(\S+)', ln))
                rows.append((int(kv['cycle']), float(kv['m_erased']), float(kv['m_bound']) + float(kv['m_unbound'])))
    if rows:
        c = np.array(rows); _, i = np.unique(c[:, 0], return_index=True); c = c[i]
        tt = np.array([tc.get(int(x), np.nan) for x in c[:, 0]])
        L.append(f'   {len(c)} cycles, t {np.nanmin(tt):.1f}-{np.nanmax(tt):.1f}: cumulative m_erased {c[:, 1].sum() / MSTAR:.4f} M*, '
                 f'cumulative recovered {c[:, 2].sum() / MSTAR:.4f} M*')
        for k in sorted([k for k in P if k[0] == 'X11'], key=lambda k: float(R[k]['t'])):
            t0 = float(R[k]['t']); s = np.abs(tt - t0) < 1.0
            d = P[k]; mf = d['dM'][d['flr']].sum()
            L.append(f'   t {t0:6.1f}: distinct floor mass (3D band) {mf / MSTAR:.2e}; per-cycle m_erased median {np.median(c[s, 1]) / MSTAR:.2e}, '
                     f'per-cycle recovered median {np.median(c[s, 2]) / MSTAR:.2e}; cumulative m_erased to t / distinct {c[tt <= t0, 1].sum() / max(mf, 1e-300):.0f}')
    L.append('')
    # whole-mesh census (FOFC reading)
    cen = sorted(glob.glob(f'{a.cache}/../cen/*.npz'))
    if cen:
        L += ['== FOFC reading: whole-mesh interior census (3D): density-floored (rho < 1.02 dfloor), T-floored (P/rho < 1e-14,',
              '   not density-floored), other; counts and rest mass (M_BH units).']
        for f in cen:
            z = np.load(f); c = z['census']; n = c[:3].sum()
            L.append(f'   {str(z["rdir"]):24s} t {float(z["t"]):6.1f}: N {int(n)}; rho-floored {int(c[0])} ({c[0] / n:.3f}, M {c[3]:.1e}); '
                     f'T-floored {int(c[1])} ({c[1] / n:.4f}, M {c[4]:.1e}); other {int(c[2])} ({c[2] / n:.3f})')
        # the in-code FOFC tally on the first cycle after the X11 t = 158.5 restart (seg_2.out)
        import re as _re
        fs = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/RC_R6X11/seg_2.out'
        if os.path.exists(fs):
            for ln in open(fs):
                if ln.startswith('CRECOV') and 'path=steps' in ln:
                    kv = dict(_re.findall(r'(\w+)=(\S+)', ln))
                    L.append(f'   in-code tally, RC_R6X11 seg_2 cycle {kv["cycle"]} (the t = 158.5 restart): fofc_hydro {float(kv["fofc_hydro"]):.4e} per '
                             f'CRECOV window = stages 2 + 3 (reset at stage-1 C2P, printed at stage-3 C2P) -> {float(kv["fofc_hydro"]) / 2:.4e} per stage; '
                             f'fofc_rescued {float(kv["fofc_rescued"]):.3e}, fofc_exc {float(kv["fofc_exc"]):.3e}; '
                             f'(fofc_hydro + fofc_exc)/2 = {(float(kv["fofc_hydro"]) + float(kv["fofc_exc"])) / 2:.4e} per stage vs the '
                             f'density-floored census at that restart (line above, RC_R6X11 t 158.5)')
                    break
    txt = '\n'.join(L) + '\n'
    print(txt)
    with open(f'{a.out}/c089_hot.txt', 'w') as fo:
        fo.write(txt)


if __name__ == '__main__':
    main()
