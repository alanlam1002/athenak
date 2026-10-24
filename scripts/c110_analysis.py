#!/usr/bin/env python3
"""2x2 general analysis (c110): time series of the rp10 arms (CN comoving, BN BH frame, ...), with ref7 R6_u
(rp10 comoving, dual energy on, t 113-170) as a reference. One 2 x 4 figure plus a CSV of the series:
  (a) rho_max / rho_c (user hst)                     (b) star-BH separation: the z = 0 density maximum (prim_xy)
                                                         minus x_BH(t), against the test-particle geodesic
  (c) bound mass / M* (CRECOV Mbound_all) and the    (d) entropy-tracer total scal-0 / its t0 value - 1 (mhd hst)
      mass on the grid / M* (mhd hst)
  (e) K/K0 at the density maximum (prim_xy;          (f) total energy tot-E relative to its t0 value (mhd hst)
      K0 = its t = 0 value)
  (g) alpha_min (user hst)                           (h) dt (out.log) and the cumulative regrid count; the wall
                                                         time per M of evolution (out.log elapsed)
    c110_analysis.py OUTDIR LABEL=RUNDIR [LABEL=RUNDIR ...] [--ref LABEL=RUNDIR]
The decks are read from RUNDIR/parfile.par (the BH position and the geodesic's release state).
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tde_ensemble_plots import bc                 # noqa: E402  (stubs h5py, as anim_global_local does)

B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC, GAM, MSTAR = 3.24e-4, 5.0 / 3.0, 7.47e-5
COL = {'CN': 'C0', 'BN': 'C3', 'CD': 'C2', 'BD': 'C1'}


def par_get(d, k, dflt=0.0):
    m = re.search(rf'^{k}\s*=\s*([-+.\deE]+)', open(f'{d}/parfile.par').read(), re.M)
    return float(m.group(1)) if m else dflt


def hst(d, f):
    fn = glob.glob(f'{d}/{B}.{f}.hst')[0]
    hdr = [l for l in open(fn) if l.startswith('#')][-1]
    names = [x.split('=')[-1] for x in hdr.replace('#', '').split()]
    a = np.loadtxt(fn)
    _, i = np.unique(a[:, 0], return_index=True)
    return names, a[np.sort(i)]


def col(d, f, q):
    n, a = hst(d, f)
    return a[:, 0], a[:, n.index(q)]


def bh_xy(d, t):
    """x_BH(t) = puncture_x - xi(t); xi = the frame_disp columns (zero in the BH frame)"""
    p0 = np.array([par_get(d, 'puncture_x1'), par_get(d, 'puncture_x2')])
    n, a = hst(d, 'user')
    if 'frame_disp_x1' in n:
        xi = np.stack([np.interp(t, a[:, 0], a[:, n.index(f'frame_disp_x{k}')]) for k in (1, 2)], -1)
    else:
        xi = np.zeros((len(np.atleast_1d(t)), 2))
    return p0[None, :] - xi


def dump(fn):
    """time, the z = 0 density maximum's (x, y) and K there (NaN for a file still being written)"""
    try:
        p = bc.read_binary(fn)
    except Exception:
        return (np.nan,) * 4
    g = np.asarray(p['mb_geometry'])
    nx, ny = p['nx1_out_mb'], p['nx2_out_mb']
    rho = np.asarray(p['mb_data']['dens'])[:, 0]
    pr = np.asarray(p['mb_data']['press'])[:, 0]
    m, j, i = np.unravel_index(np.argmax(rho), rho.shape)
    x = g[m][0] + (i + 0.5) * (g[m][1] - g[m][0]) / nx
    y = g[m][2] + (j + 0.5) * (g[m][3] - g[m][2]) / ny
    return p['time'], x, y, pr[m, j, i] / rho[m, j, i]**GAM


def log_series(d):
    """(t, dt, elapsed) per cycle line; regrid times; (t, Mbound_all) from CRECOV"""
    cyc_t, T, DT, W, rg, mb = {}, [], [], [], [], []
    w0 = 0.0
    for l in open(f'{d}/out.log', errors='replace'):
        g = re.match(r'elapsed=(\S+) cycle=(\d+) time=(\S+) dt=(\S+)', l)
        if g:
            w = float(g.group(1))
            if W and w < W[-1] - w0:          # a new segment restarts the elapsed clock
                w0 = W[-1]
            c, t = int(g.group(2)), float(g.group(3))
            cyc_t[c] = t
            T.append(t); DT.append(float(g.group(4))); W.append(w + w0)
            continue
        g = re.match(r'MLEDGER0 cycle=(\d+) .*path=regrid', l)
        if g and int(g.group(1)) in cyc_t:
            rg.append(cyc_t[int(g.group(1))])
            continue
        g = re.match(r'CRECOV cycle=(\d+) .*Mbound_all=(\S+)', l)
        if g and int(g.group(1)) in cyc_t:
            mb.append((cyc_t[int(g.group(1))], float(g.group(2))))
    return np.array(T), np.array(DT), np.array(W), np.array(rg), np.array(mb).reshape(-1, 2)


def geodesic(d, tmax):
    import tde_parabolic_orbit_ic as P
    x0 = par_get(d, 'star_center_x1') - par_get(d, 'puncture_x1')
    s = P.solve_ic(x0, 10.0)
    o = P.integrate(x0, s['px'], s['py'], tmax, 0.05)
    return o[:, 0], np.hypot(o[:, 1], o[:, 2])


def series(lab, d, nproc):
    fs = sorted(glob.glob(f'{d}/bin/{B}.prim_xy.*.bin'))
    with Pool(nproc) as pool:
        S = np.array(pool.map(dump, fs))
    S = S[np.isfinite(S[:, 0])]
    S = S[np.argsort(S[:, 0])]
    bh = bh_xy(d, S[:, 0])
    sep = np.hypot(S[:, 1] - bh[:, 0], S[:, 2] - bh[:, 1])
    out = {'t_dump': S[:, 0], 'sep': sep, 'Kc': S[:, 3] / S[0, 3] if S[0, 0] == 0 else S[:, 3] / np.nan}
    out['t_u'], out['rhomax'] = col(d, 'user', 'rho-max')
    _, out['amin'] = col(d, 'user', 'alpha-min')
    out['t_m'], out['mass'] = col(d, 'mhd', 'mass')
    _, out['scal0'] = col(d, 'mhd', 'scal-0')
    _, out['totE'] = col(d, 'mhd', 'tot-E')
    out['T'], out['DT'], out['W'], out['rg'], out['mb'] = log_series(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--ref', action='append', default=[])
    ap.add_argument('--nproc', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {}
    for spec in a.runs + a.ref:
        lab, d = spec.split('=', 1)
        R[lab] = (d, series(lab, d, a.nproc), spec in a.ref)
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.6))
    ax = ax.ravel()
    tmax = max(np.nanmax(s['t_u']) for _, s, _ in R.values())
    for lab, (d, s, ref) in R.items():
        kw = dict(color=COL.get(lab, '0.5'), lw=1.6 if not ref else 1.0, ls='-' if not ref else '--', label=lab)
        ax[0].plot(s['t_u'], s['rhomax'] / RHOC, **kw)
        ax[1].plot(s['t_dump'], s['sep'], **kw)
        if len(s['mb']):
            ax[2].plot(s['mb'][:, 0], s['mb'][:, 1] / MSTAR, **kw)
        ax[2].plot(s['t_m'], s['mass'] / MSTAR, **{**kw, 'lw': 0.8, 'label': None})
        if not ref:
            ax[3].plot(s['t_m'], s['scal0'] / s['scal0'][0] - 1, **kw)
            ax[4].plot(s['t_dump'], s['Kc'], **kw)
            ax[5].plot(s['t_m'], s['totE'] / s['totE'][0] - 1, **kw)
        ax[6].plot(s['t_u'], s['amin'], **kw)
        ax[7].plot(s['T'], s['DT'], **kw)
        if not ref and len(s['W']) > 1:
            tw = np.arange(np.ceil(s['T'][0]), s['T'][-1], 5.0)
            wt = np.interp(tw, s['T'], s['W'])
            ax7b = ax[7].twinx() if lab == next(k for k, v in R.items() if not v[2]) else ax7b
            ax7b.step(tw[1:], np.diff(wt) / 5.0, where='pre', color=kw['color'], lw=0.8, ls=':')
            ax7b.set_ylabel('wall s per M (dotted)')
    if a.runs:
        d0 = R[a.runs[0].split('=')[0]][0]
        tg, rg = geodesic(d0, tmax)
        ax[1].plot(tg, rg, 'k:', lw=1, label='geodesic (r_p = 10)')
    labels = [(r'$\rho_{max}/\rho_c$', 'log'), ('star–BH separation [M]', 'linear'),
              (r'$M_{bound}/M_*$ (thick); grid mass $/M_*$ (thin)', 'linear'),
              ('scal-0 / scal-0(0) − 1 (entropy tracer)', 'linear'), (r'$K/K_0$ at $\rho_{max}$ (z = 0)', 'linear'),
              ('tot-E / tot-E(0) − 1', 'linear'), (r'$\alpha_{min}$', 'linear'), ('dt [M] (solid)', 'linear')]
    for k, (yl, sc) in enumerate(labels):
        ax[k].set_ylabel(yl); ax[k].set_yscale(sc); ax[k].set_xlabel('t [M]'); ax[k].grid(alpha=0.3)
        ax[k].axvline(135.9, color='0.6', lw=0.6, ls='--')
    for lab, (d, s, ref) in R.items():
        if not ref and len(s['rg']):
            ax[7].plot(s['rg'], np.full(len(s['rg']), ax[7].get_ylim()[0]), '|', color=COL.get(lab, '0.5'), ms=6)
    ax[0].legend(fontsize=9); ax[1].legend(fontsize=8)
    fig.suptitle('2x2 rp10 (L4, base8, GPU IPC off): ' + ', '.join(
        f'{k} to t = {np.nanmax(v[1]["t_u"]):.0f}' for k, v in R.items() if not v[2]) +
        '; dashed = reference; grey line = design periapsis t = 135.9; ticks on (h) = regrids')
    fig.tight_layout()
    png = f'{a.out}/c110_rp10_series.png'
    fig.savefig(png, dpi=80)
    rows = ['label,t,rhomax_over_rhoc,sep,K_over_K0']
    for lab, (d, s, ref) in R.items():
        for t, sp, kc in zip(s['t_dump'], s['sep'], s['Kc']):
            rows.append(f'{lab},{t:.3f},{np.interp(t, s["t_u"], s["rhomax"]) / RHOC:.6g},{sp:.4f},{kc:.5g}')
    open(f'{a.out}/c110_rp10_series.csv', 'w').write('\n'.join(rows) + '\n')
    print(png, os.path.getsize(png))


if __name__ == '__main__':
    main()
