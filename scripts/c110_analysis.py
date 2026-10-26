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
    c110_analysis.py OUTDIR LABEL=RUNDIR [LABEL=RUNDIR ...] [--ref LABEL=RUNDIR] [--split]
--split also writes each panel as its own figure, c110_<name>_<a-h>_<tag>.png.
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
LABELS = [(r'$\rho_{max}/\rho_c$', 'log'), ('star–BH separation [M]', 'linear'),
          (r'$M_{bound}/M_*$ (thick); grid mass $/M_*$ (thin)', 'linear'),
          ('scal-0 / scal-0(0) − 1 (entropy tracer)', 'linear'), (r'$K/K_0$ at $\rho_{max}$ (z = 0)', 'linear'),
          ('tot-E / tot-E(0) − 1', 'linear'), (r'$\alpha_{min}$', 'linear'), ('dt [M] (solid)', 'linear')]
SPLIT = [('rhomax', 'density maximum'), ('separation', 'star–BH separation vs the geodesic'),
         ('mass', 'bound mass (thick) and grid mass (thin)'), ('tracer', 'entropy-tracer total'),
         ('K_at_rhomax', 'adiabat at the density maximum'), ('totE', 'total energy (Eulerian)'),
         ('alphamin', 'minimum lapse'), ('dt_cost', 'time step, wall time per M (dotted), regrids (ticks)')]


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


def geodesic(d, tmax, rp=10.0):
    import tde_parabolic_orbit_ic as P
    x0 = par_get(d, 'star_center_x1') - par_get(d, 'puncture_x1')
    s = P.solve_ic(x0, rp)
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
    ap.add_argument('--tp', type=float, default=135.9)    # design periapsis (rp18: 132.0)
    ap.add_argument('--rp', type=float, default=10.0)    # design periapsis R_areal of the geodesic
    ap.add_argument('--name', default='rp10')             # output prefix and title (rp18)
    ap.add_argument('--split', action='store_true')       # also one PNG per panel: c110_<name>_<a-h>_<tag>.png
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {}
    for spec in a.runs + a.ref:
        lab, d = spec.split('=', 1)
        R[lab] = (d, series(lab, d, a.nproc), spec in a.ref)
    tmax = max(np.nanmax(s['t_u']) for _, s, _ in R.values())
    geo = None
    if a.runs:
        geo = geodesic(R[a.runs[0].split('=')[0]][0], tmax, a.rp)

    def panel(k, ax, legend):
        """draw panel k ((a)-(h) of the docstring) on ax"""
        first = next((lab for lab, v in R.items() if not v[2]), None)
        axb = None
        for lab, (d, s, ref) in R.items():
            kw = dict(color=COL.get(lab, '0.5'), lw=1.6 if not ref else 1.0, ls='-' if not ref else '--',
                      label=lab)
            if k == 0:
                ax.plot(s['t_u'], s['rhomax'] / RHOC, **kw)
            elif k == 1:
                ax.plot(s['t_dump'], s['sep'], **kw)
            elif k == 2:
                if len(s['mb']):
                    ax.plot(s['mb'][:, 0], s['mb'][:, 1] / MSTAR, **kw)
                ax.plot(s['t_m'], s['mass'] / MSTAR, **{**kw, 'lw': 0.8, 'label': None})
            elif k in (3, 4, 5) and not ref:
                if k == 3:
                    ax.plot(s['t_m'], s['scal0'] / s['scal0'][0] - 1, **kw)
                elif k == 4:
                    ax.plot(s['t_dump'], s['Kc'], **kw)
                else:
                    ax.plot(s['t_m'], s['totE'] / s['totE'][0] - 1, **kw)
            elif k == 6:
                ax.plot(s['t_u'], s['amin'], **kw)
            elif k == 7:
                ax.plot(s['T'], s['DT'], **kw)
                if not ref and len(s['W']) > 1:
                    tw = np.arange(np.ceil(s['T'][0]), s['T'][-1], 5.0)
                    wt = np.interp(tw, s['T'], s['W'])
                    axb = ax.twinx() if lab == first else axb
                    axb.step(tw[1:], np.diff(wt) / 5.0, where='pre', color=kw['color'], lw=0.8, ls=':')
                    axb.set_ylabel('wall s per M (dotted)')
        if k == 1 and geo is not None:
            ax.plot(geo[0], geo[1], 'k:', lw=1, label=f'geodesic (r_p = {a.rp:g})')
        yl, sc = LABELS[k]
        ax.set_ylabel(yl); ax.set_yscale(sc); ax.set_xlabel('t [M]'); ax.grid(alpha=0.3)
        ax.axvline(a.tp, color='0.6', lw=0.6, ls='--')
        if k == 7:
            for lab, (d, s, ref) in R.items():
                if not ref and len(s['rg']):
                    ax.plot(s['rg'], np.full(len(s['rg']), ax.get_ylim()[0]), '|', color=COL.get(lab, '0.5'),
                            ms=6)
        if legend:
            ax.legend(fontsize=9 if k == 0 else 8)

    fig, ax = plt.subplots(2, 4, figsize=(20, 8.6))
    ax = ax.ravel()
    for k in range(8):
        panel(k, ax[k], legend=k in (0, 1))
    fig.suptitle(f'2x2 {a.name} (L4, base8, GPU IPC off): ' + ', '.join(
        f'{k} to t = {np.nanmax(v[1]["t_u"]):.0f}' for k, v in R.items() if not v[2]) +
        f'; dashed = reference; grey line = design periapsis t = {a.tp}; ticks on (h) = regrids')
    fig.tight_layout()
    png = f'{a.out}/c110_{a.name}_series.png'
    fig.savefig(png, dpi=80)
    rows = ['label,t,rhomax_over_rhoc,sep,K_over_K0']
    for lab, (d, s, ref) in R.items():
        for t, sp, kc in zip(s['t_dump'], s['sep'], s['Kc']):
            rows.append(f'{lab},{t:.3f},{np.interp(t, s["t_u"], s["rhomax"]) / RHOC:.6g},{sp:.4f},{kc:.5g}')
    open(f'{a.out}/c110_{a.name}_series.csv', 'w').write('\n'.join(rows) + '\n')
    print(png, os.path.getsize(png))
    if a.split:   # one figure per panel, each with its own legend
        for k, (tag, title) in enumerate(SPLIT):
            f1, a1 = plt.subplots(figsize=(7.5, 5))
            panel(k, a1, legend=True)
            a1.set_title(f'{a.name}: {title}' + (f' (grey: design periapsis t = {a.tp})'), fontsize=10)
            f1.tight_layout()
            fn = f'{a.out}/c110_{a.name}_{chr(97 + k)}_{tag}.png'
            f1.savefig(fn, dpi=100)
            plt.close(f1)
            print(fn, os.path.getsize(fn))


if __name__ == '__main__':
    main()
