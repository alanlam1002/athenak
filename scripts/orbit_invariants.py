#!/usr/bin/env python3
"""Orbital invariants E(t), L(t) and centre-of-mass areal radius of the star, from a
CFC TDE run's z=0 bin slices (prim_xy + adm_xy).

For a body on a geodesic of the stationary, axisymmetric trumpet background, the
Killing energy and angular momentum

    E = -u_t = alpha W - beta^i p_i ,   L = x p_y - y p_x ,   p_i = psi^4 u^i ,
    W = sqrt(1 + psi^4 |u|^2)                 (velx/y/z in the dumps are u^i = W v^i)

are exactly conserved. Drift in either one before disruption is therefore numerical.
Values are density-weighted over the stellar core (rho > --core * rho_max). The CoM
uses a looser cut (--com). Both are z=0-slice quantities (the orbit is planar).

Measured on the elliptic production run: E, L equal to the design values to 1e-4 at
t=0, then L decays ~0.015-0.02%/M (4.9% by t=250) -- the reason its achieved periapsis was
R_areal=6.8, not the designed 10.0. Use this to check any new orbit before trusting it.

Comoving gauge (<cfc> gauge_xdot*/gauge_accel*, src/cfc/DEVELOPMENT.md item 68): the
dumps are in grid coordinates x' = x - X(t) and the dumped shift is beta' = beta + Xdot.
If BASENAME.user.hst carries gauge-Xdot*/gauge-X* columns, both are mapped back to the
BH frame before E and L are formed: x = x' + X(t), beta = beta' - Xdot, i.e.
u_t(BH) = u_t' - u_i Xdot^i. Without those columns nothing changes.

Needs numpy+scipy (e.g. /opt/aurora/default/frameworks/*/bin/python3). h5py is stubbed:
bin_convert only needs it for the athdf writer.

    orbit_invariants.py RUN_OUTPUT_DIR BASENAME [--every 5] [--tmax 60]
"""
import argparse
import glob
import math
import os
import sys
import types

import numpy as np

sys.modules.setdefault('h5py', types.ModuleType('h5py'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vis', 'python'))
sys.path.insert(0, HERE)
import bin_convert as bc                 # noqa: E402
import tde_elliptical_orbit_ic as ic     # noqa: E402


def gauge_from_hst(outdir, basename):
    """(t -> Xdot[3], t -> X[3]) from the user history file's gauge columns, or None."""
    f = os.path.join(outdir, f'{basename}.user.hst')
    if not os.path.exists(f):
        return None
    labels = {}
    for line in open(f):
        if line.startswith('#') and '[1]=' in line:
            for tok in line[1:].split():
                if '=' in tok:
                    i, name = tok.split('=', 1)
                    labels[name] = int(i.strip('[]')) - 1
            break
    need = [f'gauge-Xdot{a}' for a in (1, 2, 3)] + [f'gauge-X{a}' for a in (1, 2, 3)]
    if not all(n in labels for n in need):
        return None
    h = np.loadtxt(f, comments='#', ndmin=2)
    t = h[:, labels['time']]
    xd = [h[:, labels[n]] for n in need[:3]]
    xx = [h[:, labels[n]] for n in need[3:]]
    return (lambda tt: np.array([np.interp(tt, t, c) for c in xd]),
            lambda tt: np.array([np.interp(tt, t, c) for c in xx]))


def measure(pf, af, core, comcut, gauge=None):
    p = bc.read_binary(pf)
    a = bc.read_binary(af)
    xdot, xoff = np.zeros(3), np.zeros(3)
    if gauge is not None:
        xdot, xoff = gauge[0](p['time']), gauge[1](p['time'])
    nx, ny = p['nx1_out_mb'], p['nx2_out_mb']
    g = np.asarray(p['mb_geometry'])
    ga = {tuple(l): i for i, l in enumerate(np.asarray(a['mb_logical']).tolist())}
    dens = np.asarray(p['mb_data']['dens'])
    rmax = dens.max()
    acc = dict(w=0.0, E=0.0, L=0.0, cw=0.0, cx=0.0, cy=0.0)
    for m, l in enumerate(np.asarray(p['mb_logical']).tolist()):
        q = ga.get(tuple(l))
        if q is None:
            continue
        rho = dens[m, 0]
        dx = (g[m, 1] - g[m, 0]) / nx
        dy = (g[m, 3] - g[m, 2]) / ny
        X, Y = np.meshgrid(g[m, 0] + (np.arange(nx) + .5) * dx + xoff[0],
                           g[m, 2] + (np.arange(ny) + .5) * dy + xoff[1])
        c = rho > comcut * rmax
        if c.any():
            w = rho[c] * dx * dy
            acc['cw'] += w.sum(); acc['cx'] += (w * X[c]).sum(); acc['cy'] += (w * Y[c]).sum()
        s = rho > core * rmax
        if not s.any():
            continue
        A = lambda v: np.asarray(a['mb_data'][v])[q, 0][s]
        P = lambda v: np.asarray(p['mb_data'][v])[m, 0][s]
        ux, uy, uz = P('velx'), P('vely'), P('velz')
        ps4 = A('adm_psi4')
        px, py = ps4 * ux, ps4 * uy
        W = np.sqrt(1.0 + ps4 * (ux * ux + uy * uy + uz * uz))
        E = A('adm_alpha') * W - ((A('adm_betax') - xdot[0]) * px +
                                   (A('adm_betay') - xdot[1]) * py +
                                   (A('adm_betaz') - xdot[2]) * ps4 * uz)
        L = X[s] * py - Y[s] * px
        w = rho[s] * dx * dy
        acc['w'] += w.sum(); acc['E'] += (w * E).sum(); acc['L'] += (w * L).sum()
    x, y = acc['cx'] / acc['cw'], acc['cy'] / acc['cw']
    r = math.hypot(x, y)
    return p['time'], acc['E'] / acc['w'], acc['L'] / acc['w'], r, ic.trumpet_iso_to_areal(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outdir', help='run output dir containing bin/')
    ap.add_argument('basename')
    ap.add_argument('--every', type=int, default=5, help='use every Nth dump')
    ap.add_argument('--tmax', type=float, default=1e30)
    ap.add_argument('--core', type=float, default=0.5, help='E,L weighting cut / rho_max')
    ap.add_argument('--com', type=float, default=1e-2, help='CoM cut / rho_max')
    a = ap.parse_args()
    pfs = sorted(glob.glob(os.path.join(a.outdir, 'bin', f'{a.basename}.prim_xy.*.bin')))
    gauge = gauge_from_hst(a.outdir, a.basename)
    if gauge is not None:
        print('comoving gauge: mapping positions and shift back to the BH frame')
    rows = []
    for pf in pfs[::a.every]:
        af = pf.replace('.prim_xy.', '.adm_xy.')
        if not os.path.exists(af):
            continue
        r = measure(pf, af, a.core, a.com, gauge)
        if r[0] > a.tmax:
            break
        rows.append(r)
    if not rows:
        sys.exit('no dumps found')
    E0, L0 = rows[0][1], rows[0][2]
    print(f"{'t':>7} {'E':>11} {'L':>10} {'dL/L0 %':>9} {'dE':>10} {'CoM r_iso':>10} {'R_areal':>8}")
    for t, E, L, r, R in rows:
        print(f"{t:7.1f} {E:11.7f} {L:10.5f} {100*(L/L0-1):9.3f} {E-E0:10.2e} {r:10.3f} {R:8.3f}")
    k = min(range(len(rows)), key=lambda i: rows[i][4])
    if len(rows) > 2:
        t = np.array([r[0] for r in rows]); L = np.array([r[2] for r in rows])
        slope = np.polyfit(t, L / L0, 1)[0]
        print(f"\nL drift rate: {100*slope:.4f} %/M   (elliptic production: ~0.019 %/M over t=0-250)")
    print(f"min CoM R_areal so far: {rows[k][4]:.3f} at t={rows[k][0]:.1f}")


if __name__ == '__main__':
    main()
