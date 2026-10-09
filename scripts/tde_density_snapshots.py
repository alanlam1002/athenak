#!/usr/bin/env python3
"""Side-by-side z=0 density snapshots for a CFC TDE run: a global BH+star panel and a
star-frame panel, both with MeshBlock outlines.

Global panel also shows the analytic test-particle orbit from the run's own
<problem> star_center/star_vel (integrated in the trumpet background with the
tde_elliptical_orbit_ic geodesic RHS), the analytic position at the snapshot time,
the tidal radius r_t and the design periapsis (areal radii, drawn in isotropic
coordinates). The star frame is centred on the density-weighted CoM (rho > --com*rho_max).

Needs numpy+scipy+matplotlib (e.g. /opt/aurora/default/frameworks/*/bin/python3).

    tde_density_snapshots.py RUN_OUTPUT_DIR BASENAME [--every 5] [--outdir DIR]
"""
import argparse
import glob
import math
import os
import sys
import types

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.collections import LineCollection     # noqa: E402
from matplotlib.colors import LogNorm                 # noqa: E402

sys.modules.setdefault('h5py', types.ModuleType('h5py'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vis', 'python'))
sys.path.insert(0, HERE)
import bin_convert as bc                 # noqa: E402
import tde_elliptical_orbit_ic as ic     # noqa: E402


def par(pfile, block, key):
    blk = None
    for line in open(pfile):
        line = line.split('#', 1)[0].strip()
        if line.startswith('<'):
            blk = line
        elif '=' in line and blk == block:
            k, v = line.split('=', 1)
            if k.strip() == key:
                return float(v)
    raise KeyError(f'{block} {key}')


def analytic_orbit(pfile, t_max, dt=0.05):
    x0 = [par(pfile, '<problem>', f'star_center_x{i}') for i in (1, 2, 3)]
    v = [par(pfile, '<problem>', f'star_vel_x{i}') for i in (1, 2, 3)]
    psi0 = ic.metric_at(*x0)[0]
    W = 1.0 / math.sqrt(1.0 - psi0 ** 4 * sum(c * c for c in v))
    y = list(x0) + [psi0 ** 4 * W * c for c in v]          # p_i = psi^4 u^i
    out = [(0.0, y[0], y[1])]
    t = 0.0
    while t < t_max:
        k1 = ic.geodesic_rhs(t, y)
        k2 = ic.geodesic_rhs(t, [y[i] + 0.5 * dt * k1[i] for i in range(6)])
        k3 = ic.geodesic_rhs(t, [y[i] + 0.5 * dt * k2[i] for i in range(6)])
        k4 = ic.geodesic_rhs(t, [y[i] + dt * k3[i] for i in range(6)])
        y = [y[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(6)]
        t += dt
        out.append((t, y[0], y[1]))
        if math.hypot(y[0], y[1]) < 2.0:                     # plunged (should not happen)
            break
    return np.array(out)


def draw(ax, blocks, view, norm, cmap):
    (x0, x1), (y0, y1) = view
    segs = []
    for (xa, xb, ya, yb, d) in sorted(blocks, key=lambda b: -(b[1] - b[0])):
        if xb < x0 or xa > x1 or yb < y0 or ya > y1:
            continue
        im = ax.imshow(d, origin='lower', extent=(xa, xb, ya, yb), norm=norm, cmap=cmap,
                       interpolation='nearest')
        segs += [[(xa, ya), (xb, ya)], [(xb, ya), (xb, yb)],
                 [(xb, yb), (xa, yb)], [(xa, yb), (xa, ya)]]
    ax.add_collection(LineCollection(segs, colors='w', linewidths=0.25, alpha=0.6))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rundir', help='run output dir containing bin/ and parfile.par')
    ap.add_argument('basename')
    ap.add_argument('--every', type=int, default=5, help='use every Nth dump')
    ap.add_argument('--outdir', default=None, help='default RUNDIR/plots/density')
    ap.add_argument('--global-half', type=float, default=40.0)
    ap.add_argument('--local-half', type=float, default=None,
                    help='star-frame half-width (default 3x the t=0 stellar radius)')
    ap.add_argument('--rt', type=float, default=16.4, help='tidal radius, areal r_g')
    ap.add_argument('--rp', type=float, default=10.0, help='design periapsis, areal r_g')
    ap.add_argument('--com', type=float, default=1e-2, help='CoM cut / rho_max')
    ap.add_argument('--decades', type=float, default=10.0, help='colour range below rhoc')
    a = ap.parse_args()

    pfile = os.path.join(a.rundir, 'parfile.par')
    rhoc = par(pfile, '<problem>', 'rhoc')
    outdir = a.outdir or os.path.join(a.rundir, 'plots', 'density')
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.rundir, 'bin', f'{a.basename}.prim_xy.*.bin')))
    files = files[::a.every] + ([files[-1]] if (len(files) - 1) % a.every else [])
    if not files:
        sys.exit('no prim_xy dumps')

    orb = analytic_orbit(pfile, t_max=400.0)
    r_t_iso = ic.trumpet_areal_to_iso(a.rt)
    r_p_iso = ic.trumpet_areal_to_iso(a.rp)
    norm = LogNorm(vmin=rhoc * 10 ** -a.decades, vmax=rhoc)
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('k')
    local_half = a.local_half

    for f in files:
        p = bc.read_binary(f)
        t = p['time']
        nx, ny = p['nx1_out_mb'], p['nx2_out_mb']
        g = np.asarray(p['mb_geometry'])
        dens = np.asarray(p['mb_data']['dens'])
        rmax = dens.max()
        blocks, cw, cx, cy, rr = [], 0.0, 0.0, 0.0, []
        for m in range(len(g)):
            d = dens[m, 0]
            blocks.append((g[m, 0], g[m, 1], g[m, 2], g[m, 3], np.ma.masked_invalid(d)))
            dx, dy = (g[m, 1] - g[m, 0]) / nx, (g[m, 3] - g[m, 2]) / ny
            X, Y = np.meshgrid(g[m, 0] + (np.arange(nx) + .5) * dx,
                               g[m, 2] + (np.arange(ny) + .5) * dy)
            c = d > a.com * rmax
            if c.any():
                w = d[c] * dx * dy
                cw += w.sum(); cx += (w * X[c]).sum(); cy += (w * Y[c]).sum()
                rr.append((X[c], Y[c]))
        cx, cy = cx / cw, cy / cw
        if local_half is None:     # fixed for every frame: 3x the t=0 stellar extent
            ext = max(np.hypot(X_ - cx, Y_ - cy).max() for X_, Y_ in rr)
            local_half = max(3.0 * ext, 1.0)

        fig, (ag, al) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        G = a.global_half
        im = draw(ag, blocks, ((-G, G), (-G, G)), norm, cmap)
        ag.plot(orb[:, 1], orb[:, 2], 'c--', lw=1.0, label='analytic geodesic')
        j = int(np.argmin(abs(orb[:, 0] - t)))
        ag.plot(orb[j, 1], orb[j, 2], 'o', mfc='none', mec='c', ms=9,
                label='analytic position')
        ag.plot(cx, cy, 'x', color='lime', ms=8, mew=1.5, label='star CoM')
        th = np.linspace(0, 2 * np.pi, 400)
        ag.plot(r_t_iso * np.cos(th), r_t_iso * np.sin(th), color='deepskyblue', lw=1.2,
                ls=':', label=f'tidal radius r_t={a.rt:g} (areal)')
        ag.plot(r_p_iso * np.cos(th), r_p_iso * np.sin(th), color='magenta', lw=0.9,
                ls=':', label=f'design periapsis {a.rp:g} (areal)')
        L = local_half
        ag.add_patch(plt.Rectangle((cx - L, cy - L), 2 * L, 2 * L, fill=False,
                                   ec='lime', lw=0.8))
        ag.plot(0, 0, '+', color='w', ms=8)
        ag.legend(loc='lower left', fontsize=8, framealpha=0.6)
        ag.set_title(f'global (BH at origin)   t = {t:.1f} M')
        ag.set_xlabel('x [M]'); ag.set_ylabel('y [M]')

        draw(al, blocks, ((cx - L, cx + L), (cy - L, cy + L)), norm, cmap)
        al.plot(cx, cy, 'x', color='lime', ms=8, mew=1.5)
        al.set_title(f'star frame, centred on CoM ({cx:.2f}, {cy:.2f}), '
                     f'R_areal={ic.trumpet_iso_to_areal(math.hypot(cx, cy)):.2f}')
        al.set_xlabel('x [M]')
        fig.colorbar(im, ax=[ag, al], shrink=0.85, label=r'$\rho$ (z=0)')
        fig.suptitle(f'{a.basename}   {os.path.basename(os.path.dirname(a.rundir.rstrip("/")))}'
                     f'   {len(g)} MeshBlocks in slice', fontsize=10)
        out = os.path.join(outdir, f'density_{t:07.2f}.png')
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(out)


if __name__ == '__main__':
    main()
