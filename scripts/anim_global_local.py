#!/usr/bin/env python3
"""Side-by-side animation of one TDE run: global view (star + BH, z = 0 density) and the
star's local frame (±2 R* around rho_max: density and K/K0).

The global panel also shows the tidal radius r_t (circle about the BH, --rt) and the
test-particle geodesic of the release state (tde_parabolic_orbit_ic: E = 1, areal periapsis
--periapsis, released at the deck's star-BH separation), drawn about x_BH(t) since it is
integrated in the BH-centred trumpet coordinates (the comoving gauge is a uniform spatial
shift, same coordinate time); the marker is the geodesic's position at the frame's t.

The BH position is x_BH(t) = puncture_x - xi(t), with xi = the history's frame_disp
columns (comoving gauge); in the BH frame xi = 0 and x_BH = puncture_x (or the origin).
Frames render in parallel to PNGs, then join into a GIF (and an MP4 if ffmpeg exists).

RUNDIR may be a comma-separated list of segments of one run (e.g. a run and its restart
continuation in another directory); their prim_xy dumps are joined in time order, later
segments winning on duplicate times.

    anim_global_local.py RUNDIR[,RUNDIR2...] [--label L4] [--half 40] [--every 1] [--rt 16.4]
                         [--periapsis 10] [--out DIR]
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm   # noqa: E402
from PIL import Image   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_ovn_plots as O                         # noqa: E402
from tde_ensemble_plots import bc                 # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
A = {}


def par_get(d, k, dflt=0.0):
    m = re.search(rf'^{k}\s*=\s*([-+.\deE]+)', open(f'{d}/parfile.par').read(), re.M)
    return float(m.group(1)) if m else dflt


def geodesic(d, rp, t_max, dt=0.05):
    """test-particle geodesic (t, x, y) relative to the BH, released at the deck's separation"""
    import tde_parabolic_orbit_ic as P
    x0 = par_get(d, 'star_center_x1') - par_get(d, 'puncture_x1')
    s = P.solve_ic(x0, rp)
    o = P.integrate(x0, s['px'], s['py'], t_max, dt)
    return o[:, 0], o[:, 1], o[:, 2]


def bh_track(d):
    get = lambda k, dflt=0.0: par_get(d, k, dflt)
    p0 = np.array([get('puncture_x1'), get('puncture_x2'), get('puncture_x3')])
    h = np.loadtxt(glob.glob(f'{d}/*.user.hst')[0])
    hdr = open(glob.glob(f'{d}/*.user.hst')[0]).readlines()[1]
    if 'frame_disp_x1' in hdr:
        c = [int(re.search(rf'\[(\d+)\]=frame_disp_x{a}', hdr).group(1)) - 1 for a in (1, 2, 3)]
        xi = h[:, c]
    else:
        xi = np.zeros((len(h), 3))
    return h[:, 0], p0[None, :] - xi


def frame(job):
    n, f, png = job
    lab, half, t_h, xb_h = A['lab'], A['half'], A['t'], A['xb']
    k = O.consts(lab)
    s = O.read(f, k)
    xb = [np.interp(s['t'], t_h, xb_h[:, a]) for a in (0, 1)]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4), gridspec_kw=dict(width_ratios=[1.25, 1, 1]),
                           constrained_layout=True)
    a = ax[0]
    O.draw(a, O.blocks(s, s['d'] / k['rhoc']), ((-half, half), (-half, half)),
           LogNorm(vmin=1e-9, vmax=1.0), O.CR, outline=False)
    a.plot(*xb, 'o', ms=7, mfc='none', mec='c', mew=1.5)
    a.annotate('BH', xb, xytext=(6, 6), textcoords='offset points', color='c', fontsize=9)
    if A['rt'] > 0:
        a.add_patch(plt.Circle(xb, A['rt'], fill=False, ec='y', ls=':', lw=1.2))
        a.annotate('r_t', (xb[0] + A['rt'] / np.sqrt(2), xb[1] + A['rt'] / np.sqrt(2)), color='y',
                   fontsize=9)
    if A['geo'] is not None:
        tg, gx, gy = A['geo']
        a.plot(xb[0] + gx, xb[1] + gy, '--', color='lime', lw=0.9,
               label=f'geodesic (r_p = {A["rp"]:g}; × = its position at t)')
        if s['t'] <= tg[-1]:
            a.plot(xb[0] + np.interp(s['t'], tg, gx), xb[1] + np.interp(s['t'], tg, gy), 'x', color='lime',
                   ms=8, mew=1.8)
        a.legend(loc='lower left', fontsize=8, facecolor='w', framealpha=0.8)
    h2 = 2 * k['rstar']
    a.add_patch(plt.Rectangle((s['x0'] - h2, s['y0'] - h2), 2 * h2, 2 * h2, fill=False, ec='w', lw=0.8))
    a.set_aspect('equal'); a.set_xlabel('x [M]'); a.set_ylabel('y [M]')
    r = np.hypot(s['x0'] - xb[0], s['y0'] - xb[1])
    a.set_title(f'global, z = 0 (grid frame), ±{half:.0f} M; star–BH {r:.1f} M')
    for j, fld in ((1, 'rho'), (2, 'K')):
        O.star_panel(ax[j], lab, s, fld)
        ax[j].set_aspect('equal')
    ax[1].set_title(rf'star frame ±2 R*: $\rho/\rho_c$ (max {s["rmax"]:.3f})')
    ax[2].set_title(rf'star frame: $K/K_0$ (at $\rho_{{max}}$ {s["kmax"]:.3f})')
    fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(1e-9, 1), cmap=O.CR), ax=ax[0], shrink=0.85,
                 label=r'$\rho/\rho_c$')
    fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(1e-7, 1.2), cmap=O.CR), ax=ax[1], shrink=0.85)
    fig.colorbar(plt.cm.ScalarMappable(norm=O.NK, cmap=O.CK), ax=ax[2], shrink=0.85, label=r"$K/K_0$")
    fig.suptitle(f'{A["title"]}   t = {s["t"]:.1f} M')
    fig.savefig(png, dpi=A['dpi']); plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--label', default='run')
    ap.add_argument('--title', default=None)
    ap.add_argument('--half', type=float, default=30.0)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--dpi', type=int, default=60)
    ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--rt', type=float, default=16.4, help='tidal radius [M]; 0 = off')
    ap.add_argument('--periapsis', type=float, default=10.0, help='geodesic areal r_p; 0 = off')
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    runs = a.run.split(',')
    O.RUNS[a.label] = (runs[0], 'k', '-', a.label)
    O.NK = TwoSlopeNorm(vcenter=1.0, vmin=0.8, vmax=6.0)   # K/K0 from 1 to ~6 after the heating
    tr = [bh_track(d) for d in runs]
    t_h = np.concatenate([t for t, _ in tr]); xb = np.concatenate([x for _, x in tr])
    o = np.argsort(t_h, kind='stable'); t_h, xb = t_h[o], xb[o]
    byt = {}
    for d in runs:                                  # later segments win on duplicate times
        for f in sorted(glob.glob(f'{d}/bin/{B}.prim_xy.*.bin')):
            byt[round(float(bc.read_binary(f)['time']), 6)] = f
    fs = [byt[t] for t in sorted(byt)][::a.every]
    geo = geodesic(runs[0], a.periapsis, t_h[-1] + 1.0) if a.periapsis > 0 else None
    A.update(lab=a.label, half=a.half, t=t_h, xb=xb, dpi=a.dpi, title=a.title or a.label,
             rt=a.rt, geo=geo, rp=a.periapsis)
    tmp = f'{a.out}/frames_{a.label}'
    os.makedirs(tmp, exist_ok=True)
    jobs = [(i, f, f'{tmp}/f{i:04d}.png') for i, f in enumerate(fs)]
    with Pool(a.nproc) as p:
        pngs = p.map(frame, jobs)
    ims = [Image.open(f).convert('P', palette=Image.ADAPTIVE) for f in pngs]
    gif = f'{a.out}/{a.label}_global_local.gif'
    ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(1000 / a.fps), loop=0)
    print(gif, os.path.getsize(gif) // 1024, 'kB,', len(ims), 'frames')
    if shutil.which('ffmpeg'):
        mp4 = f'{a.out}/{a.label}_global_local.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(a.fps), '-i', f'{tmp}/f%04d.png',
                        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', mp4], check=True)
        print(mp4, os.path.getsize(mp4) // 1024, 'kB')


if __name__ == '__main__':
    main()
