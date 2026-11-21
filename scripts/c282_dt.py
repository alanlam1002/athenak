#!/usr/bin/env python3
"""c282 (R-268, C-248): why WENOZ's dt at dx 1/32 (W32) is 2-4x smaller than PPMX's (L32, c203 ladder) before pericentre.
    c282_dt.py OUTDIR [--tmax 92] [--nproc 8] [--half 4] [--mesh 0.4]
Writes
  c282_dt.png        code dt vs t (W32, L32, and the 1/16 pair A1W/A1P), dt ratio WENOZ/PPMX at matched t, cycles vs t,
                     CRECOV recoveries per step, slice-estimated dt (xy plane) vs the code dt, and where the limiting
                     slice cell sits (distance from the star's centre, rho/rho_c, population)
  c282_dt_cells.csv  per slice dump: t, slice dt, the limiting cell's x, y, r/R*, rho/rho_c, |V|, c_s, alpha, dx, population
  c282_dt_anim.{gif,mp4}  W32 | L32 side by side, z = 0 plane around the star: log10 of the cell's CFL dt, rho contours
                     (1e-1, 1e-3, 1e-6 rho_c), MeshBlock outlines; + = the slice's limiting cell (arrow when outside)
Cell CFL dt (ARGUED approximation, as tde_dt_cells.py): rate_a = (|alpha v^a - beta^a| + alpha c_s / sqrt(gamma_aa)) / dx_a,
v^a = u^a / W, c_s^2 = Gamma P/(rho h), dt_cell = cfl / max_a rate_a; cells with alpha < excise_lapse are dropped
(dt_skip_excised = true in both decks). Only the z = 0 plane is dumped with the metric (adm_xy), so the slice dt is an
upper bound on the code's (validated against the code dt in panel e). Read-only on the runs."""
import argparse, glob, os, re, subprocess, sys, types
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.colors import LogNorm, Normalize    # noqa: E402
from matplotlib.collections import LineCollection   # noqa: E402

sys.modules.setdefault('h5py', types.ModuleType('h5py'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vis', 'python'))
import bin_convert as bc                           # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
B = 'cfc_tde_wd_imbh_parabolic_tracker'
RUNS = {'W32': (f'{C}/c282_w32/W32', '#d9731a', 'WENOZ 1/32 (W32)'),
        'L32': (f'{C}/c203_chain/L32_P18_r82', '#1f5fa8', 'PPMX 1/32 (L32)'),
        'A1W': (f'{C}/c276_wenoz/A1W', '#d9731a', 'WENOZ 1/16 (A1W)'),
        'A1P': (f'{C}/c276_wenoz/A1P', '#1f5fa8', 'PPMX 1/16 (A1P)')}
GAM, CFL, XL = 5.0 / 3.0, 0.4, 0.25          # W32.par:222, :164, :181 (same in L32_P18_r82.par)
RHOC, RSTAR = 3.24e-4, 0.677
POPS = {'star (>1e-3 rho_c)': '#2ca02c', 'envelope/bow (<4 R*)': '#9467bd', 'atmosphere': '#7f7f7f', 'near BH (alpha<0.5)': 'k'}


def hst(d):
    h = np.loadtxt(glob.glob(f'{d}/{B}.mhd.hst')[0])
    _, i = np.unique(h[:, 0][::-1], return_index=True)
    return h[::-1][i]


def crecov(d):
    ct, rows = {}, {}
    for lg in sorted(glob.glob(f'{d}/out*.log')):
        for l in open(lg):
            if l.startswith('CRECOV') and 'path=steps' in l:
                m = dict(re.findall(r'(\w+)=([-0-9.e+]+)', l)); rows[int(m['cycle'])] = float(m['n'])
            else:
                m = re.search(r'cycle=(\d+) time=([0-9.e+-]+)', l)
                if m and not l.startswith('ATMZONE'):
                    ct[int(m.group(1))] = float(m.group(2))
    c = np.array(sorted(rows)); kc = np.array(sorted(ct))
    return np.interp(c, kc, [ct[k] for k in kc]), np.array([rows[k] for k in c])


def dumps(d, kind, tmax):
    out = {}
    for f in glob.glob(f'{d}/bin/{B}.{kind}.*.bin'):
        out[int(f.split('.')[-2])] = f
    return [out[k] for k in sorted(out)]


def rates(fp, fa):
    p, q = bc.read_binary(fp), bc.read_binary(fa)
    md = {k: np.asarray(v)[:, 0] for k, v in p['mb_data'].items()}
    ad = {k: np.asarray(v)[:, 0] for k, v in q['mb_data'].items()}
    g = np.asarray(p['mb_geometry']); nx = p['nx1_out_mb']
    rho, P = md['dens'], md['press']
    u = [md['velx'], md['vely'], md['velz']]
    gd = [ad['adm_gxx'], ad['adm_gyy'], ad['adm_gzz']]
    alp = ad['adm_alpha']; beta = [ad['adm_betax'], ad['adm_betay'], ad['adm_betaz']]
    W = np.sqrt(1 + sum(gd[a] * u[a]**2 for a in range(3)))
    cs = np.sqrt(np.clip(GAM * P / (rho + GAM / (GAM - 1) * P), 0, None))
    dx = ((g[:, 1] - g[:, 0]) / nx)[:, None, None]
    rate = np.zeros_like(rho)
    for a in range(3):
        rate = np.maximum(rate, (np.abs(alp * u[a] / W - beta[a]) + alp * cs / np.sqrt(gd[a])) / dx)
    rate[alp < XL] = 0.0
    dtc = np.where(rate > 0, CFL / np.where(rate > 0, rate, 1), np.inf)
    xc = np.broadcast_to(g[:, 0, None, None] + (np.arange(nx)[None, None, :] + .5) * dx, rho.shape)
    yc = np.broadcast_to(g[:, 2, None, None] + (np.arange(nx)[None, :, None] + .5) * dx, rho.shape)
    m = np.unravel_index(np.argmax(rho), rho.shape)
    V = np.sqrt(sum(gd[a] * (u[a] / W)**2 for a in range(3)))
    return dict(t=p['time'], g=g, nx=nx, dtc=dtc, rho=rho, cs=cs, V=V, alp=alp, x=xc, y=yc, dx=np.broadcast_to(dx, rho.shape),
                xs=xc[m], ys=yc[m])


def pop(s, i):
    if s['alp'][i] < 0.5: return 'near BH (alpha<0.5)'
    if s['rho'][i] > 1e-3 * RHOC: return 'star (>1e-3 rho_c)'
    if np.hypot(s['x'][i] - s['xs'], s['y'][i] - s['ys']) < 4 * RSTAR: return 'envelope/bow (<4 R*)'
    return 'atmosphere'


def limit(s):
    i = np.unravel_index(np.argmin(s['dtc']), s['dtc'].shape)
    return dict(t=s['t'], dt=s['dtc'][i], x=s['x'][i], y=s['y'][i], rR=np.hypot(s['x'][i] - s['xs'], s['y'][i] - s['ys']) / RSTAR,
                rho=s['rho'][i] / RHOC, V=s['V'][i], cs=s['cs'][i], alp=s['alp'][i], dx=s['dx'][i], pop=pop(s, i))


def cells_one(args):
    lab, fp, fa = args
    return lab, limit(rates(fp, fa))


def draw(ax, s, half, mesh, norm, title, lim):
    xs, ys = s['xs'], s['ys']
    g, nx = s['g'], s['nx']
    for b in range(g.shape[0]):
        xa, xb, ya, yb = g[b, 0], g[b, 1], g[b, 2], g[b, 3]
        if xb < xs - half or xa > xs + half or yb < ys - half or ya > ys + half: continue
        im = ax.imshow(np.log10(s['dtc'][b]), origin='lower', extent=(xa, xb, ya, yb), cmap='magma', norm=norm,
                       interpolation='nearest')
    segs = [[(g[b, 0], g[b, 2]), (g[b, 1], g[b, 2]), (g[b, 1], g[b, 3]), (g[b, 0], g[b, 3]), (g[b, 0], g[b, 2])]
            for b in range(g.shape[0]) if not (g[b, 1] < xs - half or g[b, 0] > xs + half or g[b, 3] < ys - half or g[b, 2] > ys + half)]
    if mesh: ax.add_collection(LineCollection(segs, colors='0.25', linewidths=mesh, alpha=0.5))
    for b in range(g.shape[0]):
        xa, xb, ya, yb = g[b, 0], g[b, 1], g[b, 2], g[b, 3]
        if xb < xs - half or xa > xs + half or yb < ys - half or ya > ys + half: continue
        X = np.linspace(xa, xb, nx + 1)[:-1] + (xb - xa) / nx / 2; Y = np.linspace(ya, yb, nx + 1)[:-1] + (yb - ya) / nx / 2
        r = s['rho'][b] / RHOC
        if r.max() > 1e-6 and r.min() < 1e-1:
            ax.contour(X, Y, np.log10(np.maximum(r, 1e-30)), levels=[-6, -3, -1], colors=['#00bfff', '#00bfff', '#0000cd'],
                       linewidths=[0.5, 0.8, 1.1], linestyles=[':', '--', '-'])
    ax.set_xlim(xs - half, xs + half); ax.set_ylim(ys - half, ys + half); ax.set_aspect('equal')
    lx, ly = lim['x'], lim['y']
    if abs(lx - xs) < half and abs(ly - ys) < half:
        ax.plot(lx, ly, '+', color='lime', ms=14, mew=2)
    else:
        ang = np.arctan2(ly - ys, lx - xs); ex, ey = xs + 0.85 * half * np.cos(ang), ys + 0.85 * half * np.sin(ang)
        ax.annotate(f'limiting cell at r = {lim["rR"]:.1f} R*', (ex, ey), xytext=(xs, ys + 0.9 * half), color='lime', fontsize=8,
                    ha='center', arrowprops=dict(arrowstyle='->', color='lime'))
    ax.set_title(title, fontsize=10); ax.set_xlabel('x [M]'); ax.set_ylabel('y [M]')
    return im


def frame(args):
    i, fw, aw, fl, al, codew, codel, half, mesh, out = args
    sw, sl = rates(fw, aw), rates(fl, al)
    lw, ll = limit(sw), limit(sl)
    norm = Normalize(vmin=-2.0, vmax=0.0)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2), constrained_layout=True)
    im = draw(ax[0], sw, half, mesh, norm, f'WENOZ W32  t = {sw["t"]:.1f}\ncode dt {codew:.4f}, slice dt {lw["dt"]:.4f} ({lw["pop"]})', lw)
    draw(ax[1], sl, half, mesh, norm, f'PPMX L32  t = {sl["t"]:.1f}\ncode dt {codel:.4f}, slice dt {ll["dt"]:.4f} ({ll["pop"]})', ll)
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label('log10 cell CFL dt  (cfl / max wave rate)')
    fig.suptitle('c282: where the time step is set, z = 0 plane around the star (blue: rho = 0.1 / 1e-3 / 1e-6 rho_c; + limiting cell)',
                 fontsize=10)
    fn = f'{out}/frames/f{i:04d}.png'; fig.savefig(fn, dpi=80); plt.close(fig)
    return fn


def slab(f, plane, var='rho'):
    """slice: block boxes (a0, a1, b0, b1) in the plane's two axes and rho/rho_c or P/rho (nmb, nb, na)"""
    p = bc.read_binary(f)
    r = np.asarray(p['mb_data']['dens'])
    if var == 'T':
        r = np.asarray(p['mb_data']['press']) / r
    else:
        r = r / RHOC
    g = np.asarray(p['mb_geometry'])
    if plane == 'xy':
        return p['time'], g[:, [0, 1, 2, 3]], r[:, 0]
    return p['time'], g[:, [0, 1, 4, 5]], r[:, :, 0, :]


def draw_rho(ax, t, g, r, ctr, half, mesh, norm, title, ylab, cmap='inferno'):
    xs, ys = ctr
    keep = ~((g[:, 1] < xs - half) | (g[:, 0] > xs + half) | (g[:, 3] < ys - half) | (g[:, 2] > ys + half))
    im = None
    for b in np.nonzero(keep)[0]:
        im = ax.imshow(np.maximum(r[b], 1e-30), origin='lower', extent=tuple(g[b]), cmap=cmap, norm=norm,
                       interpolation='nearest')
    if mesh:
        segs = [[(g[b, 0], g[b, 2]), (g[b, 1], g[b, 2]), (g[b, 1], g[b, 3]), (g[b, 0], g[b, 3]), (g[b, 0], g[b, 2])]
                for b in np.nonzero(keep)[0]]
        ax.add_collection(LineCollection(segs, colors='w', linewidths=mesh, alpha=0.35))
    ax.add_patch(plt.Circle((xs, ys), RSTAR, fill=False, ec='c', ls=':', lw=0.8))
    ax.set_xlim(xs - half, xs + half); ax.set_ylim(ys - half, ys + half); ax.set_aspect('equal')
    ax.set_title(title, fontsize=10); ax.set_xlabel('x [M]'); ax.set_ylabel(ylab)
    return im


def frame_rho(args):
    i, fw, fl, fwz, flz, half, mesh, out = args
    norm = LogNorm(vmin=1e-21, vmax=1.0)          # dfloor 1e-24 = 3.1e-21 rho_c (W32.par:219)
    tnorm = LogNorm(vmin=1e-4, vmax=1e-1)
    fig, ax = plt.subplots(3, 2, figsize=(11.5, 16), constrained_layout=True)
    for col, (fxy, fxz, lab) in enumerate(((fw, fwz, 'WENOZ W32'), (fl, flz, 'PPMX L32'))):
        t, g, r = slab(fxy, 'xy')
        m = np.unravel_index(np.argmax(r), r.shape)          # star centre = density maximum
        nx = r.shape[-1]; dx = (g[m[0], 1] - g[m[0], 0]) / nx
        ctr = (g[m[0], 0] + (m[2] + .5) * dx, g[m[0], 2] + (m[1] + .5) * dx)
        im = draw_rho(ax[0, col], t, g, r, ctr, half, mesh, norm, f'{lab}  t = {t:.1f}  rho (z = 0)', 'y [M]')
        t2, g2, r2 = slab(fxz, 'xz')
        draw_rho(ax[1, col], t2, g2, r2, (ctr[0], 0.0), half, mesh, norm, f'{lab}  t = {t2:.1f}  rho (y = 0)', 'z [M]')
        _, g3, r3 = slab(fxy, 'xy', 'T')
        imt = draw_rho(ax[2, col], t, g3, r3, ctr, half, mesh, tnorm, f'{lab}  t = {t:.1f}  P/rho (z = 0)', 'y [M]', 'viridis')
    fig.colorbar(im, ax=ax[:2, :], shrink=0.7).set_label('rho / rho_c (floor 3.1e-21)')
    fig.colorbar(imt, ax=ax[2, :], shrink=0.9).set_label('P / rho (temperature proxy)')
    fig.suptitle(f'c282: WENOZ vs PPMX at dx 1/32 around the star (dotted circle: R* = {RSTAR}; ±{half:g} M)', fontsize=11)
    fn = f'{out}/frames_rho/f{i:04d}.png'; fig.savefig(fn, dpi=65); plt.close(fig)
    return fn


def make_movie(fr, out, stem, fps=4):
    from PIL import Image
    ims = [Image.open(f).convert('P', palette=Image.ADAPTIVE) for f in fr]
    ims[0].save(f'{out}/{stem}.gif', save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
    ff = sorted(glob.glob('/opt/aurora/*/spack/unified/*/install/linux-x86_64/ffmpeg-*/bin/ffmpeg'))
    if ff:
        d = os.path.dirname(fr[0])
        subprocess.run([ff[-1], '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{d}/f%04d.png', '-vf',
                        'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', f'{out}/{stem}.mp4'])


def main_rho(a):
    os.makedirs(f'{a.out}/frames_rho', exist_ok=True)
    tw = hst(RUNS['W32'][0])[:, 0].max()
    D = {k: (dumps(RUNS[k][0], 'prim_xy', 0), dumps(RUNS[k][0], 'prim_xz', 0)) for k in ('W32', 'L32')}
    n = min(len(D['W32'][0]), len(D['W32'][1]), len(D['L32'][0]))
    jobs = [(i, D['W32'][0][i], D['L32'][0][i], D['W32'][1][i], D['L32'][1][i], a.half, a.mesh, a.out) for i in range(n)]
    with ProcessPoolExecutor(a.nproc) as ex:
        fr = list(ex.map(frame_rho, jobs))
    make_movie(fr, a.out, f'c282_rho_anim_h{a.half:g}')
    print('wrote density frames', n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--tmax', type=float, default=92.0); ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--half', type=float, default=4.0); ap.add_argument('--mesh', type=float, default=0.4)
    ap.add_argument('--rho', action='store_true', help='density animation only')
    a = ap.parse_args()
    if a.rho:
        return main_rho(a)
    os.makedirs(f'{a.out}/frames', exist_ok=True)
    H = {k: hst(v[0]) for k, v in RUNS.items()}
    CR = {k: crecov(v[0]) for k, v in RUNS.items()}
    # slice dumps, matched by index (both write every 2 M from t = 0)
    D = {k: (dumps(RUNS[k][0], 'prim_xy', a.tmax), dumps(RUNS[k][0], 'adm_xy', a.tmax)) for k in ('W32', 'L32')}
    tw = H['W32'][:, 0].max()
    n = min(len(D['W32'][0]), len(D['L32'][0]))
    jobs = []
    for k in ('W32', 'L32'):
        for j in range(n):
            jobs.append((k, D[k][0][j], D[k][1][j]))
    with ProcessPoolExecutor(a.nproc) as ex:
        res = list(ex.map(cells_one, jobs))
    L = {'W32': [r for k, r in res if k == 'W32'], 'L32': [r for k, r in res if k == 'L32']}
    L['L32'] = [r for r in L['L32'] if r['t'] <= tw + 0.01]
    with open(f'{a.out}/c282_dt_cells.csv', 'w') as f:
        f.write('run,t,slice_dt,x,y,r_over_Rstar,rho_over_rhoc,V,cs,alpha,dx,population\n')
        for k in ('W32', 'L32'):
            for r in L[k]:
                f.write(f"{k},{r['t']:.3f},{r['dt']:.5g},{r['x']:.4f},{r['y']:.4f},{r['rR']:.3f},{r['rho']:.3e},{r['V']:.4f},"
                        f"{r['cs']:.4f},{r['alp']:.4f},{r['dx']:.5f},{r['pop']}\n")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for k, (d, c, lab) in RUNS.items():
        h = H[k]; m = h[:, 0] <= max(tw, 200) if k.startswith('A') else h[:, 0] <= tw + 1
        ls = '-' if k in ('W32', 'L32') else ':'
        ax[0, 0].semilogy(h[m, 0], h[m, 1], ls, color=c, lw=1, label=lab)
        ax[0, 2].semilogy(*CR[k], ls, color=c, lw=0.6, label=lab) if len(CR[k][0]) else None
    for (kw, kp, lab) in (('W32', 'L32', '1/32: W32 / L32'), ('A1W', 'A1P', '1/16: A1W / A1P')):
        t = H[kw][:, 0]; t = t[t <= min(H[kp][:, 0].max(), H[kw][:, 0].max())]
        ax[0, 1].plot(t, np.interp(t, H[kw][:, 0], H[kw][:, 1]) / np.interp(t, H[kp][:, 0], H[kp][:, 1]),
                      '-' if kw == 'W32' else ':', color='k', lw=1, label=lab)
    ax[0, 1].axhline(1, color='0.6', lw=0.8)
    for k in ('W32', 'L32'):
        h = H[k]; m = h[:, 0] <= tw + 1
        t = h[m, 0]; ncyc = np.cumsum(np.r_[0, np.diff(t) / h[m, 1][:-1]])
        ax[1, 0].plot(t, ncyc, color=RUNS[k][1], label=f'{RUNS[k][2]}: {ncyc[-1]:.0f} cycles to t = {t[-1]:.1f}')
        ax[1, 1].semilogy(h[m, 0], h[m, 1], color=RUNS[k][1], lw=0.6, alpha=0.6, label=f'{k} code dt')
        for p, pc in POPS.items():
            q = [r for r in L[k] if r['pop'] == p]
            if q:
                ax[1, 1].semilogy([r['t'] for r in q], [r['dt'] for r in q], 'o' if k == 'W32' else 's', mfc=pc, mec=RUNS[k][1], ms=5,
                                  label=f'{k} slice dt, limit in {p}')
                ax[1, 2].semilogy([r['t'] for r in q], [r['rR'] for r in q], 'o' if k == 'W32' else 's', mfc=pc, mec=RUNS[k][1], ms=5,
                                  label=f'{k}: {p}')
    for (i, j), (xl, yl, ti) in {(0, 0): ('t [M]', 'dt (hst)', 'code time step'),
                                 (0, 1): ('t [M]', 'dt WENOZ / dt PPMX', 'dt ratio at matched t'),
                                 (0, 2): ('t [M]', 'C2P recoveries per step (CRECOV n)', 'failed-C2P recoveries'),
                                 (1, 0): ('t [M]', 'cycles', 'cycles to reach t (integral of 1/dt)'),
                                 (1, 1): ('t [M]', 'dt', 'slice (z = 0) CFL estimate vs code dt; colour = population'),
                                 (1, 2): ('t [M]', 'r / R* of the limiting slice cell', 'limiting cell: distance from star centre')}.items():
        ax[i, j].set_xlabel(xl); ax[i, j].set_ylabel(yl); ax[i, j].set_title(ti, fontsize=10); ax[i, j].grid(alpha=0.3)
        ax[i, j].legend(fontsize=7)
    ax[0, 0].set_xlim(0, max(tw, 100)); ax[0, 1].set_xlim(0, max(tw, 100)); ax[0, 2].set_xlim(0, max(tw, 100))
    fig.suptitle(f'c282 (R-268): WENOZ vs PPMX time step at dx 1/32 (W32 to t = {tw:.1f}) with the 1/16 pair (dotted)')
    fig.tight_layout(); fig.savefig(f'{a.out}/c282_dt.png', dpi=80); plt.close(fig)

    nf = min(len(L['W32']), len(L['L32']))
    fj = [(i, D['W32'][0][i], D['W32'][1][i], D['L32'][0][i], D['L32'][1][i],
           np.interp(L['W32'][i]['t'], H['W32'][:, 0], H['W32'][:, 1]), np.interp(L['L32'][i]['t'], H['L32'][:, 0], H['L32'][:, 1]),
           a.half, a.mesh, a.out) for i in range(nf)]
    with ProcessPoolExecutor(a.nproc) as ex:
        fr = list(ex.map(frame, fj))
    from PIL import Image
    ims = [Image.open(f).convert('P', palette=Image.ADAPTIVE) for f in fr]
    ims[0].save(f'{a.out}/c282_dt_anim.gif', save_all=True, append_images=ims[1:], duration=300, loop=0)
    ff = sorted(glob.glob('/opt/aurora/*/spack/unified/*/install/linux-x86_64/ffmpeg-*/bin/ffmpeg'))
    if ff:
        subprocess.run([ff[-1], '-y', '-loglevel', 'error', '-framerate', '4', '-i', f'{a.out}/frames/f%04d.png', '-vf',
                        'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', f'{a.out}/c282_dt_anim.mp4'])
    print('wrote', a.out, 'frames', nf)


if __name__ == '__main__':
    main()
