#!/usr/bin/env python3
"""c285 test 4 (R-271): figures and animation of the P10 pilot with <mhd> atm_ceiling (P10C, from 00019 t = 654.39 to 684.4) against
the P10 pilot's own run over the same window (P10_r82, athena_c203 = the same source without the ceiling, same restart and ranks).
    c285_p10_plots.py OUTDIR [--nproc 8] [--mesh 0.3]
Writes
  c285_p10.png         dt, cycles, CRECOV per step, rho_max, the ATMCEIL ledger (reset cells / mass, dS, d(tau+D), near-band mass
                       vs the alarm limit), each class's minimum density over the threshold (classes_*.csv from c285_classes.py)
  c285_p10_census.png  whole-domain mass per density decade, ceiling vs reference at the matched 3D dumps (t = 660, 672, 684),
                       and the relative difference per decade (the gas above the band must be unchanged)
  c285_p10_anim.{gif,mp4}  z = 0 slices every 2 M: rho (reference | ceiling | |d rho|/rho), P/rho (reference | ceiling), BH marked,
                       MeshBlock outlines"""
import argparse, csv, glob, os, re, subprocess, sys, types
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LogNorm               # noqa: E402
from matplotlib.collections import LineCollection   # noqa: E402
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/vis/python')
import bin_convert as bc                            # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
B = 'cfc_tde_wd_imbh_parabolic_tracker'
REF, CEIL = f'{C}/c203_chain/P10_r82', f'{C}/c285_ceil/P10C'
DECK = f'{C}/c285_ceil/P10C.par'
RC, MS, T0, T1 = 3.24e-4, 7.47e-5, 654.39, 684.4
THR = 100*1.0e-24/RC
ST = {'ref': ('#1f5fa8', 'P10 reference (no ceiling)'), 'ceil': ('#d9731a', 'P10C (ceiling)')}


def hst(d, kind):
    h = np.loadtxt(glob.glob(f'{d}/{B}.{kind}.hst')[0])
    _, i = np.unique(h[:, 0][::-1], return_index=True)
    h = h[::-1][i]
    return h[(h[:, 0] >= T0 - 1e-6) & (h[:, 0] <= T1 + 1e-6)]


def logscan(d, tag, keys):
    out = []
    for lg in sorted(glob.glob(f'{d}/out*.log')):
        for l in open(lg):
            if l.startswith(tag + ' '):
                m = dict(re.findall(r'(\w+)=\(?([-0-9.e+]+)', l))
                try:
                    out.append([float(m[k]) for k in keys])
                except KeyError:
                    pass
    return np.array(out)


def crecov(d):
    ct, rows = {}, {}
    for lg in sorted(glob.glob(f'{d}/out*.log')):
        for l in open(lg):
            if l.startswith('CRECOV') and 'path=steps' in l:
                m = dict(re.findall(r'(\w+)=([-0-9.e+]+)', l)); rows[int(m['cycle'])] = float(m['n'])
            else:
                m = re.search(r'cycle=(\d+) time=([0-9.e+-]+)', l)
                if m and not l.startswith(('ATMZONE', 'ATMCEIL')):
                    ct[int(m.group(1))] = float(m.group(2))
    c = np.array(sorted(rows)); kc = np.array(sorted(ct))
    t = np.interp(c, kc, [ct[k] for k in kc]); n = np.array([rows[k] for k in c])
    s = (t >= T0) & (t <= T1)
    return t[s], n[s]


def rd(f):
    return list(csv.DictReader(open(f))) if os.path.exists(f) else []


def col(rows, k):
    return np.array([float(r[k]) if r.get(k) not in (None, '', 'nan') else np.nan for r in rows])


def figures(out):
    fig, ax = plt.subplots(3, 3, figsize=(17, 13))
    for lab, d in (('ref', REF), ('ceil', CEIL)):
        c, name = ST[lab]
        h = hst(d, 'mhd'); u = hst(d, 'user')
        ax[0, 0].plot(h[:, 0], h[:, 1], color=c, lw=0.8, label=name)
        cyc = np.cumsum(np.r_[0, np.diff(h[:, 0])/h[:-1, 1]])
        ax[0, 1].plot(h[:, 0], cyc, color=c, label=f'{name}: {cyc[-1]:.0f} cycles')
        t, n = crecov(d)
        ax[0, 2].semilogy(t, n, color=c, lw=0.6, label=name)
        ax[1, 0].plot(u[:, 0], u[:, 2]/RC, color=c, lw=0.8, label=name)
        ax[1, 1].plot(h[:, 0], (h[:, 2] - h[0, 2])/h[0, 2], color=c, lw=0.8, label=name)
    L = logscan(CEIL, 'ATMCEIL', ['time', 'n', 'M_reset', 'max_rho_reset', 'dM', 'dE', 'near', 'n_near', 'cum_dE'])
    near = logscan(CEIL, 'ATMCEIL', ['time'])
    nb = []
    for lg in sorted(glob.glob(f'{CEIL}/out*.log')):
        for l in open(lg):
            if l.startswith('ATMCEIL '):
                m = re.search(r'near=\(([^)]*)\)', l); nb.append(sum(float(x) for x in m.group(1).split(',')))
    nb = np.array(nb)
    a = ax[1, 2]
    a.semilogy(L[:, 0], L[:, 2]/MS, color='#d9731a', label='mass of the reset cells / M*')
    a.semilogy(L[:, 0], nb/MS, color='#7f3fbf', label='near-band mass (thr..1e3 thr) / M*')
    a.axhline(1e-15/MS, color='r', ls=':', label='alarm limit 1e-15 (abs) / M*')
    a.semilogy(L[:, 0], np.maximum(np.abs(L[:, 5]), 1e-40)/MS, color='0.4', lw=0.7, label='|d(tau + D)| per cycle / M*')
    ax2 = ax[2, 0]
    ax2.plot(L[:, 0], L[:, 1], color='#d9731a', label='cells reset per cycle')
    ax2.plot(L[:, 0], L[:, 7], color='#7f3fbf', label='near-band cells')
    ax2.set_yscale('log')
    for lab, f, mk in (('ref', 'classes_P10ref.csv', 's'), ('ceil', 'classes_P10C.csv', 'o')):
        r = rd(f'{out}/{f}'); c = ST[lab][0]
        if not r: continue
        t = col(r, 't')
        for k, ls in (('core', ':'), ('outer', '--'), ('stream_nbr', '-')):
            if f'{k}_min_over_thr' in r[0]:
                ax[2, 1].semilogy(t, col(r, f'{k}_min_over_thr'), ls, marker=mk, color=c, label=f'{lab} {k}')
        ax[2, 2].semilogy(t, np.maximum(col(r, 'stream_Mband') + col(r, 'stream_nbr_Mband'), 1e-30), marker=mk, color=c,
                          label=f'{lab}: band mass in stream + its 1 M neighbourhood / M*')
        ax[2, 2].semilogy(t, col(r, 'M_band_all'), ls=':', marker=mk, color=c, label=f'{lab}: band mass, whole domain / M*')
    ax[2, 1].axhline(1e3, color='r', ls=':', label='pass: >= 1e3 x threshold')
    T = {(0, 0): ('dt', 'time step'), (0, 1): ('cycles', 'cycles since t = 654.39'), (0, 2): ('CRECOV n per step', 'failed-C2P recoveries'),
         (1, 0): ('rho_max / rho_c', 'densest debris'), (1, 1): ('(M - M0)/M0', 'total mass'),
         (1, 2): ('M / M*', 'ATMCEIL ledger (ceiling run)'), (2, 0): ('cells', 'ATMCEIL cell counts'),
         (2, 1): ('class minimum rho / threshold', 'debris safety: class minima (offline, per 3D dump)'),
         (2, 2): ('M / M*', 'mass in the band (rho < 100 dfloor)')}
    for (i, j), (yl, ti) in T.items():
        ax[i, j].set_xlabel('t [M]'); ax[i, j].set_ylabel(yl); ax[i, j].set_title(ti, fontsize=10); ax[i, j].grid(alpha=0.3)
        ax[i, j].legend(fontsize=7)
    fig.suptitle('c285 test 4: P10 pilot + atm_ceiling (factor 100, K = K0) vs the P10 pilot, t = 654.4 -> 684.4, CFC on')
    fig.tight_layout(); fig.savefig(f'{out}/c285_p10.png', dpi=75); plt.close(fig)

    R, Cc = rd(f'{out}/classes_P10ref.csv'), rd(f'{out}/classes_P10C.csv')
    ks = [k for k in (R[0] if R else {}) if k.startswith('Mdec_')]
    if R and Cc:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        e = np.array([int(k[5:]) for k in ks]) - 0.5
        for j, tm in enumerate((660.0, 672.0, 684.0)):
            rr = min(R, key=lambda r: abs(float(r['t']) - tm)); cc = min(Cc, key=lambda r: abs(float(r['t']) - tm))
            if abs(float(rr['t']) - float(cc['t'])) > 0.6: continue
            mr = np.array([float(rr[k]) for k in ks]); mc = np.array([float(cc[k]) for k in ks])
            ax[0].semilogy(e, np.maximum(mr, 1e-30), 's-', color=f'C{j}', alpha=0.6, label=f'ref t={float(rr["t"]):.1f}')
            ax[0].semilogy(e, np.maximum(mc, 1e-30), 'o--', color=f'C{j}', label=f'ceil t={float(cc["t"]):.1f}')
            rel = np.where(mr > 0, (mc - mr)/np.where(mr > 0, mr, 1), np.nan)
            ax[1].plot(e, rel, 'o-', color=f'C{j}', label=f't = {float(cc["t"]):.1f}')
        for a in ax:
            a.axvline(np.log10(THR), color='r', ls=':', label='threshold 100 dfloor')
            a.set_xlabel('log10 rho / rho_c (decade centre)'); a.grid(alpha=0.3); a.legend(fontsize=7)
        ax[0].set_ylabel('mass in decade / M*'); ax[0].set_ylim(1e-16, 1.5)
        ax[1].set_ylabel('(ceiling - reference) / reference'); ax[1].set_yscale('symlog', linthresh=1e-6)
        ax[0].set_title('whole-domain mass per density decade'); ax[1].set_title('relative difference per decade')
        fig.suptitle('c285 test 4: the gas above the band, ceiling vs reference at matched 3D dumps')
        fig.tight_layout(); fig.savefig(f'{out}/c285_p10_census.png', dpi=75); plt.close(fig)


def slab(f, var):
    p = bc.read_binary(f); r = np.asarray(p['mb_data']['dens'])[:, 0]
    if var == 'T': r = np.asarray(p['mb_data']['press'])[:, 0]/r
    else: r = r/RC
    return p['time'], np.asarray(p['mb_geometry'])[:, [0, 1, 2, 3]], r


def xbh(d, t):
    p0 = []
    for k in (1, 2, 3):
        for l in open(DECK):
            m = re.match(rf'\s*puncture_x{k}\s*=\s*([-0-9.eE+]+)', l)
            if m: p0.append(float(m.group(1))); break
    fh = glob.glob(f'{d}/{B}.user.hst')[0]; hd = open(fh).readlines()[1]; h = np.loadtxt(fh)
    c = [int(re.search(rf'\[(\d+)\]=frame_disp_x{k}', hd).group(1)) - 1 for k in (1, 2, 3)]
    return np.array(p0) - h[np.argmin(abs(h[:, 0] - t)), c]


def draw(ax, g, r, win, norm, cmap, mesh, title):
    (x0, x1), (y0, y1) = win
    keep = ~((g[:, 1] < x0) | (g[:, 0] > x1) | (g[:, 3] < y0) | (g[:, 2] > y1))
    im = None
    for b in np.nonzero(keep)[0]:
        im = ax.imshow(np.maximum(r[b], 1e-40), origin='lower', extent=tuple(g[b]), cmap=cmap, norm=norm, interpolation='nearest')
    if mesh:
        segs = [[(g[b, 0], g[b, 2]), (g[b, 1], g[b, 2]), (g[b, 1], g[b, 3]), (g[b, 0], g[b, 3]), (g[b, 0], g[b, 2])]
                for b in np.nonzero(keep)[0]]
        ax.add_collection(LineCollection(segs, colors='w', linewidths=mesh, alpha=0.3))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal'); ax.set_title(title, fontsize=9)
    return im


def frame(args):
    i, fr, fc, mesh, out, win = args
    tr, g, rr = slab(fr, 'rho'); tc, gc, rc = slab(fc, 'rho')
    _, _, Tr = slab(fr, 'T'); _, _, Tc = slab(fc, 'T')
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)
    nr, nt = LogNorm(1e-21, 1e-1), LogNorm(1e-7, 1e-1)
    im = draw(ax[0, 0], g, rr, win, nr, 'inferno', mesh, f'reference  rho/rho_c  t = {tr:.1f}')
    draw(ax[0, 1], gc, rc, win, nr, 'inferno', mesh, f'ceiling  rho/rho_c  t = {tc:.1f}')
    ir = {tuple(np.round(x, 9)): k for k, x in enumerate(g)}
    pair = [(ir[tuple(np.round(x, 9))], k) for k, x in enumerate(gc) if tuple(np.round(x, 9)) in ir]
    nmatch = len(pair)
    if nmatch:
        kr, kc = map(np.array, zip(*pair))
        dr = np.abs(rc[kc] - rr[kr])/np.maximum(rr[kr], 1e-300); dr[(rr[kr] < THR) & (rc[kc] < THR)] = np.nan
        imd = draw(ax[0, 2], gc[kc], np.where(np.isfinite(dr), np.maximum(dr, 1e-16), np.nan), win, LogNorm(1e-12, 1), 'viridis',
                   mesh, f'|rho_ceil - rho_ref|/rho_ref, above the band; blocks in both meshes ({nmatch} of {len(gc)} / {len(g)})')
        fig.colorbar(imd, ax=ax[0, 2], shrink=0.8)
    imt = draw(ax[1, 0], g, Tr, win, nt, 'viridis', mesh, f'reference  P/rho  t = {tr:.1f}')
    draw(ax[1, 1], gc, Tc, win, nt, 'viridis', mesh, f'ceiling  P/rho  t = {tc:.1f}')
    for a in list(ax[0]) + list(ax[1, :2]):
        xb = xbh(CEIL, tc); a.plot(xb[0], xb[1], 'o', mfc='none', mec='c', ms=8)
    fig.colorbar(im, ax=ax[0, :2], shrink=0.8).set_label('rho / rho_c (floor 3.1e-21; band < 3.1e-19)')
    fig.colorbar(imt, ax=ax[1, :2], shrink=0.8).set_label('P / rho')
    ax[1, 2].axis('off')
    ax[1, 2].text(0.02, 0.6, 'P10 pilot, t = 654.4 -> 684.4 (CFC on)\nreference: athena_c203, no ceiling\nceiling: same source + atm_ceiling\n'
                  '(factor 100, K = K0, vmax 0.9)\ncyan circle: the BH (puncture - frame_disp)', fontsize=11, transform=ax[1, 2].transAxes)
    fn = f'{out}/frames_p10/f{i:04d}.png'; fig.savefig(fn, dpi=65); plt.close(fig)
    return fn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--mesh', type=float, default=0.3); a = ap.parse_args()
    os.makedirs(f'{a.out}/frames_p10', exist_ok=True)
    figures(a.out)
    fs = sorted(glob.glob(f'{CEIL}/bin/{B}.prim_xy.*.bin'))
    pairs = [(f'{REF}/bin/{os.path.basename(f)}', f) for f in fs if os.path.exists(f'{REF}/bin/{os.path.basename(f)}')]
    win = ((-80.0, 40.0), (-40.0, 80.0))
    with ProcessPoolExecutor(a.nproc) as ex:
        fr = list(ex.map(frame, [(i, r, c, a.mesh, a.out, win) for i, (r, c) in enumerate(pairs)]))
    from PIL import Image
    ims = [Image.open(f).convert('P', palette=Image.ADAPTIVE) for f in fr]
    ims[0].save(f'{a.out}/c285_p10_anim.gif', save_all=True, append_images=ims[1:], duration=400, loop=0)
    ff = sorted(glob.glob('/opt/aurora/*/spack/unified/*/install/linux-x86_64/ffmpeg-*/bin/ffmpeg'))
    if ff:
        subprocess.run([ff[-1], '-y', '-loglevel', 'error', '-framerate', '3', '-i', f'{a.out}/frames_p10/f%04d.png', '-vf',
                        'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', f'{a.out}/c285_p10_anim.mp4'])
    print('wrote', a.out, len(fr), 'frames')


if __name__ == '__main__':
    main()
