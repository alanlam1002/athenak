#!/usr/bin/env python3
"""c276 (R-266, C-236/C-237/C-238): figures of the WENOZ vs PPMX A/B from the c276_wenoz/ana tables, the runs' hst and
out.log, and the reference tables (c249_ana: P18 1/16, L32 1/32, L64 1/64 outflow; c265_ana: P18T / L32T stream transverse).
    c276_plots.py OUTDIR
Writes c276_arm1.png (approach), c276_arm2.png (B.1 D-ent window), c276_arm3.png (stream after pericentre)."""
import sys, re, csv, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

E = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c276_wenoz'
C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
OUT = sys.argv[1]
B = 'cfc_tde_wd_imbh_parabolic_tracker'
TP = 491.1
COL = {'P': ('#1f5fa8', 'ppmx'), 'W': ('#d9731a', 'wenoz')}
REF = {'P18': ('0.45', '--', 'P18 1/16 (old binary)'), 'L32': ('k', '-', '1/32'), 'L64': ('k', ':', '1/64'),
       'P18T': ('0.45', '--', 'P18T 1/16 (C-226)'), 'L32T': ('k', '-', 'L32T 1/32 (C-226)')}


def rd(f):
    return list(csv.DictReader(open(f)))


def col(rows, k):
    out = []
    for r in rows:
        try:
            out.append(float(r[k]))
        except (ValueError, KeyError):
            out.append(np.nan)
    return np.array(out)


def hst(run):
    h = np.loadtxt(f'{E}/{run}/{B}.mhd.hst')
    _, i = np.unique(h[:, 0][::-1], return_index=True)   # later rows win on duplicate times
    return h[::-1][i]


def crecov(run):
    """cumulative CRECOV n / fofc_rescued / m_erased vs time (step lines; cycle -> time from the cycle= lines)."""
    ct, rows = {}, []
    for lg in sorted(glob.glob(f'{E}/{run}/out*.log')):
        for l in open(lg):
            m = re.search(r'cycle=(\d+) time=([0-9.e+-]+)', l)
            if m and not l.startswith(('CRECOV', 'ATMZONE')):
                ct[int(m.group(1))] = float(m.group(2))
            if l.startswith('CRECOV') and 'path=steps' in l:
                d = dict(re.findall(r'(\w+)=([-0-9.e+]+)', l))
                rows.append((int(d['cycle']), float(d['n']), float(d['fofc_rescued']), float(d['m_erased'])))
    rows = sorted(dict((r[0], r) for r in rows).values())
    c = np.array([r[0] for r in rows]); kc = np.array(sorted(ct)); kt = np.array([ct[k] for k in kc])
    t = np.interp(c, kc, kt)
    a = np.array([r[1:] for r in rows])
    return t, np.cumsum(a, axis=0)


def style(ax, xl, yl, log=False):
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=0.3)
    if log:
        ax.set_yscale('log')


# ---------------- arm 2: B.1 D-ent window ----------------
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
cen = rd(f'{E}/ana/census.csv')
for s in 'PW':
    run = f'A2{s}'; c, lab = COL[s]
    h = hst(run)
    ax[0, 0].plot(h[:, 0], h[:, 1], color=c, lw=1, label=lab)
    t, cum = crecov(run)
    ax[0, 1].plot(t, cum[:, 0], color=c, label=lab)
    ax[0, 2].plot(t, cum[:, 2] / 7.47e-5, color=c, label=lab)
    r = [x for x in cen if x['arm'] == run]; tt = col(r, 't'); o = np.argsort(tt)
    ax[1, 0].plot(tt[o], col(r, 'st_M')[o] / 7.47e-5, 'o-', ms=3, color=c, label=lab)
    ax[1, 1].plot(tt[o], col(r, 'st_K')[o], 'o-', ms=3, color=c, label=lab)
    ax[1, 2].plot(tt[o], col(r, 'rem_M')[o] / 7.47e-5, 'o-', ms=3, color=c, label=f'{lab} remnant')
style(ax[0, 0], 't [M]', 'dt (hst)'); ax[0, 0].set_title('time step (WENOZ ~0.7x: hotter near-surface atmosphere)')
style(ax[0, 1], 't [M]', 'cumulative C2P recoveries (CRECOV n)', log=True); ax[0, 1].set_title('failed-C2P recoveries')
style(ax[0, 2], 't [M]', 'cumulative mass erased / M*', log=True); ax[0, 2].set_title('mass erased by recovery')
style(ax[1, 0], 't [M]', 'stream mass / M*'); ax[1, 0].set_title('stream (1e-6..1e-2 rho_c, outside r_H)')
style(ax[1, 1], 't [M]', 'stream mass-weighted K/K0'); ax[1, 1].set_title('stream heat')
style(ax[1, 2], 't [M]', 'self-bound remnant mass / M*', log=True); ax[1, 2].set_title('remnant (census)')
for a in ax.flat:
    a.legend(fontsize=8)
fig.suptitle('c276 arm 2: B.1 D-ent window, t 41.78 -> 150 (8 nodes each; A2W topped up from t = 127.67)')
fig.tight_layout(); fig.savefig(f'{OUT}/c276_arm2.png', dpi=80); plt.close(fig)

# ---------------- arm 1: P18 approach ----------------
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
refs = {'P18': f'{C}/c249_ana/outflow_P18.csv', 'L32': f'{C}/c249_ana/outflow_L32_P18.csv',
        'L64': f'{C}/c249_ana/outflow_L64_P18.csv'}
for k, f in refs.items():
    r = rd(f); t = col(r, 't'); m = t <= 205
    g, ls, lab = REF[k]
    ax[0, 0].plot(t[m], col(r, 'M_gt1.5')[m], ls, color=g, label=lab)
    ax[0, 1].plot(t[m], col(r, 'K15')[m], ls, color=g, label=lab)
for s in 'PW':
    run = f'A1{s}'; c, lab = COL[s]
    r = rd(f'{E}/ana/outflow_{run}.csv'); t = col(r, 't')
    ax[0, 0].plot(t, col(r, 'M_gt1.5'), 'o-', ms=3, color=c, label=lab)
    ax[0, 1].plot(t, col(r, 'K15'), 'o-', ms=3, color=c, label=lab)
    r = rd(f'{E}/ana/coldcore_{run}.csv'); t = col(r, 't')
    ax[0, 2].plot(t, col(r, 'core_KP'), 'o-', ms=3, color=c, label=f'{lab} core r<0.2R*')
    ax[1, 0].plot(t, col(r, 's58_KP'), 'o-', ms=3, color=c, label=f'{lab} shell 0.5-0.8R*')
    h = hst(run)
    ax[1, 1].plot(h[:, 0], h[:, 1], color=c, lw=1, label=lab)
    r = rd(f'{E}/ana/shell_{run}.csv')
    for sh, mk in (('S2', '-'), ('S3', '--')):
        q = [x for x in r if x['shell'] == sh]
        ax[1, 2].plot(col(q, 't'), col(q, 'M'), mk, color=c, label=f'{lab} {sh}')
ax[0, 2].axhline(1.0, color='k', lw=1, label='1/32, 1/64 (C-213)')
style(ax[0, 0], 't [M]', 'M(d > 1.5 R*) / M*', log=True); ax[0, 0].set_title('pre-pericentre surface outflow (numerical wind)')
style(ax[0, 1], 't [M]', 'K/K0 of the gas at d > 1.5 R*', log=True); ax[0, 1].set_title('wind heat')
style(ax[0, 2], 't [M]', 'K_P/K0'); ax[0, 2].set_title('cold core (a dx 1/16 effect)')
style(ax[1, 0], 't [M]', 'K_P/K0'); ax[1, 0].set_title('outer shell 0.5-0.8 R*')
style(ax[1, 1], 't [M]', 'dt (hst)'); ax[1, 1].set_title('time step')
style(ax[1, 2], 't [M]', 'shell mass / M*', log=True); ax[1, 2].set_title('mass in 1-1.5 R* (S2) and > 1.5 R* (S3)')
for a in ax.flat:
    a.legend(fontsize=8)
fig.suptitle(f'c276 arm 1: P18 approach at dx 1/16, t 0 -> 200 (t_p = {TP}; A1W wall-stopped at t = 187.8)')
fig.tight_layout(); fig.savefig(f'{OUT}/c276_arm1.png', dpi=80); plt.close(fig)

# ---------------- arm 3: stream after pericentre ----------------
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
for k in ('P18T', 'L32T'):
    r = [x for x in rd(f'{C}/c265_ana/stream_{k}_transverse.csv') if float(x['R']) == 40.0]
    t = col(r, 't_tp'); m = (t > 110) & (t < 260); g, ls, lab = REF[k]
    for j, q in enumerate(('fwhm_z', 'K', 'rho_pk')):
        ax[0, j].plot(t[m], col(r, q)[m], ls, marker='s', ms=3, color=g, label=lab)
for s in 'PW':
    run = f'A3{s}'; c, lab = COL[s]
    r = [x for x in rd(f'{E}/ana/stream_{run}_transverse.csv') if float(x['R']) == 40.0]
    t = col(r, 't_tp')
    for j, q in enumerate(('fwhm_z', 'K', 'rho_pk')):
        ax[0, j].plot(t, col(r, q), 'o-', ms=4, color=c, label=lab)
    r = rd(f'{E}/ana/stream_{run}_census.csv'); t = col(r, 't_tp')
    ax[1, 0].plot(t, col(r, 'M_stream'), 'o-', ms=3, color=c, label=lab)
    ax[1, 1].plot(t, col(r, 'rho_max'), 'o-', ms=3, color=c, label=lab)
    ax[1, 2].plot(t, col(r, 'M_H'), 'o-', ms=3, color=c, label=lab)
    h = hst(run)
for j in range(3):
    ax[0, j].axvline(217, color='0.6', lw=0.8)
style(ax[0, 0], 't - t_p [M]', 'FWHM_z [M]'); ax[0, 0].set_title('stream vertical width at R = 40')
style(ax[0, 1], 't - t_p [M]', 'K/K0', log=True); ax[0, 1].set_title('stream heat at R = 40')
style(ax[0, 2], 't - t_p [M]', 'peak rho', log=True); ax[0, 2].set_title('stream peak density at R = 40')
style(ax[1, 0], 't - t_p [M]', 'stream mass / M*'); ax[1, 0].set_title('stream census')
style(ax[1, 1], 't - t_p [M]', 'rho_max / rho_c'); ax[1, 1].set_title('remnant central density')
style(ax[1, 2], 't - t_p [M]', 'M_H / M*'); ax[1, 2].set_title('Hill-sphere mass')
for a in ax.flat:
    a.legend(fontsize=8)
fig.suptitle('c276 arm 3: P18T stream after pericentre, t_p + 120 -> 245 (athena_fr; grey line t_p + 217 = C-226 crossing)')
fig.tight_layout(); fig.savefig(f'{OUT}/c276_arm3.png', dpi=80); plt.close(fig)
print('wrote', OUT)
