#!/usr/bin/env python3
"""c169 (user request 2026-11-02): analysis figure for R-202's B.1 job (c161_b1, t161b.sh): VB10 (Valencia) and Be10 (keff),
both from B1N_cell's t = 41.78 restart at rms 1e-8 + mg_min_vcycles 1, DE off, trace_diag with the T_self / T_cross / B split;
C-157's plain-1e-8 arms (c160_tsk VBf, Be; 41.78 -> 120) as faint references in the dense-band panels.
Panels:
  (a) dense <K/K0 - 1> (rho dV-weighted; rho >= 0.01 rho_c solid, >= 0.1 rho_c dashed), all cells      -- c160_dense_floor.py CSV
  (b) the same excluding the cells at the temperature floor
  (c) floored share of the >= 0.01 rho_c mass
  (d) rho_max / rho_c
  (e) R = 6 d_t ln psi - D_k beta'^k [M^-1], step means (stage values dtm-weighted, steps with an invalid stage dropped),
      band b1 (rho >= 0.1 rho_c) and b0 (>= 0.5 rho_c)                                                 -- TRD lines
  (f) the trace-free deposit -1/2 psi^4 S^ij (T_self, T_cross, B)_ij integrated over band b1 (sqrt(g) dV), per step
  (g) the same parts by radius about the hole (TRR bins, rho >= 0.01 rho_c), VB10, averaged over three windows
  (h) V-cycles per solve (MGV lines; psi and beta: per-cycle mean over the three stages, then a 25-cycle running mean)
  (i) psi's initial defect (rms) per solve: how far the source moved since the previous stage's solve
Times in M; t_p = 135.9 (periapsis) marked. Norms: |.|_coord.
    c169_b1_plots.py OUT.png DENSE.csv RUNROOT"""
import csv
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

TP = 135.9
ARMS = [('VB10', 'VB10 (Valencia)', '#2a78d6'), ('Be10', 'Be10 (keff)', '#eb6834'), ('Bc10', 'Bc10 (kcfc)', '#1baf7a')]
REFS = [('VBf_1e8', 'VBf (Valencia, 1e-8, C-157)', '#2a78d6'), ('Be_1e8', 'Be (keff, 1e-8, C-157)', '#eb6834'),
        ('Bc_1e8', 'Bc (kcfc, 1e-8, C-157)', '#1baf7a')]
NB = 16     # TRD sums per band (wip/trace-split): w, wR, wR^2, w|T|, w|T|^2, wR|T|, w6dtlnpsi, wDb, g sig_tt, g RS/3,
            # w|Ts|, w|Tc|, w|B|, g dep_self, g dep_cross, g dep_B
NRB, NRC = 10, 8   # TRR: bins x (w, wR, w|Ts|, w|Tc|, w|B|, g dep_self, g dep_cross, g dep_B)


def dense(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r['arm'], []).append({k: float(v) for k, v in r.items() if k != 'arm'})
    return {k: sorted(v, key=lambda x: x['t']) for k, v in out.items()}


def trd(log):
    """per step: t, R b0/b1, deposits b1 (self, cross, B); stages dtm-weighted, steps with an invalid stage dropped"""
    st = {}
    for l in open(log):
        if not l.startswith('TRD '):
            continue
        h = dict(re.findall(r'(\w+)=(\S+)', l.split(' b0=')[0]))
        b = [np.array([float(x) for x in s.split(',')]) for s in re.findall(r' b\d=(\S+)', l)]
        st.setdefault(int(h['cycle']), []).append((float(h['time']), float(h['dtm']), int(h['valid']), b))
    rows = []
    for c in sorted(st):
        s = st[c]
        if len(s) != 3 or not all(x[2] for x in s):
            continue
        wd = np.array([x[1] for x in s]); wd = wd/wd.sum()
        avg = [sum(wd[i]*s[i][3][bb] for i in range(3)) for bb in range(3)]
        rows.append((s[0][0], avg[0][1]/avg[0][0], avg[1][1]/avg[1][0], avg[1][13], avg[1][14], avg[1][15]))
    return np.array(rows)


def trr(log):
    out = []
    for l in open(log):
        if l.startswith('TRR '):
            t = float(re.search(r'time=(\S+)', l).group(1))
            a = np.array([[float(x) for x in s.split(',')] for s in re.findall(r' r\d=(\S+)', l)])
            out.append((t, a))
    return out


def mgv(log):
    cyc = {}
    for l in open(log):
        if not l.startswith('MGV '):
            continue
        c = int(re.search(r'cycle= (\d+)', l).group(1)); t = float(re.search(r'time= (\S+)', l).group(1))
        v = {k: (int(n), float(d0)) for k, n, d0 in re.findall(r' (\w+)= (-?\d+),(\S+?),', l)}
        cyc.setdefault(c, [t, []])[1].append(v)
    rows = []
    for c in sorted(cyc):
        t, vs = cyc[c]
        if c <= 500 or len(vs) < 1:
            continue
        rows.append((t, np.mean([v['psi'][0] for v in vs]), np.mean([v['beta'][0] for v in vs]),
                     max(v['psi'][0] for v in vs), np.mean([v['psi'][1] for v in vs])))
    return np.array(rows)


def run_avg(y, n=25):
    """centred running mean (the window shrinks at the ends; no zero padding)"""
    h = n//2
    return np.array([np.mean(y[max(0, i - h):i + h + 1]) for i in range(len(y))])


def run_med(y, n=25):
    """centred running median (the window shrinks at the ends; no zero padding)"""
    h = n//2
    return np.array([np.median(y[max(0, i - h):i + h + 1]) for i in range(len(y))])


def main():
    out, dpath, root = sys.argv[1], sys.argv[2], sys.argv[3]
    D = dense(dpath)
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    ax = ax.ravel()
    for lab, nm, col in REFS:
        if lab in D:
            t = np.array([r['t'] for r in D[lab]])
            ax[0].plot(t, 100*np.array([r['d01_mean'] for r in D[lab]]), color=col, alpha=0.35, lw=1, label=nm)
            ax[1].plot(t, 100*np.array([r['d01_mean_unfl'] for r in D[lab]]), color=col, alpha=0.35, lw=1)
            ax[2].plot(t, 100*np.array([r['d01_floor_frac'] for r in D[lab]]), color=col, alpha=0.35, lw=1)
            ax[3].plot(t, [r['rhomax'] for r in D[lab]], color=col, alpha=0.35, lw=1)
    for lab, nm, col in ARMS:
        if lab in D:
            t = np.array([r['t'] for r in D[lab]])
            ax[0].plot(t, 100*np.array([r['d01_mean'] for r in D[lab]]), color=col, lw=2, label=nm + ' >= 0.01 rho_c')
            ax[0].plot(t, 100*np.array([r['d1_mean'] for r in D[lab]]), color=col, lw=1.5, ls='--', label=nm + ' >= 0.1 rho_c')
            ax[1].plot(t, 100*np.array([r['d01_mean_unfl'] for r in D[lab]]), color=col, lw=2)
            ax[1].plot(t, 100*np.array([r['d1_mean_unfl'] for r in D[lab]]), color=col, lw=1.5, ls='--')
            ax[2].plot(t, 100*np.array([r['d01_floor_frac'] for r in D[lab]]), color=col, lw=2)
            ax[3].plot(t, [r['rhomax'] for r in D[lab]], color=col, lw=2)
        log = f'{root}/{lab}/out.log'
        T = trd(log)
        if len(T):
            ax[4].plot(T[:, 0], T[:, 2], color=col, lw=0.4, alpha=0.25)
            ax[4].plot(T[:, 0], run_med(T[:, 2]), color=col, lw=1.8, label=nm + ' b1 (>= 0.1 rho_c), 25-step median')
            ax[4].plot(T[:, 0], run_med(T[:, 1]), color=col, lw=1, ls=':', label=nm + ' b0 (>= 0.5 rho_c), median')
            for k, ls, pn in ((3, '-', 'T_self'), (4, '--', 'T_cross'), (5, ':', 'B')):
                ax[5].plot(T[:, 0], T[:, k], color=col, lw=1, ls=ls, label=f'{nm} {pn}')
        M = mgv(log)
        if len(M):
            k = 25
            ax[7].plot(M[:, 0], run_avg(M[:, 1], k), color=col, lw=1.5, label=nm + ' psi')
            ax[7].plot(M[:, 0], run_avg(M[:, 2], k), color=col, lw=1, ls='--',
                       label=nm + ' beta')
            ax[7].plot(M[:, 0], M[:, 3], color=col, lw=0.3, alpha=0.4, label=nm + ' psi, max per cycle')
            ax[8].semilogy(M[:, 0], M[:, 4], color=col, lw=0.3, alpha=0.3)
            ax[8].semilogy(M[:, 0], run_med(M[:, 4]), color=col, lw=1.8, label=nm + ' (25-cycle median)')
    # (g) radial deposit profile, VB10
    R = trr(f'{root}/VB10/out.log')
    if R:
        edges = np.exp(np.log(2.0) + np.arange(NRB - 1)*np.log(100.0)/(NRB - 2))
        rc = np.r_[1.0, np.sqrt(edges[:-1]*edges[1:]), 300.0]
        for (lo, hi), ls in (((60, 110), ':'), ((110, 160), '-'), ((160, 300), '--')):
            sel = [a for t, a in R if lo <= t < hi]
            if not sel:
                continue
            a = np.mean(sel, axis=0)
            for k, col, pn in ((5, '#1baf7a', 'T_self'), (6, '#8a5cd6', 'T_cross'), (7, '#d64a4a', 'B')):
                y = np.abs(a[:, k]); y[y == 0] = np.nan
                ax[6].plot(rc, y, color=col, ls=ls, marker='o', ms=3,
                           label=f'{pn} t in [{lo},{hi})')
        ax[6].set_xscale('log'); ax[6].set_yscale('log')
    ttl = ['(a) dense <K/K0 - 1> [%], all cells', '(b) dense <K/K0 - 1> [%], floored cells excluded',
           '(c) floored share of the >= 0.01 rho_c mass [%]', '(d) rho_max / rho_c',
           '(e) R = 6 d_t ln psi - D_k beta\'^k [1/M] (faint: every step)',
           '(f) trace-free deposit over band b1 (sqrt(g) dV; symlog, lin < 1e-22)',
           '(g) VB10: |deposit| by r about the hole [M] (rho >= 0.01 rho_c)', '(h) V-cycles per solve (25-cycle mean)',
           '(i) psi initial defect (rms) per solve (faint: every cycle)']
    for i, a in enumerate(ax):
        a.set_title(ttl[i], fontsize=10)
        if i != 6:
            a.axvline(TP, color='0.5', lw=0.8, ls='--')
            a.set_xlabel('t [M]')
        a.grid(alpha=0.25)
    ax[6].set_xlabel('r = |x - x_BH| [M]')
    for i in (4, 5):
        ax[i].set_yscale('symlog', linthresh=1e-10 if i == 4 else 1e-22)
    for i in (0, 4, 5, 6, 7, 8):
        ax[i].legend(fontsize=6.5, loc='best')
    fig.suptitle('B.1 at converged solves (rms 1e-8 + 1 forced V-cycle), t = 41.78 -> 300, DE off; dashed grey = periapsis '
                 f't_p = {TP}', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    print('wrote', out)


if __name__ == '__main__':
    main()
