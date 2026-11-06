#!/usr/bin/env python3
"""Paper fig. 1 (FIGURES_PLAN.md #1): B.1, mass-weighted K/K0 - 1 (%) vs t, converged metric, DE off.
(a) dense core (>= 0.1 rho_c); (b) >= 0.01 rho_c. Valencia (VB10), K_eff (Be10), kcfc (Bc10), kcfc_bg (Bb10); Valencia at the
plain rms 1e-8 threshold (VBf, to t = 120) dashed-thin. Sources: code/results/c169/dense_floor.csv (C-167/C-169),
c171_b1kbg/dense_floor.csv (Bb10). Columns d1_mean / d01_mean (all cells, floored included = the text's table)."""
import csv, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import paper_style as ps
R = '/home/tlam/athenak_tde_project/code/results'
rows = list(csv.DictReader(open(f'{R}/c169/dense_floor.csv'))) + list(csv.DictReader(open(f'{R}/c171_b1kbg/dense_floor.csv')))
ARMS = [('VB10', 'valencia', {}), ('Be10', 'keff', {}), ('Bc10', 'kcfc', {}), ('Bb10', 'kcfc_bg', {}),
        ('VBf', 'valencia', {'ls': '--', 'lw': 0.8, 'label': 'Valencia, plain rms $10^{-8}$'})]
out = open(f'{ps.OUT}/fig1.csv', 'w')
out.write('arm,family,t,K_core_pct,K_ge001_pct\n')
ps.setup()
fig, axs = ps.plt.subplots(1, 2, figsize=(ps.W2, 2.4), sharex=True)
for arm, fam, kw in ARMS:
    r = [x for x in rows if x['arm'] == arm]
    t = [float(x['t']) for x in r]; k1 = [100*float(x['d1_mean']) for x in r]; k01 = [100*float(x['d01_mean']) for x in r]
    for a, b, c in zip(t, k1, k01):
        out.write(f'{arm},{fam},{a:.2f},{b:.4f},{c:.4f}\n')
    ps.line(axs[0], fam, t, k1, **kw); ps.line(axs[1], fam, t, k01, **dict(kw, label='_'))
for ax, lab in zip(axs, ['(a) dense core, $\\rho\\geq0.1\\rho_c$', '(b) $\\rho\\geq0.01\\rho_c$']):
    ps.mark_tp(ax); ax.axhline(0, color=ps.INK2, lw=0.5)
    ax.set_xlabel('$t$ [$M$]'); ax.set_title(lab, fontsize=8, loc='left')
    ax.set_xlim(41.78, 300)
axs[0].set_ylabel('$\\langle K/K_0\\rangle - 1$ [%]')
axs[0].legend(loc='lower left')
ps.save(fig, 'fig1')
