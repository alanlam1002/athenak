#!/usr/bin/env python3
"""Paper fig. 5 (FIGURES_PLAN.md #5): the dual-energy comparison on B.1 (converged metric, kcfc_bg), t = 41.78 -> 150.
(a) dense core (>= 0.1 rho_c) and (b) outer layers (0.01-0.1 rho_c) mass-weighted K/K0 - 1 (%); (c) the floored share of the
stream mass (%, log; zero values omitted). Arms: energy only (Bbr: Bb10 rerun with the regrid max norm), the production DE (Bd10),
D-ent (De; De2 = D-ent with rho_sw 1e-6 rho_c, the production setting, thin), the internal-energy partner D-int (Di).
Outer layers = (M01 K01 - M1 K1)/(M01 - M1) from the census bands. Sources: code/results/c186_kfta/census.csv (Bbr),
c171_b1kbg/census.csv (Bd10), athenak_run/cfc/c192_b1de/ana/census.csv (De, Di), c194_de2/ana/census.csv (De2)."""
import csv, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import paper_style as ps
R = '/home/tlam/athenak_tde_project/code/results'; C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
SRC = {'Bbr': f'{R}/c186_kfta/census.csv', 'Bd10': f'{R}/c171_b1kbg/census.csv', 'De': f'{C}/c192_b1de/ana/census.csv',
       'De2': f'{C}/c194_de2/ana/census.csv', 'Di': f'{C}/c192_b1de/ana/census.csv'}
ARMS = [('Bbr', 'energy_only', {}), ('Bd10', 'prod_de', {}), ('De', 'dent', {}),
        ('De2', 'dent', {'ls': '-', 'lw': 0.7, 'label': 'D-ent, $\\rho_{\\rm sw}=10^{-6}\\rho_c$'}), ('Di', 'dint', {})]
ps.setup()
fig, axs = ps.plt.subplots(1, 3, figsize=(ps.W2, 2.4), sharex=True)
out = open(f'{ps.OUT}/fig5.csv', 'w'); out.write('arm,family,t,K_core_pct,K_outer_pct,stream_floored_pct,rhomax\n')
for arm, fam, kw in ARMS:
    seen, T, KC, KO, FL = set(), [], [], [], []
    for r in csv.DictReader(open(SRC[arm])):
        t = float(r['t'])
        if r['arm'] != arm or t > 150.01 or round(t, 2) in seen: continue
        seen.add(round(t, 2))
        M01, K01, M1, K1 = (float(r[k]) for k in ('d01_M', 'd01_K', 'd1_M', 'd1_K'))
        T.append(t); KC.append(100*K1); KO.append(100*(M01*K01 - M1*K1)/(M01 - M1)); FL.append(100*float(r['st_fl']))
        out.write(f'{arm},{fam},{t:.2f},{KC[-1]:.4f},{KO[-1]:.4f},{FL[-1]:.4f},{float(r["rhomax"]):.4f}\n')
    ps.line(axs[0], fam, T, KC, **kw); ps.line(axs[1], fam, T, KO, **dict(kw, label='_'))
    ps.line(axs[2], fam, [a for a, b in zip(T, FL) if b > 0], [b for b in FL if b > 0], **dict(kw, label='_'))
for ax, lab in zip(axs, ['(a) core, $\\rho\\geq0.1\\rho_c$', '(b) outer layers, $0.01$–$0.1\\rho_c$', '(c) stream, floored mass']):
    ps.mark_tp(ax); ax.set_xlabel('$t$ [$M$]'); ax.set_title(lab, fontsize=8, loc='left'); ax.set_xlim(41.78, 150)
for ax in axs[:2]: ax.axhline(0, color=ps.INK2, lw=0.5)
axs[0].set_ylabel('$\\langle K/K_0\\rangle - 1$ [%]'); axs[2].set_ylabel('floored [% of stream mass]')
axs[2].set_yscale('log'); fig.subplots_adjust(wspace=0.38)
axs[0].legend(loc='upper left', fontsize=6)
ps.save(fig, 'fig5')
