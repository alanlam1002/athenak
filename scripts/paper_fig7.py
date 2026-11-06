#!/usr/bin/env python3
"""Paper fig. 7 (FIGURES_PLAN.md #7): the shock regressions (1D, CPU strict; C-191 §3). The captured fraction of the
Rankine-Hugoniot entropy f_RH = (K - 1)/(K_RH - 1) in the shocked slab vs the Mach number, cold boosted colliding flows at
v_b = 0.3, for upstream gas (a) above eta1 (P0 1.6e-5, q_ad ~ 5e-4) and (b) below eta1 (P0 3.2e-6, q_ad ~ 1e-4): energy only (DE
off), the production DE (legacy guard) and D-ent. (c) the C-107 moving shocks (warm, P 0.01): f_RH vs the boost v_b at M = 2
(the legacy guard's boost dependence 1.03 -> 1.25; D-ent 0.996-1.001). Source: athenak_run/cfc/c190_dent/cpu/metrics_dent.txt."""
import re, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import paper_style as ps
P = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c190_dent/cpu/metrics_dent.txt'
f = {}
for l in open(P):
    m = re.match(r'(\S+)\s+[-+.\de]+ \|.*\|\s*([-.\d]+) /\s*([-.\d]+) /\s*([-.\d]+) \|', l)
    if m: f[m.group(1)] = tuple(float(m.group(i)) for i in (2, 3, 4))
SCH = [(0, 'energy_only'), (1, 'prod_de'), (2, 'dent')]
MACH = [1.5, 3, 6, 20]
ps.setup()
fig, axs = ps.plt.subplots(1, 3, figsize=(ps.W2, 2.4), gridspec_kw={'width_ratios': [1, 1, 0.8]})
out = open(f'{ps.OUT}/fig7.csv', 'w'); out.write('set,case,x,f_RH_energy_only,f_RH_production_DE,f_RH_D_ent\n')
for ax, p, lab in ((axs[0], '1.6e-05', '(a) upstream above $\\eta_1$'), (axs[1], '3.2e-06', '(b) upstream below $\\eta_1$')):
    for i, fam in SCH:
        y = [f[f'cold_P{p}_M{M}_vb0.3'][i] for M in MACH]
        ps.line(ax, fam, MACH, y, marker='o', ms=3.5, label=None if p == '1.6e-05' else '_')
    for M in MACH:
        v = f[f'cold_P{p}_M{M}_vb0.3']; out.write(f'cold_vb0.3_P{p},cold_P{p}_M{M}_vb0.3,{M},{v[0]},{v[1]},{v[2]}\n')
    ax.axhline(1, color=ps.INK2, lw=0.6); ax.set_xscale('log'); ax.set_xticks(MACH); ax.set_xticklabels(['1.5', '3', '6', '20'])
    ax.set_xlabel('Mach number'); ax.set_title(lab, fontsize=8, loc='left'); ax.set_ylim(-0.3, 4.0)
axs[0].set_ylabel('$f_{\\rm RH}$'); axs[0].legend(loc='upper right', fontsize=6.5)
ax = axs[2]; vb = [0.0, 0.1, 0.2, 0.3]
for i, fam in SCH:
    ps.line(ax, fam, vb, [f[f'c107_M2_vb{v}'][i] for v in vb], marker='o', ms=3.5, label='_')
for v in vb:
    x = f[f'c107_M2_vb{v}']; out.write(f'c107_M2,c107_M2_vb{v},{v},{x[0]},{x[1]},{x[2]}\n')
ax.axhline(1, color=ps.INK2, lw=0.6); ax.set_xlabel('boost $v_b$'); ax.set_xticks(vb)
ax.set_title('(c) C-107, $M=2$', fontsize=8, loc='left'); ax.set_ylim(0.95, 1.3)
ps.save(fig, 'fig7')
