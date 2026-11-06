#!/usr/bin/env python3
"""Paper fig. 2 (FIGURES_PLAN.md #2): B.1, Valencia arm (VB10, converged), the XCFC trace-free energy deposit
-1/2 psi^4 S^ij T_ij in the band rho >= 0.01 rho_c split into T_self, T_cross and B, the total, and the trace part R S/3,
per 6 M window (sqrt(g) dV sums, code units). (a) linear; (b) log of the magnitudes, with B.
Source: code/results/c169/logs.txt (VB10 'TRD per window' table, C-167 §3).
NOTE: the plan asks for rates per unit dense internal energy; the TRD table has no e_int, so the raw sums are shown."""
import re, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import paper_style as ps
R = '/home/tlam/athenak_tde_project/code/results'
lines = open(f'{R}/c169/logs.txt').read().split('===== ')[1].splitlines()   # the VB10 block
rows = []
on = False
for l in lines:
    if l.startswith('TRD per window'): on = True; continue
    if on:
        m = re.match(r'\s*(\d+)-\s*(\d+) \|([^|]+)\|([^|]+)\|', l)
        if not m: break
        dep = [float(x) for x in m.group(4).split()]
        rows.append(((int(m.group(1)) + int(m.group(2)))/2, *dep))   # tc, trace, TF_total, self, cross, B
with open(f'{ps.OUT}/fig2.csv', 'w') as f:
    f.write('t_mid,trace,TF_total,TF_self,TF_cross,TF_B\n')
    for r in rows: f.write(','.join(f'{x:.6g}' for x in r) + '\n')
t = [r[0] for r in rows]
COMP = [('total (trace-free)', 2, ps.INK, '-'), ('$T_{\\rm self}$', 3, '#2a78d6', (0, (5, 1.5))),
        ('$T_{\\rm cross}$', 4, '#eb6834', (0, (1, 1.2))), ('trace, $RS/3$', 1, '#1baf7a', (0, (6, 1.5, 1, 1.5))),
        ('$B$', 5, '#6250d6', '-')]
ps.setup()
fig, axs = ps.plt.subplots(1, 2, figsize=(ps.W2, 2.4), sharex=True)
for lab, i, c, ls in COMP:
    y = [r[i] for r in rows]
    if i != 5:
        axs[0].plot(t, [v*1e12 for v in y], color=c, ls=ls, label=lab, lw=1.6 if i == 2 else 1.3)
    axs[1].plot(t, [abs(v) for v in y], color=c, ls=ls, label=lab if i == 5 else '_')
axs[0].axhline(0, color=ps.INK2, lw=0.5)
axs[0].set_ylabel('deposit, $\\rho\\geq0.01\\rho_c$ [$10^{-12}$]')
axs[1].set_yscale('log'); axs[1].set_ylabel('|deposit|')
for ax, lab in zip(axs, ['(a) Valencia, converged', '(b) magnitudes']):
    ps.mark_tp(ax); ax.set_xlabel('$t$ [$M$]'); ax.set_title(lab, fontsize=8, loc='left'); ax.set_xlim(36, 300)
axs[0].legend(loc='lower right'); axs[1].legend(loc='center right')
ps.save(fig, 'fig2')
