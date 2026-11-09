#!/usr/bin/env python3
"""Paper fig. 2 (FIGURES_PLAN.md #2): B.1, Valencia arm (VB10, converged), the XCFC trace-free energy deposit
-1/2 psi^4 S^ij T_ij in the band rho >= 0.01 rho_c split into T_self, T_cross and B, the total, and the trace part R S/3,
per 6 M window. (a) as rates per unit dense internal energy (R-242 item 1): deposit / E_int [1/M], E_int = sum of sqrt(g) P/(Gamma-1) dV
over rho >= 0.01 rho_c, the mean of the window's two edge dumps (VB10 prim_3d every 6 M; scripts/paper_fig2_eint.py); (b) the raw
sqrt(g) dV sums' magnitudes (log), with B.
Sources: code/results/c169/logs.txt (VB10 'TRD per window' table, C-167 §3); code/results/paper_figs/fig2_eint.csv."""
import csv, re, sys
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
EI = [(float(r['t']), float(r['Eint_sqrtg'])) for r in csv.DictReader(open(f'{ps.OUT}/fig2_eint.csv'))]
def eint(t0, t1):   # the mean of the dumps nearest the window's edges
    near = lambda tt: min(EI, key=lambda e: abs(e[0] - tt))[1]
    return 0.5*(near(t0) + near(t1))
ew = [eint(r[0] - 3, r[0] + 3) for r in rows]
with open(f'{ps.OUT}/fig2.csv', 'w') as f:
    f.write('t_mid,trace,TF_total,TF_self,TF_cross,TF_B,Eint_dense,trace_per_Eint,TF_total_per_Eint,TF_self_per_Eint,TF_cross_per_Eint,TF_B_per_Eint\n')
    for r, e in zip(rows, ew): f.write(','.join(f'{x:.6g}' for x in (*r, e, *[v/e for v in r[1:]])) + '\n')
t = [r[0] for r in rows]
COMP = [('total (trace-free)', 2, ps.INK, '-'), ('$T_{\\rm self}$', 3, '#2a78d6', (0, (5, 1.5))),
        ('$T_{\\rm cross}$', 4, '#eb6834', (0, (1, 1.2))), ('trace, $RS/3$', 1, '#1baf7a', (0, (6, 1.5, 1, 1.5))),
        ('$B$', 5, '#6250d6', '-')]
ps.setup()
fig, axs = ps.plt.subplots(1, 2, figsize=(ps.W2, 2.4), sharex=True)
for lab, i, c, ls in COMP:
    y = [r[i] for r in rows]
    if i != 5:
        axs[0].plot(t, [1e3*v/e for v, e in zip(y, ew)], color=c, ls=ls, label=lab, lw=1.6 if i == 2 else 1.3)
    axs[1].plot(t, [abs(v) for v in y], color=c, ls=ls, label=lab if i == 5 else '_')
axs[0].axhline(0, color=ps.INK2, lw=0.5)
axs[0].set_ylabel('deposit / $E_{\\rm int}$ [$10^{-3}\\,M^{-1}$]')
axs[1].set_yscale('log'); axs[1].set_ylabel('|deposit|, $\\sqrt{g}\\,dV$ sums')
for ax, lab in zip(axs, ['(a) per unit dense internal energy', '(b) raw magnitudes']):
    ps.mark_tp(ax); ax.set_xlabel('$t$ [$M$]'); ax.set_title(lab, fontsize=8, loc='left'); ax.set_xlim(36, 300)
axs[0].legend(loc='lower right'); axs[1].legend(loc='center right')
ps.save(fig, 'fig2')
