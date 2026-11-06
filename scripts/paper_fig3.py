#!/usr/bin/env python3
"""Paper fig. 3 (FIGURES_PLAN.md #3): the isolated moving star (ISO_A, v = 0.4, Valencia, DE off).
(a) the dense drift <K/K0> - 1 (rho >= 0.01 rho_c) at t = 60 vs the elliptic convergence setting (log |drift|; open marker =
negative), annotated with the share of psi solves that do no V-cycle ("skip"); (b) K_eff - Valencia (percentage points) vs the
boost v at converged solves (rms 1e-8 + one V-cycle), t = 60 and 120, both bands.
Sources: code/results/c161/c161_tol.csv (C-160 §2); code/results/c163/ana_dtscan.csv (C-163 speed scan).
Notes: max 1e-7 (d2) reached t = 47.9 only (its last dump is used); rms 1e-10's skip share (94%) is from the logged rerun to
t = 17 (C-160 §2); rms 1e-8 has no V-cycle log."""
import csv, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import paper_style as ps
R = '/home/tlam/athenak_tde_project/code/results'
tol = {r['arm']: r for r in csv.DictReader(open(f'{R}/c161/c161_tol.csv'))}
SET = [('e8', 'rms\n$10^{-8}$'), ('d1', 'max\n$10^{-6}$'), ('b10', 'rms\n$10^{-10}$'), ('d2', 'max\n$10^{-7}$ (t=48)'),
       ('a', 'rms $10^{-8}$\n+1 V-cycle'), ('c', 'rms\n$10^{-11}$')]
SKIP = {'b10': 0.94}
sc = {r['arm']: r for r in csv.DictReader(open(f'{R}/c163/ana_dtscan.csv'))}
ps.setup()
fig, axs = ps.plt.subplots(1, 2, figsize=(ps.W2, 2.5), gridspec_kw={'width_ratios': [1.25, 1]})
ax = axs[0]
c = ps.FAM['valencia'][1]
ticks = []
with open(f'{ps.OUT}/fig3a.csv', 'w') as f:
    f.write('setting,arm,t,drift_d01_pct,psi_skip_share,vcycles_per_cycle,wall_per_cycle_s\n')
    for i, (a, lab) in enumerate(SET):
        r = tol[a]; d = 100*float(r['d01_t60']); sk = r['psi_skip'] or (str(SKIP[a]) if a in SKIP else '')
        f.write(f'"{lab.replace(chr(10), " ")}",{a},{float(r["t_dump60"]):.2f},{d:.5f},{sk},{r["vc_per_cycle"]},{r["wall_per_cycle"]}\n')
        ax.bar(i, abs(d), color=c if d > 0 else 'white', edgecolor=c, lw=1.0, width=0.6)
        ax.text(i, abs(d)*1.35, f'{d:+.3f}%' if abs(d) < 0.1 else f'{d:+.2f}%', ha='center', fontsize=6.5)
        ticks.append(lab + '\nskip ' + (f'{100*float(sk):.0f}%' if sk else 'n/a'))
ax.set_yscale('log'); ax.set_ylim(1e-3, 60)
ax.set_xticks(range(len(SET))); ax.set_xticklabels(ticks, fontsize=6)
ax.set_ylabel('$|\\langle K/K_0\\rangle-1|$, $t=60$ [%]'); ax.grid(axis='x', visible=False)
ax.set_title('(a) dense drift ($\\rho\\geq0.01\\rho_c$) vs convergence', fontsize=8, loc='left')
ax = axs[1]
with open(f'{ps.OUT}/fig3b.csv', 'w') as f:
    f.write('v,band,t,keff_minus_valencia_pp\n')
    for band, blab, mk in (('d01', '$\\geq0.01\\rho_c$', 'o'), ('d1', '$\\geq0.1\\rho_c$', 's')):
        for tt, ls in (('60', '-'), ('120', '--')):
            v = [0.1, 0.2, 0.4]
            dk = [100*(float(sc[f'K{n}'][f'{band}_t{tt}']) - float(sc[f'V{n}'][f'{band}_t{tt}'])) for n in (1, 2, 4)]
            for a, b in zip(v, dk): f.write(f'{a},{band},{tt},{b:.5f}\n')
            ax.plot(v, dk, marker=mk, ms=4, ls=ls, color=ps.FAM['keff'][1],
                    mfc='white' if tt == '120' else None, label=f'{blab}, $t={tt}$')
ax.axhline(0, color=ps.INK2, lw=0.5)
ax.set_xlabel('boost $v$'); ax.set_ylabel('$K_{\\rm eff}$ $-$ Valencia [points]')
ax.set_title('(b) converged solves', fontsize=8, loc='left'); ax.legend(loc='lower left'); ax.set_xticks([0.1, 0.2, 0.4])
ps.save(fig, 'fig3')
