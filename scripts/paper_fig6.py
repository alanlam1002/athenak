#!/usr/bin/env python3
"""Paper fig. 6 (FIGURES_PLAN.md #6): the D-ent calibration on B.1 (energy only, Bbr, converged metric + kcfc_bg), t = 42 -> 150.
Per class (core >= 0.1 rho_c; outer layers 0.01-0.1 rho_c; the Hill remnant; the stream 1e-6-0.01 rho_c outside r_H), the
mass-weighted median and the 1-99% range of (a) q_en = e_int/tau from the energy solution and (b) q_tr = D eps(K_tr, rho)/tau on
the tracer's adiabat, vs t; eta1 = 3e-4 (and eta2 = 3e-3 in (a)). Floored cells (q ~ 1e-21) fall below the axis.
Source: athenak_run/cfc/c187_eta1/eta1_dumps.csv (C-187; columns q01/q50/q99 = q_en, qa01/qa50/qa99 = q_tr)."""
import csv, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import paper_style as ps
P = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c187_eta1/eta1_dumps.csv'
rows = [r for r in csv.DictReader(open(P)) if r['arm'] == 'Bbr']
CLS = [('core', 'core', '#2a78d6'), ('outer', 'outer layers', '#eb6834'), ('rem', 'remnant', '#1baf7a'),
       ('stream', 'stream', '#6250d6')]
ps.setup()
fig, axs = ps.plt.subplots(1, 2, figsize=(ps.W2, 2.5), sharey=True)
with open(f'{ps.OUT}/fig6.csv', 'w') as f:
    f.write('class,t,qen_01,qen_50,qen_99,qtr_01,qtr_50,qtr_99,floored_share\n')
    for cls, lab, c in CLS:
        seen, R = set(), []
        for r in rows:
            if r['cls'] == cls and round(float(r['t']), 2) not in seen:
                seen.add(round(float(r['t']), 2)); R.append(r)
        t = [float(r['t']) for r in R]
        for r in R:
            f.write(f'{cls},{float(r["t"]):.2f},' + ','.join(r[k] for k in ('q01', 'q50', 'q99', 'qa01', 'qa50', 'qa99', 'fl')) + '\n')
        for ax, p in ((axs[0], 'q'), (axs[1], 'qa')):
            lo, md, hi = ([max(float(r[f'{p}{k}']), 1e-9) for r in R] for k in ('01', '50', '99'))
            ax.fill_between(t, lo, hi, color=c, alpha=0.15, lw=0)
            ax.plot(t, md, color=c, label=lab)
for ax, lab in zip(axs, ['(a) $q_{\\rm en}=e_{\\rm int}/\\tau$ (energy solution)', '(b) $q_{\\rm tr}=D\\varepsilon(K_{\\rm tr},\\rho)/\\tau$ (tracer)']):
    ax.axhline(3e-4, color=ps.INK, lw=0.8, ls='--'); ax.text(44, 3.3e-4, '$\\eta_1$', fontsize=7, va='bottom')
    ps.mark_tp(ax); ax.set_xlabel('$t$ [$M$]'); ax.set_title(lab, fontsize=8, loc='left'); ax.set_xlim(42, 150)
axs[0].axhline(3e-3, color=ps.INK2, lw=0.8, ls=':'); axs[0].text(44, 3.3e-3, '$\\eta_2$', fontsize=7, va='bottom')
axs[0].set_yscale('log'); axs[0].set_ylim(5e-6, 1e-2); axs[0].set_ylabel('$q$ (median, 1–99% band)')
axs[1].legend(loc='lower right', ncol=2, fontsize=6.5)
ps.save(fig, 'fig6')
