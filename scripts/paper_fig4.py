#!/usr/bin/env python3
"""Paper fig. 4 (FIGURES_PLAN.md #4): near the hole, the trace-free energy deposit by shell (sqrt(g)-free coordinate-volume
sums, per M) into stationary gas with self-gravity: xcfc's sigma_tt alone vs kcfc_bg's net (sigma_tt + dsK), log magnitude
(open bar = negative), the net/|sigma_tt| ratio annotated. Beds: hydrostatic L4, Michel L4, Michel L3 (C-182 §3; R-210 stage 2).
The deposit is constant in time from t = 10 on; the per-shell median over t >= 10 is used (the final line of the\nMichel arms (t = 120) is off by ~7%, an end-of-run artifact).
Source: code/results/c175_r210/stage2_shells.txt ('deposit' lines of the X*b_* arms, which carry both sigma_tt and dsK; the
X*x_* arms give xcfc's own sigma_tt; they differ from the b arms' sigma_tt by the solutions' small difference)."""
import re, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import paper_style as ps
R = '/home/tlam/athenak_tde_project/code/results/c175_r210/stage2_shells.txt'
import statistics
raw = {}
for l in open(R):
    m = re.match(r'(X\d[xb]_L\d) t=\s*([\d.]+) deposit (.*)', l)
    if m and float(m.group(2)) >= 10:
        raw.setdefault(m.group(1), []).append(re.findall(r'r\[([\d,]+)\): sig_tt ([-+.\de]+) dsK ([-+.\de]+)', m.group(3)))
dep = {}   # the per-shell median over the lines at t >= 10 (the last line can carry an end-of-run artifact)
for k, v in raw.items():
    dep[k] = (f'10-end ({len(v)} lines)', [(v[0][i][0], statistics.median(float(x[i][1]) for x in v),
                                          statistics.median(float(x[i][2]) for x in v)) for i in range(len(v[0]))])
BEDS = [('X3b_L4', '(a) hydrostatic, L4'), ('X2b_L4', '(b) Michel, L4'), ('X2b_L3', '(c) Michel, L3')]
ps.setup()
fig, axs = ps.plt.subplots(1, 3, figsize=(ps.W2, 2.4), sharey=True)
cx, cb = ps.FAM['valencia'][1], ps.FAM['kcfc_bg'][1]
with open(f'{ps.OUT}/fig4.csv', 'w') as f:
    f.write('bed,t,shell,sigma_tt_xcfc,dsK_kcfc_bg,net_kcfc_bg,net_over_abs_sigma\n')
    for ax, (run, lab) in zip(axs, BEDS):
        t, sh = dep[run]
        shx = dep[run.replace('b_', 'x_')][1]
        for i, ((r, sb, d), (_, s, _)) in enumerate(zip(sh, shx)):
            s, d = float(s), float(d); n = float(sb) + d
            f.write(f'{lab[4:]},{t},"{r}",{s:.4e},{d:.4e},{n:.4e},{n/abs(s):.4f}\n')
            for off, v, c, lb in ((-0.2, s, cx, 'xcfc: $\\sigma_{tt}$'), (0.2, n, cb, 'kcfc_bg: net')):
                ax.bar(i + off, abs(v), width=0.38, color=c if v > 0 else 'white', edgecolor=c, lw=1.0,
                       label=lb if i == 0 else '_')
            ax.text(i + 0.2, abs(n)*1.6, f'{n/abs(s):+.1%}' if abs(n/s) >= 0.001 else f'{n/abs(s):+.0e}', ha='center',
                    fontsize=5.5, rotation=90, va='bottom')
        ax.set_xticks(range(len(sh))); ax.set_xticklabels([r.replace(',', '–') for r, _, _ in sh], fontsize=6.5)
        ax.set_xlabel('shell, $r$ [$M$]'); ax.set_title(lab, fontsize=8, loc='left'); ax.grid(axis='x', visible=False)
axs[0].set_yscale('log'); axs[0].set_ylim(1e-21, 1e-10)
axs[0].set_ylabel('|deposit| per shell')
from matplotlib.patches import Patch
axs[0].legend(handles=[Patch(color=cx, label='xcfc: $\\sigma_{tt}$'), Patch(color=cb, label='kcfc_bg: net'),
                       Patch(facecolor='white', edgecolor=ps.INK2, label='open bar: $<0$')], loc='upper right', fontsize=6.5)
ps.save(fig, 'fig4')
