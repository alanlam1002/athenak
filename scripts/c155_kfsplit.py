#!/usr/bin/env python3
"""c155 (R-177 item 2 and its addenda 1-2, user-approved 2026-10-30): the B1K8/B1K10 KFSPLIT lines (wip/killing-dtg,
dyn_grmhd_killing.cpp ReportKillingStage), one per stage, over the masked set (rho >= killing_full_split_rho, the dense
band; all fields densitized, coordinate volume):
  item 2:     bg_dense, res_dense = sum sqrt(g) s_T dV, the background (analytic rate) and residual (chord) parts;
              rms_/max_ d_t dpsi, d_t d(alpha psi), |d_t beta_res|_coord; nz_frac (cells with a nonzero residual chord)
  addendum 1: vdS_src, vdS_flux = sum v^k dS_k/dt dV (Eulerian v^k = u^k/W) from Valencia's momentum source and from
              the fluxes; bg_tau = the background rate's net tau deposit; val_work = Valencia's own -sqrt(g) S^k d_k alpha;
              the mismatch (bg_tau - vdS_src) against the dense e_int rate (Deps_dense = sum sqrt(g) W rho eps dV)
  addendum 2: per step, the dense Delta(D eps) against |Delta xidot|_coord at the step's end (xidot from the stage-1
              lines), and Delta(D eps) split into stage 1 vs stages 2-3 (stage-start states: u0 at stage s+1 minus at s)
Rates are per unit time; a stage's applied amount is rate x beta_dt (the line's beta_dt).
    c155_kfsplit.py OUT_PREFIX LOG [LOG ...]   (logs in time order: the arm's segments' out.log files)
Writes OUT_PREFIX_stage.csv (one row per stage) and OUT_PREFIX_step.csv (one row per step)."""
import csv
import re
import sys

KEYS = ['cycle', 'stage', 'tstage', 'dtm', 'beta_dt', 'bg_all', 'res_all', 'bg_dense', 'res_dense', 'rms_dpsi',
        'rms_dap', 'rms_dbres', 'max_dpsi', 'max_dap', 'max_dbres', 'nz_frac', 'n_dense', 'vdS_src', 'vdS_flux',
        'bg_tau', 'val_work', 'Deps_dense', 'D_dense']


def parse(logs):
    rows = {}
    for f in logs:
        for ln in open(f, errors='replace'):
            if not ln.startswith('KFSPLIT '):
                continue
            d = dict(re.findall(r'(\w+)=(\S+)', ln))
            if 'vdS_src' not in d:
                continue
            r = {k: float(d[k]) for k in KEYS}
            r['xidot'] = [float(x) for x in d['xidot'].split(',')]
            rows[(int(r['cycle']), int(r['stage']))] = r   # a restart's repeated cycle: the later wins
    return [rows[k] for k in sorted(rows)]


def main():
    out, logs = sys.argv[1], sys.argv[2:]
    st = parse(logs)
    with open(out + '_stage.csv', 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(KEYS + ['mismatch', 'mismatch_over_eint', 'xidot_abs'])
        for r in st:
            e = r['Deps_dense']
            mm = r['bg_tau'] - r['vdS_src']
            w.writerow([r[k] for k in KEYS] + [mm, mm/e if e else 0.0, sum(x*x for x in r['xidot'])**0.5])
    by = {}
    for r in st:
        by.setdefault(int(r['cycle']), {})[int(r['stage'])] = r
    cyc = sorted(by)
    with open(out + '_step.csv', 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['cycle', 't', 'dt', 'Deps', 'dDeps', 'dDeps_s1', 'dDeps_s23', 'dxidot_abs', 'mismatch_applied',
                    'bg_tau_applied', 'vdS_src_applied', 'vdS_flux_applied'])
        for a, b in zip(cyc, cyc[1:]):
            s, n = by[a], by[b]
            if 1 not in s or 1 not in n or b != a + 1:
                continue
            e0, e1 = s[1]['Deps_dense'], n[1]['Deps_dense']
            e2 = s[2]['Deps_dense'] if 2 in s else float('nan')
            xd = [p - q for p, q in zip(n[1]['xidot'], s[1]['xidot'])]
            app = {k: sum(s[q][k]*s[q]['beta_dt'] for q in s) for k in ('bg_tau', 'vdS_src', 'vdS_flux')}
            w.writerow([a, s[1]['tstage'], n[1]['tstage'] - s[1]['tstage'], e0, e1 - e0, e2 - e0, e1 - e2,
                        sum(x*x for x in xd)**0.5, app['bg_tau'] - app['vdS_src'], app['bg_tau'], app['vdS_src'],
                        app['vdS_flux']])
    print(f'{out}: {len(st)} stage rows, {len(cyc)} cycles')


if __name__ == '__main__':
    main()
