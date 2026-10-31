#!/usr/bin/env python3
"""c158 in-job gate (R-186 item 1): after the fix, A^0_tr (dense, per step) on the post-regrid steps must collapse to the level
of the other steps. TB1 (before): post-regrid 5.7-8.0e-11 per step, the others |.| ~ 1e-22. Pass: max |A0tr| per step <= 1e-14
over every step of TB1F with >= 5 post-regrid steps.   gate_a0.py RUNDIR"""
import subprocess, sys, csv
E = sys.argv[1]
S = '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts/c157_ktb.py'
subprocess.run([sys.executable, S, E + '/ana_TB1F', E + '/TB1F/out.log'], check=True)
rows = [r for r in csv.DictReader(open(E + '/ana_TB1F_step.csv')) if r['group'] == 'dense']
post = [r for r in rows if r['post_regrid'] == '1']
mx = max(abs(float(r['A0tr_amt'])) for r in rows)
mp = max((abs(float(r['A0tr_amt'])) for r in post), default=0.0)
print(f'GATE_a0: steps {len(rows)} post-regrid {len(post)} max|A0tr| all {mx:.3e} post {mp:.3e}')
sys.exit(0 if (len(post) >= 5 and mx <= 1e-14) else 1)
