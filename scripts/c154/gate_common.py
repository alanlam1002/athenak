"""c154 gates: shared helpers (compare runs with cpu/cmp_runs.py, the ghost/owner probe with c147_kprobe_check.py)"""
import subprocess, sys
E = sys.argv[1]
def cmp(a, b, tol=0.0, extra=()):
    o = subprocess.run(['python3', E + '/cpu/cmp_runs.py', a, b, repr(tol)] + list(extra),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
    print(f'--- {a} vs {b} (tol {tol:g})'); print(o.strip())
    return 'RESULT PASS' in o
def probe(d):
    o = subprocess.run(['python3', '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts/c147_kprobe_check.py', d], stdout=subprocess.PIPE, universal_newlines=True).stdout
    print(o.strip())
    return ('mismatched=0 ' in o) and ('shared=0 ' not in o)
def done(name, ok):
    print(name, 'gate:', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
