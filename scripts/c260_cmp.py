#!/usr/bin/env python3
"""c260 (R-257 2(c), bind-policy pilot): baseline B0 vs policy B1, the cheap metrics (login node).
  4. history fluxes: user.hst mdot_in_r1 (r = 50), mdot_in_r2 (r = 30), integrated over the window, and mass_in_rm (r < 30)
     at matched times; pass = within 2%.
  6. MCLOSE per segment (out.log): R and PASS.
  7. recorded: blocks by level at each checkpoint (restart header), cycles, mean dt, wall/node-h per segment (runs.log).
    c260_cmp.py RUNDIR(c260_pilot)
"""
import glob, os, re, sys
from datetime import datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rst_tree import read_tree

E = sys.argv[1]
ARMS = ('B0', 'B1')


def hst(a):
    fn = f'{E}/{a}/cfc_tde_wd_imbh_parabolic_tracker.user.hst'
    names = None
    for line in open(fn):
        if line.startswith('#') and '[1]=' in line:
            names = [m.group(1) for m in re.finditer(r'\[\d+\]=(\S+)', line)]
    d = np.loadtxt(fn, comments='#')
    _, i = np.unique(d[:, 0], return_index=True)    # restarts repeat rows: keep the first of each time
    d = d[np.sort(i)]
    d = d[np.argsort(d[:, 0], kind='stable')]
    return {n: d[:, k] for k, n in enumerate(names)}


print('## 4. history (user.hst)')
H = {a: hst(a) for a in ARMS}
t = H['B0']['time']
for col, lab in (('mdot_in_r1', 'Mdot_in r=50'), ('mdot_in_r2', 'Mdot_in r=30')):
    I = {}
    for a in ARMS:
        h = H[a]
        I[a] = np.trapz(h[col], h['time'])
    # matched-time series difference on the B0 grid
    y1 = np.interp(t, H['B1']['time'], H['B1'][col])
    rel = (I['B1'] - I['B0']) / abs(I['B0'])
    print(f'{lab:14s} integral 900-1200: B0 {I["B0"]:.6e}  B1 {I["B1"]:.6e}  rel {rel:+.3%}  '
          f'{"PASS" if abs(rel) < 0.02 else "FAIL"} (2%);  max|B1-B0|/max|B0| = {np.max(abs(y1 - H["B0"][col])) / np.max(abs(H["B0"][col])):.3%}')
for tt in (1000, 1100, 1200):
    v = [np.interp(tt, H[a]['time'], H[a]['mass_in_rm']) for a in ARMS]
    rel = (v[1] - v[0]) / abs(v[0])
    print(f'M(r<30) t={tt}: B0 {v[0]:.6e}  B1 {v[1]:.6e}  rel {rel:+.3%}  {"PASS" if abs(rel) < 0.02 else "FAIL"} (2%)')
for a in ARMS:
    h = H[a]
    print(f'{a}: hst rows {len(h["time"])}, t {h["time"][0]:.2f}-{h["time"][-1]:.2f}, dt mean {np.mean(h["dt"]):.5f} '
          f'min {np.min(h["dt"]):.5f} max {np.max(h["dt"]):.5f}; dt at t=1000/1100/1200: '
          + ' '.join(f'{np.interp(x, h["time"], h["dt"]):.5f}' for x in (1000, 1100, 1199.9)))

print('\n## 6. MCLOSE (per segment, out.log)')
for a in ARMS:
    for line in open(f'{E}/{a}/out.log', errors='replace'):
        if line.startswith('MCLOSE '):
            g = lambda k: re.search(k + r'=(\S+)', line).group(1)
            print(f'{a} cycles {g("cycles"):12s} t={g("time"):5s} R={g("R"):>16s} R/M*={g("R/M\\*"):>16s} '
                  f'{"PASS" if " PASS" in line else "FAIL"}')

print('\n## 7. recorded: blocks by physical level at the checkpoints (restart headers), cost')
for a in ARMS:
    for r in sorted(glob.glob(f'{E}/{a}/checkpoints/*/*.rst')) + [sorted(glob.glob(f'{E}/{a}/rst/*00021x512.rst'))[0]]:
        tr = read_tree(r)
        lev = tr['ll'][:, 3] - tr['root']
        c = np.bincount(lev, minlength=7)
        print(f'{a} t={tr["time"]:8.2f} cyc {tr["ncycle"]:6d} nmb {tr["nmb"]:6d} by level: ' + ' '.join(f'L{i}:{n}' for i, n in enumerate(c)))
seg = re.compile(r'^(B[01]) start (\S+) job (\S+) nodes (\d+).*time/tlim=(\S+)')
end = re.compile(r'^(B[01]) rc=(\d+) (.*) end (\S+)')
for a in ARMS:
    tot = 0.0
    cur = None
    for line in open(f'{E}/runs.log'):
        m = seg.match(line)
        if m and m.group(1) == a:
            cur = m
            continue
        m2 = end.match(line)
        if m2 and m2.group(1) == a and cur is not None:
            t0 = datetime.fromisoformat(cur.group(2).rstrip('Z'))
            t1 = datetime.fromisoformat(m2.group(4).rstrip('Z'))
            h = (t1 - t0).total_seconds() / 3600
            ok = 'time limit' in m2.group(3)
            if ok:
                tot += h * int(cur.group(4))
            print(f'{a} seg to t={cur.group(5):7s} {h:5.2f} h x {cur.group(4)} n {"" if ok else "(crashed, not counted) "}{m2.group(3)[:60]}')
            cur = None
    print(f'{a} total (completed segments) {tot:.1f} node-h')
