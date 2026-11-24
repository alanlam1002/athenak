#!/usr/bin/env python3
"""bhstar WP2 branch 3, test E7(A): compare OpacityFunction (saha_kramers) with the reference
docs/numerics/wp2/opacity_levelA.py at sampled (rho, T). Pass: max relative difference < 1e-10.
Usage: run_e7a.py <path to opacity_levelA.py> <e7a_opacity executable>"""
import importlib.util, subprocess, sys
import numpy as np
spec = importlib.util.spec_from_file_location("ref", sys.argv[1]); ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
rho = np.logspace(-18, -4, 29)
T = np.logspace(np.log10(2.0e3), np.log10(1.0e6), 61)
R, TT = np.meshgrid(rho, T, indexing='ij')
pairs = np.stack([R.ravel(), TT.ravel()], axis=1)
inp = "\n".join(f"{a:.17e} {b:.17e}" for a, b in pairs) + "\n"
out = subprocess.run([sys.argv[2]], input=inp, capture_output=True, text=True, check=True).stdout
c = np.array([list(map(float, l.split())) for l in out.splitlines()])
ks, kaR, kaP = ref.kappa_levelA(c[:, 0], c[:, 1])
worst = 0.0
for name, a, b in (("kappa_s", c[:, 2], ks), ("kappa_aR", c[:, 3], kaR), ("kappa_P", c[:, 4], kaP)):
    nz = np.abs(b) > 0
    rel = np.max(np.abs(a[nz]/b[nz] - 1)) if nz.any() else 0.0
    zero_ok = np.all(a[~nz] == 0)
    print(f"{name:9s}: {nz.sum():5d} nonzero points, max rel diff {rel:.3e}; ref-zero points all zero: {zero_ok}")
    worst = max(worst, rel if zero_ok else np.inf)
# edge cases: T = 0 and very low T must give finite zero opacities (no 0 * inf)
edge = subprocess.run([sys.argv[2]], input="1e-10 0\n1e-10 10\n1e-10 100\n1e-30 1e9\n",
                      capture_output=True, text=True, check=True).stdout
print("edge cases (rho T kappa_s kappa_aR kappa_P):\n" + edge.rstrip())
ok_edge = all(np.isfinite(float(x)) for l in edge.splitlines() for x in l.split())
print(f"E7(A): max rel diff {worst:.3e} -> {'PASS' if worst < 1e-10 and ok_edge else 'FAIL'}")
sys.exit(0 if worst < 1e-10 and ok_edge else 1)
