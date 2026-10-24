"""Metrics of the flat-space moving-shock companions (c103_e3b_mk.py; C-097 E3's method). Per
(M, vb, nx): the shocked slab's post-shock entropy excess K/K0 - 1 (median over a window inside the
slab, clear of the shocks and the contact) at t_end, against Rankine-Hugoniot (K_RH - 1):
  energy-only (dual energy off): f_RH = (K - 1)/(K_RH - 1), the base scheme's shock heating;
  dual energy on (covariant): capture = (K_on - 1)/(K_off - 1) as in C-097, and its f_RH;
plus, with dual energy on, the flagged cell-dumps at / away from the shocks. Writes a CSV.

    c103_e3b_metrics.py RUNDIR OUT.csv
"""
import glob
import sys

import numpy as np

D, CSV = sys.argv[1], sys.argv[2]
GAM, K0 = 5 / 3, 0.01 / 1.0**(5 / 3)


def tabs(d, kind):
    out = []
    for fn in sorted(glob.glob(f'{d}/tab/*{kind}*.tab')):
        t = float(open(fn).readline().split('time=')[1].split()[0])
        out.append((t, np.loadtxt(fn, ndmin=2)))
    return out


def krh(M):
    p = (2 * GAM * M**2 - (GAM - 1)) / (GAM + 1)
    r = (GAM + 1) * M**2 / ((GAM - 1) * M**2 + 2)
    return p / r**GAM, r


def slab(x, rho, rc):
    i = np.nonzero(rho > (1 + 0.5 * (rc - 1)) * np.median(rho))[0]   # shocked (compressed) gas
    return x[i.min()], x[i.max()]


rows = ['M,vb,nx,K_RH-1,off_K-1,off_f_RH,on_K-1,on_capture,on_f_RH,on_flags_at_shocks,on_flags_elsewhere']
for M in ('1.2', '2'):
    kr, rc = krh(float(M))
    for vb in ('0.0', '0.1', '0.2', '0.3'):
        for nx in ('256', '512', '1024'):
            base = f'{D}/col_M{M}_vb{vb}_n{nx}'
            t, a = tabs(base + '_off', 'mhd_w_bcc')[-1]
            x = a[:, 2]
            dx = x[1] - x[0]
            xl, xr = slab(x, a[:, 3], rc)
            c, hw = 0.5 * (xl + xr), 0.5 * (xr - xl)
            win = (np.abs(x - c) > 0.04) & (np.abs(x - c) < min(0.10, hw - 4 * dx))
            koff = np.median(a[win, 7] / a[win, 3]**GAM / K0) - 1
            t, b = tabs(base + '_covariant', 'mhd_w_bcc')[-1]
            kon = np.median(b[win, 7] / b[win, 3]**GAM / K0) - 1
            at = away = 0
            for (tw, w), (td, dg) in zip(tabs(base + '_covariant', 'mhd_w_bcc'),
                                         tabs(base + '_covariant', 'mhd_de_diag')):
                if tw == 0:
                    continue
                f = dg[:, 5] > 0.5
                try:
                    l, r = slab(w[:, 2], w[:, 3], rc)
                except ValueError:
                    continue
                near = np.minimum(np.abs(x - l), np.abs(x - r)) <= 4 * dx
                at += int((f & near).sum())
                away += int((f & ~near).sum())
            rows.append(f'{M},{vb},{nx},{kr - 1:.5e},{koff:.5e},{koff / (kr - 1):.4f},{kon:.5e},'
                        f'{kon / koff:.4f},{kon / (kr - 1):.4f},{at},{away}')
open(CSV, 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
