#!/usr/bin/env python3
"""R-242 item 1: the dense internal energy for fig 2's normalisation, offline on VB10's dumps (c161_b1/VB10).
Per prim_3d dump: E_int = sum over rho >= 0.01 rho_c of sqrt(g) e_int dV, e_int = P/(Gamma - 1) (ideal gas, Gamma = 5/3),
sqrt(g) = psi^6 (CFC, conformally flat). psi^4 = g_xx is taken from the adm_xy slice at the nearest time, at the cell nearest the
dense gas's rho-centroid (one value per dump: across 1.4 R* psi^4 varies by < 1e-3 at r_BH >= 18, argued).
Same sqrt(g) dV weighting as the TRD deposit sums (fig 2's source, code/results/c169/logs.txt).
    paper_fig2_eint.py RUNDIR OUT.csv"""
import sys, types, glob
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402
GAM, RC = 5/3, 3.24e-4
run, out = sys.argv[1], sys.argv[2]
adm = {round(bc.read_binary(f)['time'], 3): f for f in glob.glob(f'{run}/bin/*.adm_xy.*.bin')}
at = np.array(sorted(adm))
rows = []
for f in sorted(glob.glob(f'{run}/bin/*.prim_3d.0*.bin')):
    P = bc.read_binary(f); t = P['time']; g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
    E = 0.0; M = 0.0; cx = np.zeros(3)
    for q in range(len(g)):
        rho = np.asarray(P['mb_data']['dens'][q], float); s = rho >= 0.01*RC
        if not s.any(): continue
        p = np.asarray(P['mb_data']['press'][q], float); dV = ((g[q, 1]-g[q, 0])/n)**3
        c = [0.5*(np.linspace(g[q, 2*k], g[q, 2*k+1], n+1)[1:] + np.linspace(g[q, 2*k], g[q, 2*k+1], n+1)[:-1]) for k in range(3)]
        zz, yy, xx = np.meshgrid(c[2], c[1], c[0], indexing='ij')
        E += (p[s]/(GAM - 1.0)).sum()*dV; m = (rho[s]*dV); M += m.sum()
        cx += np.array([(m*xx[s]).sum(), (m*yy[s]).sum(), (m*zz[s]).sum()])
    cx /= M
    A = bc.read_binary(adm[at[np.argmin(np.abs(at - t))]])
    ga = np.asarray(A['mb_geometry'], float); na = A['nx1_out_mb']; best = (1e30, np.nan)
    for q in range(len(ga)):
        if not (ga[q, 0] <= cx[0] < ga[q, 1] and ga[q, 2] <= cx[1] < ga[q, 3]): continue
        h = (ga[q, 1] - ga[q, 0])/na
        if h < best[0]:
            i = min(int((cx[0] - ga[q, 0])/h), na - 1); j = min(int((cx[1] - ga[q, 2])/((ga[q, 3] - ga[q, 2])/na)), na - 1)
            best = (h, float(np.asarray(A['mb_data']['adm_gxx'][q]).reshape(-1, na, na)[0, j, i]))
    p4 = best[1]
    rows.append((t, p4, E*p4**1.5, E, M))
    print(f't={t:.2f} psi4={p4:.6f} Eint_sqrtg={E*p4**1.5:.6e} Eint_flat={E:.6e}', flush=True)
np.savetxt(out, rows, delimiter=',', header='t,psi4,Eint_sqrtg,Eint_flat,M_dense_flat', comments='', fmt='%.8g')
