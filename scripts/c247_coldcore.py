#!/usr/bin/env python3
"""c247 (R-247 item 2, user-approved 2026-11-08): the new pilots' cold core on the approach, offline.
Per prim_3d dump (t <= --tmax) and the dediag_3d dump at the same time (same cycle: same blocks, same order):
  core = r < 0.2 R* about the rho-centroid of rho >= 0.01 rho_c (C-207, c160_census.py's definition), and the
  shells 0.2-0.5 R*, 0.5-0.8 R* (rho >= 0.01 rho_c) for where the cooling starts;
  K_P/K0 = P/rho^gamma/K0 (the pressure the C2P chose: K_en on the energy branch, K_tr on the tracer branch);
  K_tr/K0 = kmax Y^gamma/K0 (Y = s_00, the k_1g tracer after the C2P: dual_energy_variable = k_1g, nscalars 1);
  dk = de_ken_ktr = K_en/K_tr - 1 of the last C2P (codes 11-15 except 13): the core mass shares with
  dk < -0.05, -0.05 <= dk < -0.02, -0.02 <= dk < -0.005, |dk| < 0.005, dk >= 0.005;
  the branch-code mass shares (11, 12, 13, 14, 15, 18);
  r_bh = |x_c - x_BH| and the tide (16.4/r_bh)^3, to separate time from distance (P10 and P18 differ in r(t)).
    c247_coldcore.py OUT.csv RUNDIR [--tmax T] [--rstar 0.677]"""
import glob
import sys
import types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh, par   # noqa: E402

GAM = 5.0/3.0
B = 'cfc_tde_wd_imbh_parabolic_tracker'


def main():
    out, d = sys.argv[1], sys.argv[2]
    tmax = float(sys.argv[sys.argv.index('--tmax') + 1]) if '--tmax' in sys.argv else 1e30
    rs = float(sys.argv[sys.argv.index('--rstar') + 1]) if '--rstar' in sys.argv else 0.677
    rhoc, k0, kmax = par(d, 'rhoc'), par(d, 'kappa'), par(d, 'dual_energy_kmax', 1.0)
    dd = {}
    for f in glob.glob(f'{d}/bin/{B}.dediag_3d.*.bin'):
        dd[round(bc.read_binary(f)['time'], 6)] = f
    rows = []
    for fp in sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin')):
        P = bc.read_binary(fp); t = P['time']
        if t > tmax or round(t, 6) not in dd:
            continue
        Dg = bc.read_binary(dd[round(t, 6)])
        g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
        if not np.array_equal(g, np.asarray(Dg['mb_geometry'], float)):
            print(f't={t}: block layouts differ, skipped', flush=True); continue
        cat = lambda S, k: np.concatenate([np.asarray(S['mb_data'][k][b]).ravel() for b in range(len(g))]).astype(float)
        rho, p, Y = cat(P, 'dens'), cat(P, 'press'), cat(P, 's_00')
        br = cat(Dg, 'de_branch').round().astype(int); dk = cat(Dg, 'de_ken_ktr')
        h = (g[:, 1] - g[:, 0])/n
        ax = lambda lo: (g[:, lo][:, None] + (np.arange(n) + 0.5)[None, :]*h[:, None])
        X, Yc, Z = ax(0), ax(2), ax(4)
        x = np.broadcast_to(X[:, None, None, :], (len(g), n, n, n)).ravel()
        y = np.broadcast_to(Yc[:, None, :, None], (len(g), n, n, n)).ravel()
        z = np.broadcast_to(Z[:, :, None, None], (len(g), n, n, n)).ravel()
        m = rho*np.repeat(h**3, n**3)
        d01 = rho >= 0.01*rhoc
        xc = np.array([(m[d01]*c[d01]).sum()/m[d01].sum() for c in (x, y, z)])
        rr = np.sqrt((x - xc[0])**2 + (y - xc[1])**2 + (z - xc[2])**2)/rs
        rbh = float(np.linalg.norm(xc - xbh(d, t)))
        kp = p/np.maximum(rho, 1e-300)**GAM/k0
        ktr = kmax*np.maximum(Y, 0.0)**GAM/k0
        r = dict(t=t, r_bh=rbh, tide=(16.4/rbh)**3, rhomax=float(rho.max()/rhoc))
        for lab, s in (('core', rr < 0.2), ('s25', d01 & (rr >= 0.2) & (rr < 0.5)), ('s58', d01 & (rr >= 0.5) & (rr < 0.8))):
            M = m[s].sum()
            r[f'{lab}_M'] = M
            r[f'{lab}_KP'] = (m[s]*kp[s]).sum()/M
            r[f'{lab}_Ktr'] = (m[s]*ktr[s]).sum()/M
            v = s & (br >= 11) & (br <= 15) & (br != 13)
            Mv = m[v].sum()
            r[f'{lab}_Ken'] = (m[v]*ktr[v]*(1.0 + dk[v])).sum()/Mv if Mv > 0 else np.nan
            for nm, lo, hi in (('lt05', -np.inf, -0.05), ('m05_02', -0.05, -0.02), ('m02_005', -0.02, -0.005),
                               ('pm005', -0.005, 0.005), ('ge005', 0.005, np.inf)):
                r[f'{lab}_dk_{nm}'] = m[v & (dk >= lo) & (dk < hi)].sum()/Mv if Mv > 0 else np.nan
            for b in (11, 12, 13, 14, 15, 18):
                r[f'{lab}_br{b}'] = m[s & (br == b)].sum()/M
        rows.append(r)
        print(f"t={t:7.2f} r_bh={rbh:6.2f} | core K_P {r['core_KP']:.4f} K_tr {r['core_Ktr']:.4f} K_en {r['core_Ken']:.4f}"
              f" dk<-5% {r['core_dk_lt05']:.3f} -5..-2% {r['core_dk_m05_02']:.3f} -2..-0.5% {r['core_dk_m02_005']:.3f}"
              f" | br11 {r['core_br11']:.2f} 12 {r['core_br12']:.2f} 14 {r['core_br14']:.2f} 15 {r['core_br15']:.2f}"
              f" | s25 K_P {r['s25_KP']:.4f} s58 K_P {r['s58_KP']:.4f}", flush=True)
    keys = list(rows[0].keys())
    with open(out, 'w') as fo:
        fo.write(','.join(keys) + '\n')
        for r in rows:
            fo.write(','.join(str(r[k]) for k in keys) + '\n')


if __name__ == '__main__':
    main()
