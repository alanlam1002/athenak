#!/usr/bin/env python3
"""c250 (R-247 item 1 report): DE branch codes and K in the star's outer shells, per dediag_3d dump paired with the prim_3d dump
at the same time (same mesh, checked). Centre = rho-weighted centroid of rho >= 0.01 rho_c (as c206_outflow.py); d = |x - centre|.
Shells (gas rho > 1e3 rho_atm, as c206): S1 0.7 <= d/R* < 1.0, S2 1.0 <= d/R* < 1.5, S3 d/R* >= 1.5.
Per shell: mass (M*), mass shares of branch codes 0, 11-18 (18 = resync gate held, wip/resync-gate 73389902), mass-weighted K/K0
and the share with K/K0 > 1.05, mass-weighted radial velocity.
    c250_shell_codes.py OUT.csv RUNDIR [--tmax T]"""
import sys, types, glob, argparse, csv
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402

GAM, RC, K0, RS, MS = 5/3, 3.24e-4, 1.2346626297e-2, 0.677, 7.47e-5
CODES = (0, 11, 12, 13, 14, 15, 16, 17, 18)
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('run'); ap.add_argument('--tmax', type=float, default=1e9)
a = ap.parse_args()
pf = {}
for f in glob.glob(f'{a.run}/bin/*.prim_3d.0*.bin'):
    pf[round(bc.read_binary(f)['time'], 6)] = f
rows = []
for fd in sorted(glob.glob(f'{a.run}/bin/*.dediag_3d.*.bin')):
    Dg = bc.read_binary(fd); t = round(Dg['time'], 6)
    if t > a.tmax or t not in pf:
        continue
    P = bc.read_binary(pf[t])
    g = np.asarray(Dg['mb_geometry'], float); n = Dg['nx1_out_mb']
    if not np.array_equal(g, np.asarray(P['mb_geometry'], float)):
        print(f't={t}: mesh differs, skipped', flush=True); continue
    X, Y, Z, V = [], [], [], []
    for q in range(len(g)):
        c = [np.linspace(g[q, 2*k], g[q, 2*k+1], n+1) for k in range(3)]; c = [0.5*(x[1:]+x[:-1]) for x in c]
        zz, yy, xx = np.meshgrid(c[2], c[1], c[0], indexing='ij')
        X.append(xx.ravel()); Y.append(yy.ravel()); Z.append(zz.ravel()); V.append(np.full(n**3, ((g[q, 1]-g[q, 0])/n)**3))
    X, Y, Z, V = map(np.concatenate, (X, Y, Z, V))
    cat = lambda S, k: np.concatenate([np.asarray(S['mb_data'][k][q]).ravel() for q in range(len(g))]).astype(float)
    rho, p = cat(P, 'dens'), cat(P, 'press'); vx, vy, vz = cat(P, 'velx'), cat(P, 'vely'), cat(P, 'velz')
    br = cat(Dg, 'de_branch').round().astype(int)
    m = rho*V; d0 = rho >= 0.01*RC
    cx, cy, cz = (np.average(w[d0], weights=m[d0]) for w in (X, Y, Z))
    vcx, vcy, vcz = (np.average(w[d0], weights=m[d0]) for w in (vx, vy, vz))
    dx, dy, dz = X-cx, Y-cy, Z-cz; d = np.sqrt(dx*dx+dy*dy+dz*dz)/RS
    gas = rho > 1e3*1e-10*RC
    K = p/np.maximum(rho, 1e-300)**GAM/K0
    vr = ((vx-vcx)*dx+(vy-vcy)*dy+(vz-vcz)*dz)/np.maximum(d*RS, 1e-12)
    for sh, s in (('S1', gas & (d >= 0.7) & (d < 1.0)), ('S2', gas & (d >= 1.0) & (d < 1.5)), ('S3', gas & (d >= 1.5))):
        M = m[s].sum(); row = dict(t=t, shell=sh, M=M/MS)
        for k in CODES:
            row[f'c{k}'] = m[s & (br == k)].sum()/M if M > 0 else np.nan
        row['other'] = m[s & ~np.isin(br, CODES)].sum()/M if M > 0 else np.nan
        row['K'] = np.average(K[s], weights=m[s]) if M > 0 else np.nan
        row['fK105'] = m[s & (K > 1.05)].sum()/M if M > 0 else np.nan
        row['vr'] = np.average(vr[s], weights=m[s]) if M > 0 else np.nan
        rows.append(row)
        print({k: (f'{v:.3g}' if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
with open(a.out, 'w') as fo:
    w = csv.DictWriter(fo, list(rows[0])); w.writeheader(); w.writerows(rows)
