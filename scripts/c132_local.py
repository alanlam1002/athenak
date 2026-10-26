#!/usr/bin/env python3
"""c132 (R-145 item 3; research TDE_DEBRIS_SPEC.md §4): diagnostics of a tde_local run (the star
in its own freely falling frame under <cfc> tidal_lapse) from its 3D primitive dumps.
Per dump:
 - stream geometry: the principal axis of the gas with rho > rho_thr (mass-weighted second
   moment about the centre of mass); along it, bins of width ds: the ridge density (max rho),
   the transverse FWHM w = 2 sqrt(2 ln 2) sigma_perp (mass-weighted transverse second moment,
   Gaussian equivalent), the local cell size dx (mass-weighted) and N_perp = w/dx; the mass
   fraction of stream gas in bins with N_perp >= N_w (acceptance, §5);
 - self-gravity: Lambda = rho/rho_tid with rho_tid = |E_max|/(2 pi), E_max the eigenvalue of
   E_ij(t) of largest modulus (the radial one) from the run's tidal table; the stream mass
   fraction with Lambda > 1. Convention: the Newtonian analogue of R-140 item 5's
   M_BH/(2 pi r^3) is |E_rr|/(4 pi); the factor 2 is the radial eigenvalue's -2M/r^3;
 - refinement cost: blocks per level (from each block's cell size);
 - energy spread: eps = 1/2 |v - v_cm|^2 + 1/2 E_ij x^i x^j (v = u~/W, flat-space W; the Fermi
   frame's Newtonian specific energy, argued adequate for the spread), mass quantiles, and a
   dM/d eps histogram file per dump.
    c132_local.py OUTDIR RUNDIR [--thr 1e-8] [--ds 0.25] [--nw 4]"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC = 3.24e-4


def table(d):
    f = glob.glob(f'{d}/tidal_*.txt')
    if not f:
        return None
    return np.loadtxt(f[0])


def eij_at(T, t):
    if T is None:
        return np.zeros((3, 3))
    e = [np.interp(t, T[:, 0], T[:, 4 + q]) for q in range(6)]
    return np.array([[e[0], e[1], e[2]], [e[1], e[3], e[4]], [e[2], e[4], e[5]]])


def load(f):
    P = bc.read_binary(f)
    g = np.asarray(P['mb_geometry']); n = P['nx1_out_mb']
    q = P['mb_data']
    X, Y, Z, R, VX, VY, VZ, DX = [], [], [], [], [], [], [], []
    for b in range(len(g)):
        h = (g[b, 1] - g[b, 0])/n
        c = lambda lo: lo + (np.arange(n) + 0.5)*h
        z, y, x = np.meshgrid(c(g[b, 4]), c(g[b, 2]), c(g[b, 0]), indexing='ij')
        X.append(x.ravel()); Y.append(y.ravel()); Z.append(z.ravel())
        R.append(np.asarray(q['dens'][b]).ravel())
        VX.append(np.asarray(q['velx'][b]).ravel()); VY.append(np.asarray(q['vely'][b]).ravel())
        VZ.append(np.asarray(q['velz'][b]).ravel()); DX.append(np.full(n**3, h))
    hs = np.array([(g[b, 1] - g[b, 0])/n for b in range(len(g))])
    cat = np.concatenate
    return float(P['time']), cat(X), cat(Y), cat(Z), cat(R), cat(VX), cat(VY), cat(VZ), \
        cat(DX), hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('run')
    ap.add_argument('--thr', type=float, default=1e-8)    # rho_thr / rho_c
    ap.add_argument('--ds', type=float, default=0.25)
    ap.add_argument('--nw', type=float, default=4.0)
    a = ap.parse_args()
    lab = os.path.basename(a.run.rstrip('/'))
    T = table(a.run)
    rows, prof = [], []
    for f in sorted(glob.glob(f'{a.run}/bin/{B}.prim_3d.*.bin')):
        t, X, Y, Z, rho, ux, uy, uz, dx, hs = load(f)
        dm = rho*dx**3
        m = rho > a.thr*RHOC
        if not m.any():
            continue
        M = dm[m].sum()
        c = np.array([(X*dm)[m].sum(), (Y*dm)[m].sum(), (Z*dm)[m].sum()])/M
        r = np.stack([X - c[0], Y - c[1], Z - c[2]], axis=1)[m]
        w = dm[m]
        I = (r*w[:, None]).T @ r/M
        ev, evec = np.linalg.eigh(I)
        ax = evec[:, -1]                                   # the stream axis
        s = r @ ax
        perp2 = np.sum(r*r, axis=1) - s*s
        W = np.sqrt(1.0 + ux[m]**2 + uy[m]**2 + uz[m]**2)
        v = np.stack([ux[m], uy[m], uz[m]], axis=1)/W[:, None]
        vcm = (v*w[:, None]).sum(axis=0)/M
        E = eij_at(T, t)
        eps = 0.5*np.sum((v - vcm)**2, axis=1) + 0.5*np.einsum('ni,ij,nj->n', r, E, r)
        evE = np.linalg.eigvalsh(E)
        emax = np.abs(evE).max()
        lam = rho[m]/(emax/(2*np.pi)) if emax > 0 else np.full(m.sum(), np.inf)
        # bins along the axis
        edges = np.arange(np.floor(s.min()/a.ds)*a.ds, s.max() + a.ds, a.ds)
        ib = np.digitize(s, edges) - 1
        okmass = 0.0
        for b in range(len(edges) - 1):
            k = ib == b
            if not k.any():
                continue
            wb = w[k]
            sig = np.sqrt((perp2[k]*wb).sum()/wb.sum())
            fwhm = 2.0*np.sqrt(2.0*np.log(2.0))*sig
            dxb = (dx[m][k]*wb).sum()/wb.sum()
            nperp = fwhm/dxb
            if nperp >= a.nw:
                okmass += wb.sum()
            prof.append(f'{lab},{t:.4g},{0.5*(edges[b] + edges[b + 1]):.4g},{wb.sum():.5e},'
                        f'{rho[m][k].max()/RHOC:.5e},{fwhm:.5e},{dxb:.5e},{nperp:.4g}')
        lev = np.round(np.log2(hs.max()/hs)).astype(int)
        nl = np.bincount(lev)
        q = np.quantile(eps, [0.05, 0.25, 0.5, 0.75, 0.95])
        h, e_ = np.histogram(eps, bins=60, weights=w)
        np.savetxt(f'{a.out}/{lab}_dMdE_t{t:07.2f}.csv',
                   np.stack([0.5*(e_[1:] + e_[:-1]), h], axis=1), delimiter=',',
                   header='eps,dM', comments='')
        rows.append(f'{lab},{t:.4g},{M:.6e},{s.max() - s.min():.4g},{np.sqrt(ev[-1]):.4g},'
                    f'{np.sqrt(ev[0]):.4g},{rho.max()/RHOC:.5e},{okmass/M:.4f},'
                    f'{w[lam > 1].sum()/M:.4f},{emax:.4e},' + ','.join(f'{x:.4e}' for x in q)
                    + ',' + ' '.join(str(x) for x in nl))
    hdr = ('run,t,M_stream,length,sigma_major,sigma_minor,rhomax_over_rhoc,frac_mass_Nperp_ge_Nw,'
           'frac_mass_Lambda_gt1,E_max,eps_q05,eps_q25,eps_q50,eps_q75,eps_q95,blocks_per_level')
    open(f'{a.out}/{lab}_c132.csv', 'w').write(hdr + '\n' + '\n'.join(rows) + '\n')
    open(f'{a.out}/{lab}_c132_profile.csv', 'w').write(
        'run,t,s,dM,rho_ridge_over_rhoc,fwhm,dx,N_perp\n' + '\n'.join(prof) + '\n')
    print(hdr); print('\n'.join(rows))


if __name__ == '__main__':
    main()
