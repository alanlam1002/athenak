#!/usr/bin/env python3
"""c153 (R-175 item 2 / R-172 item 4, user-approved 2026-10-30): the remnant-heating discriminator. For the
Hill-bound remnant of a restart (centre and r_H as c151_km.py), per bin of the enclosed-mass coordinate m (M*),
which dual-energy branch protects the gas. The branch rule is the C2P's (athenak 516fee6e,
src/eos/primitive_solver_hyd.hpp:2537-2615):
  dense = D > <mhd> dual_energy_rho_switch (D = u0's IDN / sqrt(gamma), the Eulerian rest-mass density);
  dense and unflagged (de_shock == 0)  -> entropy branch (dbr 1);
  dense and flagged                    -> energy branch, unless the second-law guard (K_en <= K_tr (1 + tol))
                                          puts it back on entropy (dbr 2);
  not dense                            -> energy (dbr 0), K resynced from the energy solution.
The restart carries the flag (restart_state group dyngr.flag, restart_state.hpp), so (i) "below rho_switch",
(iii) "dense unflagged = entropy" and "dense flagged" are exact at the restart's instant. Within the flagged cells
the guard/energy split needs that stage's energy solution (after either branch the tracer equals K(w0)), so it is
not separable from a restart; "dense flagged" is the upper bound on (ii).
Also (iv): the Hill sphere's mass-weighted bulk Eulerian velocity |<v>|_phys = sqrt(psi^4 delta_ij <v^i><v^j>)
and the mass-weighted mean speed <|v|_phys> (v^i = gamma^ij u_j / W; the Eulerian velocity does not depend on the
comoving shift of the spatial coordinates).
    c153_de_branch.py OUT LABEL=RST:TP [...]  -> OUT/c153_branch.csv (per restart, m bin), OUT/c153_summary.csv"""
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import remnant as RM   # noqa: E402
import tde_dmde as T   # noqa: E402

MSTAR = 7.47e-5
MBH = 1.0
MEDGES = np.round(np.arange(0.0, 0.2001, 0.01), 4)
BAND = (0.04, 0.15)        # c151's heated shells (R-175 item 2)


def de_cells(fn, c):
    """per cell, in tde_dmde.cells' order (rcut as c), the flag slot, D, the tracer K_tr and the stored
    primitives' K = P/rho^Gamma (w0)"""
    t = T.header(fn)
    par = t['par']
    lay = re.search(r'(?m)^restart_state_layout\s*=\s*(\S+)', par).group(1)
    off, iflag, iw0 = 0, None, None
    for g in lay.split(','):
        name, n = g.split(':')
        if name == 'dyngr.flag': iflag = off
        if name == 'dyngr.w0': iw0 = off
        off += int(n)
    gp = lambda k: float(re.search(r'(?m)^%s\s*=\s*([-+.\deE]+)' % k, par).group(1))
    rsw, kmax = gp('dual_energy_rho_switch'), gp('dual_energy_kmax')
    gam = 5.0 / 3.0
    nmb = t['nmb']
    nx, ng, nscal = 16, 4, 1
    no = nx + 2 * ng
    nmhd = 5 + nscal
    ncc = no**3
    bsz = (no + 1) * no * no * 3
    base = nmhd * ncc + bsz + 17 * ncc
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(nmb, t['dsize'] // 8))
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    ll, lev = t['ll'], t['ll'][:, 3]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min])
    lo = np.array([x1min, x2min, x3min])
    s = slice(ng, ng + nx)
    ic = np.arange(nx) + 0.5
    xbh = c['xbh']
    F, Ds, KT, KW = [], [], [], []
    for b in range(nmb):
        rec = mm[b]
        cons = rec[:nmhd * ncc].reshape(nmhd, no, no, no)[:, s, s, s]
        adm = rec[nmhd * ncc + bsz:base].reshape(17, no, no, no)[:, s, s, s]
        st = rec[base:].reshape(-1, no, no, no)[:, s, s, s]
        ext = w / 2.0**lev[b]
        dx = ext / nx
        x = lo[0] + ll[b, 0] * ext[0] + ic * dx[0]
        y = lo[1] + ll[b, 1] * ext[1] + ic * dx[1]
        z = lo[2] + ll[b, 2] * ext[2] + ic * dx[2]
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        keep = np.sqrt((X - xbh[0])**2 + (Y - xbh[1])**2 + (Z - xbh[2])**2) >= 3.0
        if not keep.any():
            continue
        # the same mask as tde_dmde.cells (D > 0, finite tau), with its determinant
        gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
        det = (gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz))
        sg = np.sqrt(det)
        D = cons[0] / sg
        ok = keep & (D > 0) & np.isfinite(cons[4] / sg)
        F.append(st[iflag][ok]); Ds.append(D[ok])
        KT.append(kmax * np.clip(cons[5][ok] / cons[0][ok], 0.0, None))   # Y = DY/D, ypow = 1
        KW.append(st[iw0 + 4][ok] / st[iw0][ok]**gam)
    cat = np.concatenate
    return cat(F), cat(Ds), cat(KT), cat(KW), rsw


def one(lab, fn, tp):
    c = T.cells(fn, rcut=3.0)
    fl, D, ktr, kw, rsw = de_cells(fn, c)
    assert len(fl) == len(c['dM']), 'cell order mismatch'
    Mb, xc, _, _ = RM.bound_mass(c)
    X = np.stack([c['x'], c['y'], c['z']])
    r = np.linalg.norm(X - xc[:, None], axis=0)
    rbh = float(np.linalg.norm(xc - np.asarray(c['xbh'], dtype=float)))
    rH = rbh * (Mb / (3.0 * MBH))**(1.0 / 3.0)
    sel = np.where(r < rH)[0]
    o = sel[np.argsort(r[sel])]
    dm = c['dM'][o]
    m = (np.cumsum(dm) - 0.5 * dm) / MSTAR
    dense = D[o] > rsw
    flag = fl[o] != 0
    cls = {'below_rsw': ~dense, 'dense_unflagged_entropy': dense & ~flag, 'dense_flagged': dense & flag}
    rows = []
    for a, b in zip(MEDGES[:-1], MEDGES[1:]):
        q = (m >= a) & (m < b)
        if not q.any():
            continue
        M = dm[q].sum()
        row = dict(label=lab, t=c['t'], t_minus_tp=c['t'] - tp, m_lo=a, m_hi=b, r_mean=float(np.average(r[o][q], weights=dm[q])))
        for k, v in cls.items():
            row[k] = dm[q & v].sum() / M
        row['K_K0'] = float(np.average(kw[o][q], weights=dm[q])) / 1.2346626297e-2
        rows.append(row)
    q = (m >= BAND[0]) & (m < BAND[1])
    M = dm[q].sum()
    # (iv) over the Hill sphere: Eulerian velocity, |.|_phys with gamma_ij = psi^4 delta_ij
    w_ = c['dM'][sel]
    p4 = c['psi4'][sel]
    vx, vy, vz = c['vx'][sel], c['vy'][sel], c['vz'][sel]
    vb = np.array([np.average(vx, weights=w_), np.average(vy, weights=w_), np.average(vz, weights=w_)])
    p4m = np.average(p4, weights=w_)
    vbulk = float(np.sqrt(p4m * (vb**2).sum()))
    vmean = float(np.average(np.sqrt(p4 * (vx**2 + vy**2 + vz**2)), weights=w_))
    vrel = float(np.average(np.sqrt(p4 * ((vx - vb[0])**2 + (vy - vb[1])**2 + (vz - vb[2])**2)), weights=w_))
    summ = dict(label=lab, t=c['t'], t_minus_tp=c['t'] - tp, M_rem=Mb / MSTAR, r_H=rH, r_bh=rbh,
                band_mass=M / MSTAR, rho_switch=rsw,
                **{k: dm[q & v].sum() / M for k, v in cls.items()},
                v_bulk_phys=vbulk, v_mean_phys=vmean, v_rel_bulk_phys=vrel)
    return rows, summ


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    R, S = [], []
    for a in sys.argv[2:]:
        lab, rest = a.split('=')
        fn, tp = rest.rsplit(':', 1)
        rows, summ = one(lab, fn, float(tp))
        R += rows; S.append(summ)
        print(summ, flush=True)
    for name, data in (('c153_branch.csv', R), ('c153_summary.csv', S)):
        with open(f'{out}/{name}', 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            wr.writeheader(); wr.writerows(data)


if __name__ == '__main__':
    main()
