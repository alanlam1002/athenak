#!/usr/bin/env python3
"""c260 (R-257 2(c), bind-policy pilot): the full-3D metrics from one restart, streamed per MeshBlock (multiprocessing),
so a 350 GB restart needs only small accumulators. Per-cell physics as tde_dmde.cells (same c2p, E = -u_t - 1 with the
frame's xidot, dM = sqrt(gamma) D dV, r_BH >= rcut = 3); cells with D <= 1e-16 (the 1e-22 atmosphere) are skipped:
their mass is < 1e-9 of M*.
  1. dM/dE: f_bound (E < 0) and the bound set's 10/50/90% mass quantiles of E/dE (histogram, 1e-4 dE bins);
  2. bound stream (E < 0, 1e-6 <= rho/rho_c < 0.01, c253's class) at R = 50, 100, 150 (|r_BH - R| < 1): the azimuth (2-deg
     bins, orbital plane about x_BH) with the most bound-stream mass; cells within +-15 deg: mass-weighted vertical FWHM
     (2.3548 sigma_z) and the peak rho (c253's moment method, restricted to bound gas);
  3. bound stream inside R_zone = 120: mass-weighted K/K0 (K = P/rho^Gamma);
  5. radial profiles of the bound stream, 2 M bins over r 60-200: mass, mass-weighted K/K0, peak rho, mass-weighted level.
    c260_rst.py RST OUT.npz [--nproc 96] [--maxblocks N (test)]
"""
import argparse, os, sys, time
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_dmde import header, c2p, frame_state

GAM, RC, K0 = 5 / 3, 3.24e-4, 1.2346626297e-2
DE = 0.677 / 16.4**2
EDGES = np.arange(-6.0, 6.0 + 1e-9, 1e-4)          # E/dE histogram bins
RB = np.arange(60.0, 200.0 + 1e-9, 2.0)            # radial profile bins
RADII = (50.0, 100.0, 150.0)
RZONE = 120.0
G = {}


def init(fn, nscal):
    t = header(fn)
    G.update(t=t, fn=fn, nscal=nscal)
    G['xbh'], G['xdot'] = frame_state(t['par'], t['time'])


def work(rng):
    t, nscal = G['t'], G['nscal']
    nmb, root = t['nmb'], t['root']
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    nx, ng = 16, 4
    no = nx + 2 * ng
    nmhd = 5 + nscal
    ncc = no**3
    bsz = (no + 1) * no * no * 3
    mm = np.memmap(G['fn'], dtype=np.float64, mode='r', offset=t['data_off'], shape=(nmb, t['dsize'] // 8))
    ll, lev = t['ll'], t['ll'][:, 3]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min])
    lo = np.array([x1min, x2min, x3min])
    s = slice(ng, ng + nx)
    ic = np.arange(nx) + 0.5
    xbh, xdot = G['xbh'], G['xdot']
    hE = np.zeros(len(EDGES) - 1)
    Mtot = Mb = 0.0
    zK = np.zeros(2)                                   # [sum dM K/K0, sum dM] bound stream r < RZONE
    pM, pK, pL = (np.zeros(len(RB) - 1) for _ in range(3))
    pR = np.zeros(len(RB) - 1)
    ann = {R: [] for R in RADII}
    for b in range(*rng):
        rec = np.asarray(mm[b])
        cons = rec[:nmhd * ncc].reshape(nmhd, no, no, no)[:, s, s, s]
        if not (cons[0] > 1e-16).any():
            continue
        adm = rec[nmhd * ncc + bsz:nmhd * ncc + bsz + 17 * ncc].reshape(17, no, no, no)[:, s, s, s]
        ext = w / 2.0**lev[b]
        dx = ext / nx
        x = lo[0] + ll[b, 0] * ext[0] + ic * dx[0] - xbh[0]
        y = lo[1] + ll[b, 1] * ext[1] + ic * dx[1] - xbh[1]
        z = lo[2] + ll[b, 2] * ext[2] + ic * dx[2] - xbh[2]
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        r = np.sqrt(X * X + Y * Y + Z * Z)
        gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
        det = (gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz))
        sg = np.sqrt(det)
        D = cons[0] / sg
        ok = (r >= 3.0) & (D > 1e-16) & np.isfinite(cons[4])
        if not ok.any():
            continue
        ixx = (gyy * gzz - gyz * gyz) / det; ixy = (gxz * gyz - gxy * gzz) / det
        ixz = (gxy * gyz - gxz * gyy) / det; iyy = (gxx * gzz - gxz * gxz) / det
        iyz = (gxy * gxz - gxx * gyz) / det; izz = (gxx * gyy - gxy * gxy) / det
        Sx, Sy, Sz = cons[1] / sg, cons[2] / sg, cons[3] / sg
        Ssq = ixx * Sx * Sx + iyy * Sy * Sy + izz * Sz * Sz + 2 * (ixy * Sx * Sy + ixz * Sx * Sz + iyz * Sy * Sz)
        Dk, tau = D[ok], np.maximum(cons[4][ok] / sg[ok], 0.0)
        rho, P, W, h = c2p(Dk, Ssq[ok], tau)
        al = adm[13][ok]
        ut = al * W - ((adm[14][ok] - xdot[0]) * Sx[ok] + (adm[15][ok] - xdot[1]) * Sy[ok]
                       + (adm[16][ok] - xdot[2]) * Sz[ok]) / (Dk * h)
        E = ut - 1.0
        dM = cons[0][ok] * dx.prod()
        Mtot += dM.sum()
        bnd = E < 0
        Mb += dM[bnd].sum()
        hE += np.histogram(E / DE, bins=EDGES, weights=dM)[0]
        st = bnd & (rho >= 1e-6 * RC) & (rho < 0.01 * RC)
        if not st.any():
            continue
        rr, K = r[ok][st], P[st] / rho[st]**GAM / K0
        m, rs = dM[st], rho[st]
        inz = rr < RZONE
        zK += (np.sum(m[inz] * K[inz]), np.sum(m[inz]))
        k = np.digitize(rr, RB) - 1
        v = (k >= 0) & (k < len(RB) - 1)
        np.add.at(pM, k[v], m[v]); np.add.at(pK, k[v], m[v] * K[v]); np.add.at(pL, k[v], m[v] * (lev[b] - root))
        np.maximum.at(pR, k[v], rs[v])
        for R in RADII:
            a = np.abs(rr - R) < 1.0
            if a.any():
                ann[R].append(np.stack([np.degrees(np.arctan2(Y[ok][st][a], X[ok][st][a])) % 360.0,
                                        Z[ok][st][a], rs[a], m[a]]))
    ann = {R: (np.concatenate(v, axis=1) if v else np.zeros((4, 0))) for R, v in ann.items()}
    return hE, Mtot, Mb, zK, pM, pK, pL, pR, ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rst'); ap.add_argument('out')
    ap.add_argument('--nproc', type=int, default=96); ap.add_argument('--nscal', type=int, default=1)
    ap.add_argument('--maxblocks', type=int, default=0); ap.add_argument('--first', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    t = header(a.rst)
    no, nx = 24, 16
    exp = ((5 + a.nscal) * no**3 + (no + 1) * no * no * 3 + 17 * no**3) * 8
    if t['dsize'] < exp or (t['dsize'] - exp) % (no**3 * 8):
        raise SystemExit(f'data_size {t["dsize"]} != {exp} + k x {no**3 * 8} (nscal={a.nscal}?)')
    nmb = min(t['nmb'] - a.first, a.maxblocks) if a.maxblocks else t['nmb'] - a.first
    ch = max(1, nmb // (a.nproc * 8))
    rngs = [(i, min(i + ch, a.first + nmb)) for i in range(a.first, a.first + nmb, ch)]
    with mp.Pool(a.nproc, initializer=init, initargs=(a.rst, a.nscal)) as pool:
        res = pool.map(work, rngs, chunksize=1)
    hE = sum(r[0] for r in res); Mtot = sum(r[1] for r in res); Mb = sum(r[2] for r in res)
    zK = sum(r[3] for r in res)
    pM, pK, pL = (sum(r[i] for r in res) for i in (4, 5, 6))
    pR = np.max([r[7] for r in res], axis=0)
    ann = {R: np.concatenate([r[8][R] for r in res], axis=1) for R in RADII}
    # 1. quantiles of the bound set
    eb = EDGES[:-1] + 5e-5
    hb = np.where(eb < 0, hE, 0.0)
    cb = np.cumsum(hb) / hb.sum()
    q = {p: float(np.interp(p, cb, eb)) for p in (0.1, 0.5, 0.9)}
    out = dict(t=t['time'], cycle=t['ncycle'], nmb=t['nmb'], Mtot=Mtot, Mb=Mb, fb=Mb / Mtot,
               q10=q[0.1], q50=q[0.5], q90=q[0.9], hE=hE, edges=EDGES,
               Kzone=zK[0] / max(zK[1], 1e-300), Mzone=zK[1], rb=RB, pM=pM, pK=pK, pL=pL, pR=pR)
    print(f'{a.rst}: t={t["time"]:.3f} nmb {t["nmb"]} (blocks read {nmb}) M {Mtot:.6e} f_bound {Mb / Mtot:.6f} '
          f'E/dE q10/50/90 {q[0.1]:+.4f} {q[0.5]:+.4f} {q[0.9]:+.4f}  K/K0(bound stream, r<{RZONE:.0f}) {out["Kzone"]:.4f} '
          f'(M {zK[1]:.4e})')
    for R in RADII:
        phi, zz, rs, m = ann[R]
        if m.size == 0:
            print(f'  R={R:.0f}: no bound stream'); continue
        hb_, _ = np.histogram(phi, bins=180, range=(0, 360), weights=m)
        p0 = (np.argmax(hb_) + 0.5) * 2.0
        sel = np.abs((phi - p0 + 180.0) % 360.0 - 180.0) <= 15.0
        mm_, zz_ = m[sel], zz[sel]
        zc = np.sum(mm_ * zz_) / mm_.sum()
        fw = 2.3548 * np.sqrt(np.sum(mm_ * (zz_ - zc)**2) / mm_.sum())
        out.update({f'phi{R:.0f}': p0, f'fwhm{R:.0f}': fw, f'peak{R:.0f}': rs[sel].max(), f'm{R:.0f}': mm_.sum()})
        print(f'  R={R:.0f}: phi {p0:6.1f} deg  FWHM_z {fw:.4f}  peak rho {rs[sel].max():.4e}  M(+-15deg) {mm_.sum():.4e}')
    np.savez(a.out, **out)
    print(f'  wall {time.time() - t0:.0f} s, {a.nproc} procs')


if __name__ == '__main__':
    main()
