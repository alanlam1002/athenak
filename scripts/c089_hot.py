#!/usr/bin/env python3
"""C-089 (R-092 item 4 = R-088 Q2 tests 1-2, R-089, R-090 Q2): per-cell census of the sub-switch gas
from 3D restarts. Stage 1 (this file, --extract): per restart, every interior cell in the C-086 band
(1e-9 rho_c < D <= dual_energy_rho_switch, r_BH >= 3; c086_rst.py:177-185) is written to a cache npz
with the fields below. Stage 2 (c089_tables.py) tabulates.

Readers and definitions reused (not re-derived):
  - restart layout, header, frame (x_BH, xidot), ideal-gas C2P:   tde_dmde.header/frame_state/c2p
    (scripts/tde_dmde.py:37-76, 79-104); BH-frame Killing E = alpha W - (beta' - xidot).u_i - 1 with
    u_i = S_i/(D h) (tde_dmde.py:162-167);
  - the dual-energy shock-flag replica: c086_rst.flags + the block arithmetic of c086_rst.block
    (scripts/c086_rst.py:43-100), itself c049_divflag.block (validated against the stored in-code
    de_shock, C-049 / C-086 item 8); deck values from c049_divflag.deck (c049_divflag.py:37-46).
New per-cell fields (all on the cell's own MeshBlock including its 4 ghost layers):
  KK0   = P / (K0 rho^Gamma)                     (the entropy excess of C-086 item 8 / R-089)
  PoR   = P / rho                                 (P/rho of the restart's conserved state)
  L     = |(x - x_BH) x u_i|                      (as the zone's |L|, primitive_solver_hyd.hpp:2263-2265)
  vr    = u_i (x - x_BH)^i / |x - x_BH|           (sign: infall < 0; coordinate)
  psiv  = max_a |v^a(+) - 2 v^a + v^a(-)|         (the sensor's psi_v, c086_rst.py:84, raw, NOT / c_s)
  dv2   = max_a |v^a(+) - v^a(-)|                 (two-cell velocity jump, the kink amplitude)
  C     = -div v dx                               (coordinate divergence, c086_rst.py:83)
  glnr  = max_a |ln rho(+) - ln rho(-)| / 2        (|grad ln rho| dx)
  d_atm = Euclidean distance (cells) to the nearest cell with D <= 1e-9 rho_c (outside the band;
          the stream-atmosphere boundary); inf if none within the block + ghosts (>= 4-8 cells)
  d_flr = ditto to a density-floored cell (rho < dthreshold * dfloor: these always fail the FOFC probe,
          reset_floor.hpp:37 + primitive_solver_hyd.hpp:1890-1895, 2127-2140)
  d_tfl = ditto to a cell at the temperature floor (P/rho < TFL_PROXY; fails the probe through
          reset_floor.hpp:47 unless fofc_cold_recovery rescues it)
  d_tflb= ditto to a temperature-floor cell inside the band (D > 1e-9 rho_c): floored stream gas
  d_lev = distance (cells, Chebyshev to the face/edge/corner region) to a neighbour MeshBlock at a
          different refinement level (from the restart's logical-location tree)
  lev   = MeshBlock level above root; flag = the replica's final shock flag; dM = cons(IDN) dV.

    c089_hot.py --jobs LABEL:RUNDIR:RSTNUM[,..] [--nproc N] [--cache DIR]
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import c049_divflag as F                          # noqa: E402
import c086_rst as R86                            # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC, K0, GAM = 3.24e-4, 1.2346626297e-2, 5.0 / 3.0
LO = 1e-9 * RHOC
TFL_PROXY = 1e-14      # P/rho below this = at the temperature floor (tfloor = 1e-22; see c089_tables.py (c))
NX, NG = 16, 4
NO = NX + 2 * NG


def deck_floor(par):
    import re
    g = lambda k, d: float(re.search(rf'<mhd>(?:(?!<).)*?^\s*{k}\s*=\s*(\S+)', par, re.S | re.M).group(1)) \
        if re.search(rf'<mhd>(?:(?!<).)*?^\s*{k}\s*=\s*(\S+)', par, re.S | re.M) else d
    return g('dfloor', 0.0), g('tfloor', 0.0), g('dthreshold', 1.0)


def level_dirs(ll):
    """per block: list of (dx, dy, dz) neighbour directions whose neighbour is at a different level"""
    have = {(int(a), int(b), int(c), int(l)) for a, b, c, l in ll}
    out = []
    dirs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    for x, y, z, l in ll:
        n = 1 << int(l)
        lst = []
        for d in dirs:
            q = (x + d[0], y + d[1], z + d[2])
            if min(q) < 0 or max(q) >= n:
                continue                                   # domain boundary
            if (q[0], q[1], q[2], int(l)) in have:
                continue                                   # same level
            lst.append(d)                                  # coarser (parent leaf) or finer (children)
        out.append(lst)
    return out


IC = np.arange(NX)
# distance (in own cells) from each interior cell centre to the block face on the - / + side
DM_, DP_ = IC + 0.5, NX - 0.5 - IC


def dlev_block(dirs):
    if not dirs:
        return np.full((NX, NX, NX), np.inf, np.float32)
    best = np.full((NX, NX, NX), np.inf)
    for d in dirs:
        dd = np.zeros((NX, NX, NX))
        for ax, s in enumerate(d):                         # ax 0 = x (array axis 2)
            if s == 0:
                continue
            v = DM_ if s < 0 else DP_
            shp = [1, 1, 1]; shp[2 - ax] = NX
            dd = np.maximum(dd, v.reshape(shp))
        best = np.minimum(best, dd)
    return best.astype(np.float32)


def edt(mask):
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return distance_transform_edt(~mask)


def block(rec, nscal, dx, p, xbh, xdot, bmin, flr, dirs_b):
    nmhd = 5 + nscal; ncc = NO**3; bsz = (NO + 1) * NO * NO * 3
    cons = rec[:nmhd * ncc].reshape(nmhd, NO, NO, NO)
    adm = rec[nmhd * ncc + bsz:].reshape(17, NO, NO, NO)
    gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
    det = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz)
    sg = np.sqrt(np.maximum(det, 1e-300))
    I = slice(NG, NG + NX)
    Dd = cons[0] / sg
    if not (Dd[I, I, I] > LO).any():
        return None
    # --- as c086_rst.block (c086_rst.py:61-77)
    ixx = (gyy * gzz - gyz * gyz) / det; ixy = (gxz * gyz - gxy * gzz) / det
    ixz = (gxy * gyz - gxz * gyy) / det; iyy = (gxx * gzz - gxz * gxz) / det
    iyz = (gxy * gxz - gxx * gyz) / det; izz = (gxx * gyy - gxy * gxy) / det
    S = [cons[1] / sg, cons[2] / sg, cons[3] / sg]; tau = cons[4] / sg
    ok = (Dd > 0) & np.isfinite(tau)
    Dc = np.where(ok, Dd, 1e-30); tc = np.where(ok, np.maximum(tau, 0), 0)
    gi = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
    Ssq = sum(gi[i][j] * S[i] * S[j] for i in range(3) for j in range(3))
    Ssq = np.where(ok, Ssq, 0)
    rho, P, W, h = D.c2p(Dc, Ssq, tc)
    v = [np.where(ok, sum(gi[i][j] * S[j] for j in range(3)) / (Dc * h * W), 0) for i in range(3)]
    ud = [np.where(ok, S[i] / (Dc * h), 0) for i in range(3)]
    gam = p['gam']
    rr = np.maximum(rho, 1e-300)
    cs = np.sqrt(np.maximum(gam * P / (rr * (1 + gam / (gam - 1) * P / rr)), 1e-300))
    R = slice(1, NO - 1)
    def sh(a, q, s):
        sl = [R, R, R]; sl[2 - a] = slice(1 + s, NO - 1 + s)
        return q[tuple(sl)]
    divc = sum((sh(a, v[a], 1) - sh(a, v[a], -1)) / (2 * dx) for a in range(3))
    psiv = np.max([np.abs(sh(a, v[a], 1) - 2 * sh(a, v[a], 0) + sh(a, v[a], -1)) for a in range(3)], axis=0)
    dv2 = np.max([np.abs(sh(a, v[a], 1) - sh(a, v[a], -1)) for a in range(3)], axis=0)
    lr = np.log(rr)
    glnr = np.max([np.abs(sh(a, lr, 1) - sh(a, lr, -1)) / 2 for a in range(3)], axis=0)
    Pp = [(sh(a, P, 1), sh(a, P, 0), sh(a, P, -1)) for a in range(3)]
    psip = np.max([np.abs(pp - 2 * p0 + pm) / np.maximum(pp + 2 * p0 + pm, 1e-300) for pp, p0, pm in Pp], axis=0)
    f_act = R86.flags(psiv, psip, divc, cs[R, R, R], p)
    J = slice(NG - 1, NG - 1 + NX)
    # --- new
    dfloor, tfloor, dthr = flr
    m_atm = Dd <= LO
    m_flr = rho < dthr * dfloor
    m_tfl = (P / rr < TFL_PROXY) & ~m_flr
    m_tflb = m_tfl & (Dd > LO)
    d_atm, d_flr, d_tfl, d_tflb = (edt(mm)[I, I, I] for mm in (m_atm, m_flr, m_tfl, m_tflb))
    xc = bmin[0] + (IC + 0.5) * dx; yc = bmin[1] + (IC + 0.5) * dx; zc = bmin[2] + (IC + 0.5) * dx
    Z, Y, X = np.meshgrid(zc, yc, xc, indexing='ij')
    rx, ry, rz = X - xbh[0], Y - xbh[1], Z - xbh[2]
    rbh = np.sqrt(rx * rx + ry * ry + rz * rz)
    ui = [u[I, I, I] for u in ud]
    al = adm[13][I, I, I]
    bx, by, bz = adm[14][I, I, I] - xdot[0], adm[15][I, I, I] - xdot[1], adm[16][I, I, I] - xdot[2]
    E = al * W[I, I, I] - (bx * ui[0] + by * ui[1] + bz * ui[2]) - 1.0     # tde_dmde.py:164-167
    lx = ry * ui[2] - rz * ui[1]; ly = rz * ui[0] - rx * ui[2]; lz = rx * ui[1] - ry * ui[0]
    L = np.sqrt(lx * lx + ly * ly + lz * lz)
    vr = (rx * ui[0] + ry * ui[1] + rz * ui[2]) / np.maximum(rbh, 1e-30)
    Di = Dd[I, I, I]
    sel = (Di > LO) & (Di <= p['rho_sw']) & (rbh >= 3.0)
    if not sel.any():
        return None
    rI, PI = rr[I, I, I], P[I, I, I]
    out = dict(dM=cons[0][I, I, I] * dx**3, D=Di, KK0=PI / (K0 * rI**GAM), PoR=PI / rI, rbh=rbh, E=E, L=L, vr=vr,
               psiv=psiv[J, J, J], dv2=dv2[J, J, J], C=-divc[J, J, J] * dx, glnr=glnr[J, J, J],
               d_atm=d_atm, d_flr=d_flr, d_tfl=d_tfl, d_tflb=d_tflb, flag=f_act, cs=cs[I, I, I], d_lev=dlev_block(dirs_b))
    return {k: np.asarray(a)[sel] for k, a in out.items()}


def census(rec, nscal, flr):
    """whole-mesh interior census: (n, M) of density-floored (rho < dthreshold dfloor), temperature-floored
    (P/rho < TFL_PROXY, not density-floored) and other cells"""
    nmhd = 5 + nscal; ncc = NO**3; bsz = (NO + 1) * NO * NO * 3
    I = slice(NG, NG + NX)
    cons = rec[:nmhd * ncc].reshape(nmhd, NO, NO, NO)[:, I, I, I]
    adm = rec[nmhd * ncc + bsz:].reshape(17, NO, NO, NO)[:, I, I, I]
    gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
    det = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz)
    sg = np.sqrt(np.maximum(det, 1e-300))
    ixx = (gyy * gzz - gyz * gyz) / det; ixy = (gxz * gyz - gxy * gzz) / det
    ixz = (gxy * gyz - gxz * gyy) / det; iyy = (gxx * gzz - gxz * gxz) / det
    iyz = (gxy * gxz - gxx * gyz) / det; izz = (gxx * gyy - gxy * gxy) / det
    Dd = cons[0] / sg; S = [cons[1] / sg, cons[2] / sg, cons[3] / sg]; tau = cons[4] / sg
    gi = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
    Ssq = sum(gi[i][j] * S[i] * S[j] for i in range(3) for j in range(3))
    ok = (Dd > 0) & np.isfinite(tau)
    rho, P, W, h = D.c2p(np.where(ok, Dd, 1e-30), np.where(ok, Ssq, 0), np.where(ok, np.maximum(tau, 0), 0))
    dfloor, tfloor, dthr = flr
    a = rho < dthr * dfloor
    b = (P / np.maximum(rho, 1e-300) < TFL_PROXY) & ~a
    c = ~a & ~b
    dM = cons[0]
    return np.array([a.sum(), b.sum(), c.sum(), dM[a].sum(), dM[b].sum(), dM[c].sum()])


def one(job):
    lab, rdir, num, cache, do_census = job
    fn = f'{C}/{rdir}/rst/{B}.{num}.rst'
    t = D.header(fn)
    p = F.deck(t['par'])
    flr = deck_floor(t['par'])
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(t['nmb'], t['dsize'] // 8))
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min]); lo = np.array([x1min, x2min, x3min])
    lev = t['ll'][:, 3]
    xbh, xdot = D.frame_state(t['par'], t['time'])
    dirs = level_dirs(t['ll'])
    acc = []
    cen = np.zeros(6)
    for b in range(t['nmb']):
        ext = w / 2.0**lev[b]; dx = ext[0] / NX
        bmin = lo + t['ll'][b, :3] * ext
        if do_census:
            cen += census(mm[b], 1, flr) * np.array([1, 1, 1, dx**3, dx**3, dx**3])
        o = block(mm[b], 1, dx, p, xbh, xdot, bmin, flr, dirs[b])
        if o is None:
            continue
        o['lev'] = np.full(o['dM'].shape, lev[b] - t['root'])
        acc.append(o)
    cols = {}
    for k in (acc[0] if acc else {}):
        dt = bool if k == 'flag' else (np.int8 if k == 'lev' else (np.float64 if k == 'dM' else np.float32))
        cols[k] = np.concatenate([o[k] for o in acc]).astype(dt)
    os.makedirs(cache, exist_ok=True)
    out = f'{cache}/{lab}_{rdir}_{num}.npz'
    np.savez(out, t=t['time'], rdir=rdir, num=num, lab=lab, rho_sw=p['rho_sw'], dfloor=flr[0], tfloor=flr[1],
             dthr=flr[2], xbh=xbh, census=cen, **cols)
    return out, t['time'], len(cols.get('dM', []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True, help='LABEL:RUNDIR:RSTNUM,...')
    ap.add_argument('--nproc', type=int, default=12)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--census', action='store_true', help='whole-mesh floor census (FOFC reading)')
    a = ap.parse_args()
    jobs = []
    for s in a.jobs.split(','):
        lab, rdir, num = s.split(':')
        if not os.path.exists(f'{a.cache}/{lab}_{rdir}_{num}.npz'):
            jobs.append((lab, rdir, num, a.cache, a.census))
    with Pool(a.nproc) as pool:
        for r in pool.imap_unordered(one, jobs):
            print(*r, flush=True)


if __name__ == '__main__':
    main()
