#!/usr/bin/env python3
"""c269 (user 2026-11-18: "K-008f check on our own runs"): does our B.1-family ID start the star faster than the
designed orbit, as the spin session found for V5a (code/OUTBOX_KERR.md K-008f: dv/v ~ 5e-4, the star's own shift
and lapse)? Existing t = 0 outputs only (prim_xy + adm_xy, written after CFC::InitializeMetric).

Per run, over the z = 0 slice cells with rho >= 0.1 rho_c, D-weighted (D = sqrt(gamma) rho W, as K-008f):
  full: V = alpha u~/W - beta_BH with the run's own metric (gamma_ij = adm_g, alpha, beta_BH = adm_beta - xidot(0),
        xidot from user.hst row 0, as c086_slices.py);
  bg:   V0 = alpha0 u~/W0 - beta0 with the trumpet background alone (psi0^4 delta_ij, alpha0, beta0 at x - x_BH;
        tde_elliptical_orbit_ic.metric_at = the verbatim port of cfc_puncture.hpp);
  target: the designed coordinate velocity at the release point, V_t = alpha0 v_deck - beta0 (v_deck = the deck's
        star_vel, the background Eulerian velocity tde_parabolic_orbit_ic.py solves for).
Orbit consequence (argued: test-particle Killing invariants at the star's centre in the background, from V):
  v = (V + beta0)/alpha0, W = (1 - psi0^4 v^2)^-1/2, p_i = psi0^4 W v^i, E = alpha0 W - beta0.p, L = x p_y - y p_x;
  dE = E(V_full) - E(V_t), compared with the debris energy spread dE_t = R*/r_t^2 (tde_dmde.py:207).
    c269_k008f.py RUN_DIR [RUN_DIR ...]
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'vis', 'python'))
import bin_convert as bc                 # noqa: E402
import tde_elliptical_orbit_ic as ic     # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC = 3.24e-4
DE_T = 0.677 / 16.4**2                   # R*/r_t^2 (tde_dmde.py:207)


def deck(d, key):
    par = open(f'{d}/parfile.par').read()
    return float(re.search(rf'^{key}\s*=\s*(\S+)', par, re.M).group(1))


def kill(xr, V):
    """test-particle E, L in the background at the BH-relative point xr with coordinate velocity V"""
    ps0, al0, be0 = ic.metric_at(*xr)
    be0 = np.asarray(be0)
    v = (V + be0) / al0
    W = 1.0 / np.sqrt(1.0 - ps0**4 * v @ v)
    p = ps0**4 * W * v
    return al0 * W - be0 @ p, xr[0] * p[1] - xr[1] * p[0]


def one(d):
    f = lambda k: f'{d}/bin/{B}.{k}.00000.bin'
    p, a = bc.read_binary(f('prim_xy')), bc.read_binary(f('adm_xy'))
    g = np.asarray(p['mb_geometry'])
    assert np.array_equal(g, np.asarray(a['mb_geometry']))
    P_, A_ = p['mb_data'], a['mb_data']
    v = lambda dd, k: np.asarray(dd[k])[:, 0]
    rho = v(P_, 'dens')
    u = np.array([v(P_, 'velx'), v(P_, 'vely'), v(P_, 'velz')])
    gn = ['adm_gxx', 'adm_gxy', 'adm_gxz', 'adm_gyy', 'adm_gyz', 'adm_gzz']
    gxx, gxy, gxz, gyy, gyz, gzz = (v(A_, k) for k in gn)
    G = np.array([[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, gzz]])
    ul = np.einsum('ij...,j...->i...', G, u)
    W = np.sqrt(1 + (ul * u).sum(0))
    det = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz)
    h = np.loadtxt(f'{d}/{B}.user.hst')
    xdot, xi = h[0, 4:7], h[0, 7:10]
    xbh = np.array([deck(d, f'puncture_x{i}') for i in (1, 2, 3)]) - xi
    al = v(A_, 'adm_alpha')
    be = np.array([v(A_, 'adm_betax'), v(A_, 'adm_betay'), v(A_, 'adm_betaz')]) - xdot[:, None, None, None]
    nx = p['nx1_out_mb']
    dx = (g[:, 1] - g[:, 0]) / nx
    xc = np.broadcast_to(g[:, 0, None, None] + (np.arange(nx)[None, None, :] + .5) * dx[:, None, None], rho.shape)
    yc = np.broadcast_to(g[:, 2, None, None] + (np.arange(nx)[None, :, None] + .5) * dx[:, None, None], rho.shape)
    s = rho >= 0.1 * RHOC
    wD = (np.sqrt(det) * rho * W * (dx**2)[:, None, None])[s]
    Vf = (al * u / W - be)[:, s]
    # background at every selected cell
    X = np.stack([xc[s] - xbh[0], yc[s] - xbh[1], np.zeros(s.sum()) - xbh[2]])
    bg = [ic.metric_at(*X[:, n]) for n in range(X.shape[1])]
    ps0 = np.array([b[0] for b in bg]); al0 = np.array([b[1] for b in bg]); be0 = np.array([b[2] for b in bg]).T
    us = u[:, s]
    W0 = np.sqrt(1 + ps0**4 * (us * us).sum(0))
    Vb = al0 * us / W0 - be0
    Vfm = (Vf * wD).sum(1) / wD.sum(); Vbm = (Vb * wD).sum(1) / wD.sum()
    # the target at the release point (the deck's star centre)
    xs = np.array([deck(d, f'star_center_x{i}') for i in (1, 2, 3)])
    vdeck = np.array([deck(d, f'star_vel_x{i}') for i in (1, 2, 3)])
    ps, a0, b0 = ic.metric_at(*(xs - xbh))
    # star_vel_mode = coordinate (R-289 a): the deck's star_vel is already V_t (tde_parabolic_orbit_ic.py prints it)
    coord = re.search(r'^star_vel_mode\s*=\s*coordinate', open(f'{d}/parfile.par').read(), re.M) is not None
    Vt = vdeck if coord else a0 * vdeck - np.asarray(b0)
    # the dense centroid in the slice (for the invariants)
    xcm = np.array([(xc[s] * wD).sum(), (yc[s] * wD).sum(), 0.0]) / wD.sum()
    xr = xcm - xbh
    Et, Lt = kill(xs - xbh, Vt); Ef, Lf = kill(xr, Vfm); Eb, Lb = kill(xr, Vbm)
    n = np.linalg.norm(Vt)
    that = Vt / n
    return dict(run=os.path.basename(d), t=p['time'], ncell=int(s.sum()), xdot0=xdot, Vt=Vt, Vfull=Vfm, Vbg=Vbm,
                dv_full=(Vfm - Vt) @ that / n, dv_bg=(Vbm - Vt) @ that / n, dv_full_abs=np.linalg.norm(Vfm - Vt) / n,
                xdot_minus_Vt=np.linalg.norm(xdot - Vt) / n,
                E_t=Et, L_t=Lt, dE_full=Ef - Et, dL_full=Lf - Lt, dE_bg=Eb - Et, dE_over_spread=(Ef - Et) / DE_T,
                Ucore=float(np.average(1 - al[s] / al0, weights=wD)))


if __name__ == '__main__':
    np.set_printoptions(precision=6)
    for d in sys.argv[1:]:
        r = one(d)
        print(f"== {r['run']}  t={r['t']}  cells={r['ncell']}")
        print(f"   V_target {r['Vt']}  <V>_D full {r['Vfull']}  bg-only {r['Vbg']}  xidot(0) {r['xdot0']}")
        print(f"   along-track dv/v: full {r['dv_full']:+.3e}  bg-only {r['dv_bg']:+.3e}  |dV|/|V| full {r['dv_full_abs']:.3e}"
              f"  |xidot-V_t|/|V_t| {r['xdot_minus_Vt']:.3e}")
        print(f"   E_t {r['E_t']:.9f}  dE full {r['dE_full']:+.3e} (= {r['dE_over_spread']:+.3%} of R*/r_t^2)  dE bg {r['dE_bg']:+.3e}"
              f"  L_t {r['L_t']:.6f} dL full {r['dL_full']:+.3e}  <1-alpha/alpha0>_D {r['Ucore']:.3e}")
