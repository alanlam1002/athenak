#!/usr/bin/env python3
"""c254 (R-254 items 2-3): test-particle return rate of the debris of a partial disruption.
Input: c254tp_extract.py's npz. G = c = M_BH = 1.
Field: Schwarzschild geodesics, written for the Cartesian embedding x = r n of the areal radius r, with proper time tau:
    d2x/dtau2 = -(1/r^3)(1 + 3 L^2/r^2) x  (+ a_rem),   L^2 = |x x dx/dtau|^2,
which gives r'' = -1/r^2 + L^2/r^3 - 3L^2/r^4 and conserved L (the Schwarzschild orbit equations); coordinate time from the
normalisation, dt/dtau = u^t = sqrt[(1 + (u^r)^2/f + |p_t|^2)/f], f = 1 - 2/r, u^r = n.p, p_t = p - u^r n.
Remnant: a point mass M_rem on its own (unperturbed) geodesic from its mean (eps, L), position x_rem(t) tabulated in coordinate t;
its pull on a particle is Newtonian and Plummer-softened, a_rem = -M_rem d/(|d|^2 + h^2)^(3/2), d = x - x_rem(t_particle)
(a weak-field perturbation, |a_rem| << the hole's at these r; ARGUED adequate since M_rem/M_BH ~ 1e-5).
  --soft hill: h = r_H(t) = |x_rem(t)| (M_rem/3)^(1/3) (grows with r);  --soft rrem: h = R90 (fixed);  --soft none: M_rem = 0
  (the ballistic, frozen-in reference).
Initial state per particle from the cell's conserved (eps, L-vector) at r_areal = r_iso (1 + 1/(2 r_iso))^2 along the same n, u^r =
s sqrt(eps^2 - f (1 + L^2/r^2)) (0 if negative), p_t = (L x n)/r.
Events (first one ends the particle): return = first inward crossing of R_ret after having been outside it (time interpolated
linearly in t); captured by the remnant = entering |d| < r_H(t) (only for particles starting outside it); swallowed = r < 2.5;
escaped = r > 3000 with u^r > 0 and eps_now = f u^t > 1; still out at t_end.
Per-particle RK4 with its own step dtau = eta min(r^1.5, r/|p|, (|d|^2+h^2)^0.75/sqrt(M_rem), |d|/|p - v_rem|) (vectorised).
    c254tp_integrate.py IC.npz OUT.npz --rp RP [--N 200000] [--seed 1] [--soft hill|rrem|none] [--tend_tmb 10] [--eta 0.01]"""
import argparse, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('ic'); ap.add_argument('out'); ap.add_argument('--rp', type=float, required=True)
ap.add_argument('--N', type=int, default=200000); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--soft', default='hill'); ap.add_argument('--tend_tmb', type=float, default=10.0)
ap.add_argument('--eta', type=float, default=0.01); ap.add_argument('--ewin', default='')  # 'lo,hi' in E/dE: sample only these cells (importance sampling of the gap)
ap.add_argument('--tp', type=float, required=True)
a = ap.parse_args()
Z = np.load(a.ic)
DE = 0.677/16.4**2
t0 = float(Z['t']); Rret = 3.0*a.rp
Mrem = 0.0 if a.soft == 'none' else float(Z['rem_M'])
q3 = (float(Z['rem_M'])/3.0)**(1.0/3.0)


def areal(X):
    ri = np.linalg.norm(X, axis=0)
    return X*((1.0 + 0.5/ri)**2)


def init(X, eps, L, s):
    x = areal(X); r = np.linalg.norm(x, axis=0); n = x/r
    f = 1.0 - 2.0/r
    L2 = (L*L).sum(0)
    ur = s*np.sqrt(np.maximum(eps**2 - f*(1.0 + L2/r**2), 0.0))
    pt = np.cross(L.T, n.T).T/r
    return x, ur*n + pt


def acc_bh(x, p):
    r2 = (x*x).sum(0); r = np.sqrt(r2)
    Lv = np.cross(x.T, p.T).T; L2 = (Lv*Lv).sum(0)
    return -(1.0 + 3.0*L2/r2)*x/(r2*r)


def ut_of(x, p):
    r = np.linalg.norm(x, axis=0); f = 1.0 - 2.0/r
    ur = (x*p).sum(0)/r
    pt2 = (p*p).sum(0) - ur*ur
    return np.sqrt((1.0 + ur*ur/f + pt2)/f), ur, r, f


# --- the remnant's geodesic, tabulated in coordinate time
xr, pr = init(Z['rem_x'][:, None], np.array([Z['rem_eps']]), Z['rem_L'][:, None], np.array([Z['rem_s']]))
E_all = Z['E']; w_all = Z['dM']
bound = E_all < 0
Emb = np.quantile(E_all[bound], 0.005) if bound.any() else -DE   # the most-bound edge (0.5% mass quantile of the bound debris)
w_q = None
tmb = 2*np.pi/(2*abs(Emb))**1.5          # from the debris' most-bound edge (reported)
tmb_ref = 2*np.pi/(2*DE)**1.5             # canonical (E = -dE): 1.76e4 M, sets the integration window
t_end = a.tp + a.tend_tmb*tmb_ref
tt, xx = [t0], [xr[:, 0].copy()]
t = t0
while t < t_end + 10:
    ut, ur, r, f = ut_of(xr, pr)
    h = 0.002*min(r[0]**1.5, r[0]/np.linalg.norm(pr))
    def der(x, p):
        u, *_ = ut_of(x, p)
        return p, acc_bh(x, p), u
    k1 = der(xr, pr); k2 = der(xr + 0.5*h*k1[0], pr + 0.5*h*k1[1]); k3 = der(xr + 0.5*h*k2[0], pr + 0.5*h*k2[1])
    k4 = der(xr + h*k3[0], pr + h*k3[1])
    xr = xr + h/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]); pr = pr + h/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    t += h/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])[0]
    tt.append(t); xx.append(xr[:, 0].copy())
    if np.linalg.norm(xr) < 2.5:
        print('remnant swallowed at t =', t); break
tt = np.array(tt); xx = np.array(xx).T
vv = np.gradient(xx, tt, axis=1)


def xrem(t):
    return np.stack([np.interp(t, tt, xx[k]) for k in range(3)]), np.stack([np.interp(t, tt, vv[k]) for k in range(3)])


def soft(xrm):
    if a.soft == 'rrem':
        return float(Z['rem_R90'])
    return np.linalg.norm(xrm, axis=0)*q3


# --- particles: mass-weighted sample of the debris cells
rng = np.random.default_rng(a.seed)
win = np.ones(len(w_all), bool)
if a.ewin:
    lo, hi = (float(v) for v in a.ewin.split(','))
    win = (E_all/DE >= lo) & (E_all/DE < hi)
w_s = np.where(win, w_all, 0.0)
pw = w_s/w_s.sum()
idx = rng.choice(len(w_all), size=a.N, p=pw)
W = w_s.sum()/a.N   # each particle's mass (the window's mass / N with --ewin)
x, p = init(Z['X'][:, idx], Z['eps'][idx], Z['L'][:, idx], Z['s'][idx])
tp_ = np.full(a.N, t0)
E0 = E_all[idx]
xrm0, _ = xrem(tp_)
d0 = np.linalg.norm(x - xrm0, axis=0)
out0 = d0 > soft(xrm0) if a.soft != 'rrem' else d0 > np.linalg.norm(xrm0, axis=0)*q3
r0 = np.linalg.norm(x, axis=0)
been_out = r0 > Rret
status = np.zeros(a.N, int)       # 0 active, 1 returned, 2 remnant, 3 swallowed, 4 escaped, 5 t_end
t_ev = np.full(a.N, np.nan)


def deriv(x, p, t):
    u, *_ = ut_of(x, p)
    acc = acc_bh(x, p)
    if Mrem > 0:
        xrm, _ = xrem(t)
        d = x - xrm; h = soft(xrm)
        acc = acc - Mrem*d/((d*d).sum(0) + h*h)**1.5
    return p, acc, u


it = 0
while True:
    act = np.nonzero(status == 0)[0]
    if len(act) == 0:
        break
    xa, pa, ta = x[:, act], p[:, act], tp_[act]
    r = np.linalg.norm(xa, axis=0); pn = np.linalg.norm(pa, axis=0)
    h = np.minimum(r**1.5, r/np.maximum(pn, 1e-12))
    if Mrem > 0:
        xrm, vrm = xrem(ta); d = np.linalg.norm(xa - xrm, axis=0); hs = soft(xrm)
        h = np.minimum(h, ((d*d + hs*hs)**0.75)/np.sqrt(Mrem))
        h = np.minimum(h, np.maximum(d, hs)/np.maximum(np.linalg.norm(pa - vrm, axis=0), 1e-12))
    h = a.eta*h
    k1 = deriv(xa, pa, ta)
    k2 = deriv(xa + 0.5*h*k1[0], pa + 0.5*h*k1[1], ta + 0.5*h*k1[2])
    k3 = deriv(xa + 0.5*h*k2[0], pa + 0.5*h*k2[1], ta + 0.5*h*k2[2])
    k4 = deriv(xa + h*k3[0], pa + h*k3[1], ta + h*k3[2])
    xn = xa + h/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    pn_ = pa + h/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    tn = ta + h/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    ut, ur, rn, f = ut_of(xn, pn_)
    st = np.zeros(len(act), int); tev = np.full(len(act), np.nan)
    bo = been_out[act]
    ret = bo & (rn < Rret) & (r >= Rret)
    tev[ret] = ta[ret] + (tn[ret] - ta[ret])*(r[ret] - Rret)/(r[ret] - rn[ret]); st[ret] = 1
    if Mrem > 0:
        xrm, _ = xrem(tn); dn = np.linalg.norm(xn - xrm, axis=0)
        cap = (st == 0) & out0[act] & (dn < np.linalg.norm(xrm, axis=0)*q3)
        st[cap] = 2; tev[cap] = tn[cap]
    sw = (st == 0) & (rn < 2.5); st[sw] = 3; tev[sw] = tn[sw]
    es = (st == 0) & (rn > 3000) & (ur > 0) & (f*ut > 1.0); st[es] = 4; tev[es] = tn[es]
    te = (st == 0) & (tn > t_end); st[te] = 5
    x[:, act] = xn; p[:, act] = pn_; tp_[act] = tn
    been_out[act] = bo | (rn > Rret)
    status[act] = st; t_ev[act] = tev
    it += 1
    if it % 2000 == 0:
        print(f'it {it}: active {np.sum(status == 0)}, returned {np.sum(status == 1)}, remnant {np.sum(status == 2)}, '
              f'min t active {tp_[status == 0].min() if np.any(status == 0) else 0:.0f}', flush=True)
np.savez(a.out, t_ev=t_ev, status=status, W=W, E0=E0, idx=idx, tmb=tmb, tmb_ref=tmb_ref, Emb=Emb, t0=t0, tp=a.tp, Rret=Rret, soft=a.soft,
         N=a.N, seed=a.seed, Mrem=Mrem, rem_t=tt, rem_x=xx, M_debris=w_all.sum(), M_window=w_s.sum(), ewin=a.ewin, iters=it)
lab = {1: 'returned', 2: 'captured by remnant', 3: 'swallowed', 4: 'escaped', 5: 'still out at t_end'}
print(f'{a.ic} soft={a.soft} N={a.N} seed={a.seed}: t_mb={tmb:.4g} (E_mb/dE={Emb/DE:.3f}), iters {it}; ' +
      ', '.join(f'{lab[k]} {W*np.sum(status == k)/7.47e-5:.4g} M*' for k in lab))
