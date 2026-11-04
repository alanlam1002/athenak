#!/usr/bin/env python3
"""c187 (R-223 item 1(a)): the eta1/eta2 calibration for D-ent, offline from energy-only B.1 dumps (Bb10, Bbr).
Per 3D dump (prim_3d, prim_3d_peri) and cell:
  q = e_int/tau = D eps / tau = rho W eps / (rho h W^2 - p - rho W)   (sqrt(g) cancels; the quantity psh's
      dual_energy_eta1 gate already uses: eps < eta1 tau/D), W = sqrt(1 + psi^4 |u~|^2) with u~ the dump's velocity
      (the normal-frame u^i) and psi^4 = g_xx from the nearest cell of the adm_xy slice at the nearest time (CFC:
      g_ij = psi^4 delta_ij; the star and stream lie near z = 0; argued: the psi error enters tau at <~1%).
  classes as c169: core rho >= 0.1 rho_c; outer 0.01-0.1 rho_c; remnant = inside r_H (Hill sphere about the rho-centroid
      of rho >= 0.01 rho_c); stream = outside r_H with 1e-6 <= rho/rho_c < 0.01. floored = T <= 1.0001 tfloor.
  K/K0 - 1 = p/(kappa rho^Gamma) - 1 (energy only: the energy equation's adiabat error; before t_p no shock).
Writes OUT_dumps.csv (per dump x class: mass, q quantiles, mass share below each eta) and OUT_bins.csv (per arm x phase
x class x log10 q_ad bin (q on the initial adiabat; not circular): mass, K/K0 - 1 quantiles, share |K/K0 - 1| > 1% and > 10%, floored share).
    c187_eta1_calib.py OUT_PREFIX LABEL=RUNDIR ...       (frameworks python: scipy)"""
import glob, sys, types
import numpy as np
from scipy.spatial import cKDTree
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh, par   # noqa: E402
GAM = 5.0/3.0
TP = 135.9
ETAS = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
QS = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
BINS = np.arange(-9.0, -1.49, 0.5)      # log10 q bin edges
CLS = ('core', 'outer', 'rem', 'stream')


def wq(a, w, qs):
    o = np.argsort(a); a, w = a[o], w[o]; c = np.cumsum(w); c /= c[-1]
    return [a[min(np.searchsorted(c, x), len(a) - 1)] for x in qs]


def cells(P):
    g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']; q = P['mb_data']
    cat = lambda k: np.concatenate([np.asarray(q[k][b]).ravel() for b in range(len(g))]).astype(float)
    h = (g[:, 1] - g[:, 0])/n
    ax = lambda lo: (lo[:, None] + (np.arange(n) + 0.5)[None, :]*h[:, None])
    X, Y, Z = ax(g[:, 0]), ax(g[:, 2]), ax(g[:, 4])
    nb = len(g)
    x = np.broadcast_to(X[:, None, None, :], (nb, n, n, n)).ravel()
    y = np.broadcast_to(Y[:, None, :, None], (nb, n, n, n)).ravel()
    z = np.broadcast_to(Z[:, :, None, None], (nb, n, n, n)).ravel()
    return cat, x, y, z, np.repeat(h**3, n**3)


def slice_tree(d):
    """adm_xy slices: time -> (cKDTree of (x, y), psi^4)"""
    out = []
    for f in sorted(glob.glob(f'{d}/bin/*.adm_xy.0*.bin')):
        P = bc.read_binary(f)
        g = np.asarray(P['mb_geometry'], float); nx, ny = P['nx1_out_mb'], P['nx2_out_mb']; q = P['mb_data']
        xs, ys, gx = [], [], []
        for b in range(len(g)):
            hx = (g[b, 1] - g[b, 0])/nx; hy = (g[b, 3] - g[b, 2])/ny
            xx = g[b, 0] + (np.arange(nx) + 0.5)*hx; yy = g[b, 2] + (np.arange(ny) + 0.5)*hy
            XX, YY = np.meshgrid(xx, yy)
            xs.append(XX.ravel()); ys.append(YY.ravel()); gx.append(np.asarray(q['adm_gxx'][b]).reshape(-1)[:nx*ny])
        out.append((P['time'], np.concatenate(xs), np.concatenate(ys), np.concatenate(gx)))
    return out


def main():
    pre = sys.argv[1]
    drows, brows = [], []
    for s in sys.argv[2:]:
        lab, d = s.split('=', 1)
        rhoc, k0, tfl, mbh = par(d, 'rhoc'), par(d, 'kappa'), par(d, 'tfloor'), par(d, 'puncture_mass', 1.0)
        sl = slice_tree(d); st_t = np.array([a[0] for a in sl]); trees = {}
        fs = sorted(glob.glob(f'{d}/bin/*.prim_3d.0*.bin') + glob.glob(f'{d}/bin/*.prim_3d_peri.0*.bin'),
                    key=lambda f: bc.read_binary(f)['time'])
        acc = {}
        for f in fs:
            P = bc.read_binary(f); t = P['time']
            cat, x, y, z, dv = cells(P)
            rho, p, T = cat('dens'), cat('press'), cat('temperature')
            u = np.stack([cat('velx'), cat('vely'), cat('velz')])
            i = int(np.argmin(np.abs(st_t - t)))
            if i not in trees:
                trees[i] = cKDTree(np.stack([sl[i][1], sl[i][2]], 1))
            _, j = trees[i].query(np.stack([x, y], 1))
            g4 = sl[i][3][j]
            W = np.sqrt(1.0 + g4*(u**2).sum(0))
            eps = p/((GAM - 1.0)*rho); hh = 1.0 + eps + p/rho
            tau = rho*hh*W*W - p - rho*W
            q = rho*W*eps/np.maximum(tau, 1e-300)
            # q_ad: the same ratio on the initial adiabat (p = kappa rho^Gamma): the conditioning of the flow itself,
            # independent of the energy equation's error (q is not: q ~ eps ~ K, so binning K by q is circular)
            pa = k0*rho**GAM; ea = pa/((GAM - 1.0)*rho); ta = rho*(1.0 + ea + pa/rho)*W*W - pa - rho*W
            qa = rho*W*ea/np.maximum(ta, 1e-300)
            K = p/rho**GAM/k0 - 1.0; fl = T <= 1.0001*tfl; m = rho*dv
            d01, d1 = rho >= 0.01*rhoc, rho >= 0.1*rhoc
            xc = np.array([(m[d01]*c[d01]).sum()/m[d01].sum() for c in (x, y, z)])
            xb = xbh(d, t); D = np.linalg.norm(xc - xb)
            rr = np.sqrt((x - xc[0])**2 + (y - xc[1])**2 + (z - xc[2])**2)
            mrem = m[d01].sum()
            for _ in range(3):
                rh = D*(mrem/(3.0*mbh))**(1.0/3.0); inh = rr < rh; mrem = m[inh].sum()
            sel = {'core': d1, 'outer': d01 & ~d1, 'rem': inh, 'stream': (~inh) & (rho >= 1e-6*rhoc) & (rho < 0.01*rhoc)}
            ph = 'pre' if t < TP - 1.0 else 'post'
            line = [f'{lab} t={t:6.1f} D={D:6.2f} W_med(core)={np.median(W[d1]) if d1.any() else np.nan:.4f}']
            for c in CLS:
                s_ = sel[c] & (tau > 0)
                M = m[s_].sum()
                r = dict(arm=lab, t=t, cls=c, M=M, ntau_le0=int((sel[c] & (tau <= 0)).sum()))
                if M > 0:
                    for qq, v in zip(QS, wq(q[s_], m[s_], QS)):
                        r[f'q{int(100*qq):02d}'] = v
                    for e in ETAS:
                        r[f'below_{e:.0e}'] = m[s_ & (q < e)].sum()/M
                    r['fl'] = m[s_ & fl].sum()/M
                    line.append(f"{c}: q50={r['q50']:.2e} q01={r['q01']:.2e} q99={r['q99']:.2e} fl={r['fl']:.3f}")
                    for qq, v in zip(QS, wq(qa[s_], m[s_], QS)):
                        r[f'qa{int(100*qq):02d}'] = v
                    for e in ETAS:
                        r[f'qa_below_{e:.0e}'] = m[s_ & (qa < e)].sum()/M
                    lq = np.log10(np.maximum(qa[s_], 1e-30)); b = np.digitize(lq, BINS)
                    for kb in range(1, len(BINS)):
                        mm = b == kb
                        if not mm.any():
                            continue
                        key = (lab, ph, c, kb)
                        a = acc.setdefault(key, [[], [], []])
                        a[0].append(K[s_][mm]); a[1].append(m[s_][mm]); a[2].append(fl[s_][mm])
                drows.append(r)
            print(' | '.join(line), flush=True)
        for (lab_, ph, c, kb), a in sorted(acc.items()):
            Kc, mc, fc = np.concatenate(a[0]), np.concatenate(a[1]), np.concatenate(a[2])
            r = dict(arm=lab_, phase=ph, cls=c, lo=BINS[kb - 1], hi=BINS[kb], M=mc.sum(), n=len(Kc))
            for qq, v in zip((0.05, 0.5, 0.95), wq(Kc, mc, (0.05, 0.5, 0.95))):
                r[f'K{int(100*qq):02d}'] = v
            r['gt1pc'] = mc[np.abs(Kc) > 0.01].sum()/mc.sum(); r['gt10pc'] = mc[np.abs(Kc) > 0.1].sum()/mc.sum()
            r['fl'] = mc[fc].sum()/mc.sum()
            brows.append(r)
    for nm, rows in (('dumps', drows), ('bins', brows)):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(f'{pre}_{nm}.csv', 'w') as fo:
            fo.write(','.join(keys) + '\n')
            for r in rows:
                fo.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
    print('wrote', pre + '_dumps.csv', pre + '_bins.csv')


if __name__ == '__main__':
    main()
