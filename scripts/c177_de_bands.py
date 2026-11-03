#!/usr/bin/env python3
"""c177 (R-209, B.1 Bd10 = Valencia + kcfc_bg + DE at the P10/P18 production settings): the dual-energy branch per
debris class, from the 3D DE diagnostic dumps (<output10> mhd_de_diag, dediag_3d) and the prim_3d dump at the same time.
Classes (as c169_b1_ana.py dumps): core rho >= 0.1 rho_c; outer dense layers 0.01 <= rho/rho_c < 0.1; the bound remnant
(Hill sphere about the rho >= 0.01 rho_c centroid, r_H = D (M_rem/3 M_BH)^(1/3), 3 fixed-point iterations); the stream
(1e-6 <= rho/rho_c < 0.01 outside r_H).
Per class (rho dV mass shares): the branch codes of the last C2P before the dump (de_branch; athenak wip/kbg
src/eos/primitive_solver_hyd.hpp:2544-2613): 0 below rho_switch (energy), 1 entropy (unflagged), 2 entropy by the
second-law guard (flagged, K_en <= K_tr (1 + tol)), 3 entropy after a floored / failed C2P, 4 energy (flagged, K_en above
the guard), 5 dense, C2P bad and the entropy inversion failed; the flagged share (de_shock != 0); and over the flagged
dense cells, the distribution of de_ken_ktr = K_en/K_tr - 1 (the energy solution's K relative to the tracer's,
:2547-2549; mass-weighted quantiles 5/25/50/75/95%) -- the guard decision's margin.
    c177_de_bands.py OUT.csv RUNDIR"""
import glob, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh, par   # noqa: E402


def wq(a, w, qs):
    if len(a) == 0:
        return [np.nan]*len(qs)
    o = np.argsort(a); c = np.cumsum(w[o]); c /= c[-1]
    return [float(a[o][np.searchsorted(c, q)]) for q in qs]


def main(out, d):
    rhoc, mbh = par(d, 'rhoc'), par(d, 'puncture_mass', 1.0)
    pf = {bc.read_binary(f)['time']: f for f in glob.glob(f'{d}/bin/*.prim_3d.0*.bin') + glob.glob(f'{d}/bin/*.prim_3d_peri.0*.bin')}
    pt = np.array(sorted(pf))
    rows = []
    for fd in sorted(glob.glob(f'{d}/bin/*.dediag_3d.*.bin')):
        Dg = bc.read_binary(fd); t = Dg['time']
        tp = pt[np.argmin(np.abs(pt - t))]
        P = bc.read_binary(pf[tp])
        g = np.asarray(Dg['mb_geometry'], float); n = Dg['nx1_out_mb']; gp = np.asarray(P['mb_geometry'], float)
        cat = lambda S, k, nb: np.concatenate([np.asarray(S['mb_data'][k][b]).ravel() for b in range(nb)]).astype(float)
        br = cat(Dg, 'de_branch', len(g)).round().astype(int); sh = cat(Dg, 'de_shock', len(g)); dk = cat(Dg, 'de_ken_ktr', len(g))
        h = (g[:, 1] - g[:, 0])/n
        ax = lambda G, hh, lo: (G[:, lo][:, None] + (np.arange(n) + 0.5)[None, :]*hh[:, None])
        X, Y, Z = ax(g, h, 0), ax(g, h, 2), ax(g, h, 4)
        x = np.broadcast_to(X[:, None, None, :], (len(g), n, n, n)).ravel()
        y = np.broadcast_to(Y[:, None, :, None], (len(g), n, n, n)).ravel()
        z = np.broadcast_to(Z[:, :, None, None], (len(g), n, n, n)).ravel()
        # density at the DE cells from the prim dump tp (|t - tp| ~ 0.15 M): sampled at x - v_c (t - tp), v_c the
        # rho >= 0.01 rho_c mass's mean coordinate velocity in the prim dump (the bulk motion over the offset)
        rp = [np.asarray(P['mb_data']['dens'][b], float) for b in range(len(gp))]
        hp = (gp[:, 1] - gp[:, 0])/n
        rpa = np.concatenate([a.ravel() for a in rp])
        Xp, Yp, Zp = ax(gp, hp, 0), ax(gp, hp, 2), ax(gp, hp, 4)
        xp = np.broadcast_to(Xp[:, None, None, :], (len(gp), n, n, n)).ravel()
        yp = np.broadcast_to(Yp[:, None, :, None], (len(gp), n, n, n)).ravel()
        zp = np.broadcast_to(Zp[:, :, None, None], (len(gp), n, n, n)).ravel()
        mp = rpa*np.repeat(hp**3, n**3); sel = rpa >= 0.01*rhoc
        vc = np.array([(mp[sel]*np.concatenate([np.asarray(P['mb_data'][k][b], float).ravel() for b in range(len(gp))])[sel]).sum()
                       / mp[sel].sum() for k in ('velx', 'vely', 'velz')])
        qx, qy, qz = x - vc[0]*(t - tp), y - vc[1]*(t - tp), z - vc[2]*(t - tp)
        rho = np.full(x.size, np.nan)
        for b in range(len(gp)):
            G = gp[b]
            ins = (qx >= G[0]) & (qx < G[1]) & (qy >= G[2]) & (qy < G[3]) & (qz >= G[4]) & (qz < G[5])
            if not ins.any():
                continue
            ii = np.clip(((qx[ins] - G[0])/hp[b]).astype(int), 0, n - 1)
            jj = np.clip(((qy[ins] - G[2])/hp[b]).astype(int), 0, n - 1)
            kk = np.clip(((qz[ins] - G[4])/hp[b]).astype(int), 0, n - 1)
            rho[ins] = rp[b][kk, jj, ii]
        ok = np.isfinite(rho)
        rho[~ok] = 0.0
        m = rho*np.repeat(h**3, n**3)
        d01, d1 = rho >= 0.01*rhoc, rho >= 0.1*rhoc
        xc = np.array([(m[d01]*c[d01]).sum()/m[d01].sum() for c in (x, y, z)])
        D = np.linalg.norm(xc - xbh(d, t)); rr = np.sqrt((x - xc[0])**2 + (y - xc[1])**2 + (z - xc[2])**2)
        mrem = m[d01].sum()
        for _ in range(3):
            rh = D*(mrem/(3.0*mbh))**(1.0/3.0); inh = rr < rh; mrem = m[inh].sum()
        cls = {'core': d1, 'outer': d01 & ~d1, 'rem': inh, 'stream': (~inh) & (rho >= 1e-6*rhoc) & (rho < 0.01*rhoc)}
        line = [f't={t:7.2f} (rho from t={tp:.2f}, v_c={np.linalg.norm(vc):.3f}, unmapped cells {(~ok).sum()})']
        for c, s in cls.items():
            M = m[s].sum()
            r = dict(t=t, t_prim=tp, cls=c, M=M)
            for b in range(6):
                r[f'br{b}'] = m[s & (br == b)].sum()/M if M > 0 else np.nan
            r['flagged'] = m[s & (sh != 0)].sum()/M if M > 0 else np.nan
            fs = s & (sh != 0) & (br >= 1)
            r.update(zip(('dk05', 'dk25', 'dk50', 'dk75', 'dk95'), wq(dk[fs], m[fs], (0.05, 0.25, 0.5, 0.75, 0.95))))
            rows.append(r)
            bs = '/'.join('%.3f' % r['br%d' % b] for b in range(6))
            line.append(f"{c}: M={M:.3e} br0-5={bs} fl={r['flagged']:.3f} "
                        f"dk50={r['dk50']:+.2e} [{r['dk05']:+.1e},{r['dk95']:+.1e}]")
        print(' | '.join(line), flush=True)
    keys = list(rows[0].keys())
    with open(out, 'w') as fo:
        fo.write(','.join(keys) + '\n')
        for r in rows:
            fo.write(','.join(str(r[k]) for k in keys) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
