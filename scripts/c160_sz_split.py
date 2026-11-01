#!/usr/bin/env python3
"""c160 (R-194 item 6, user-approved 2026-10-31): the background split of KF_z's s_z on B1Z snapshots.
KF_z's source (dyn_grmhd_killing.cpp AddKillingFullTerms, killing_fz) is s_z = 1/2 alpha S^ij (2 alpha K_ij - (Lie_beta' gamma)_ij)
x sqrt(g), with Lie_beta' gamma from Dx<NGHOST> of the FULL adm.g_dd and adm.beta_u (beta' = beta_BH + delta beta + xidot) and
K = adm.vK_dd assembled analytically (Ahat^0 + LX, no differencing). Splitting the fields into the trumpet background about the hole
(gamma0 = psi0^4 delta, beta0 = b(r) x, from cfc_puncture.hpp TrumpetBackground) plus xidot plus the residual, the background-split
evaluation takes Lie_{beta0 + xidot} gamma0 analytically (its 2 alpha0 K0 - Lie_beta0 gamma0 = 0 exactly; the moving term
xidot . grad gamma0) and differences only the residual. The two differ by the stencil's truncation of the background term alone:
    s_z(as run) - s_z(split) = -1/2 alpha sqrt(g) S^ij [Lie^FD - Lie^an]_{beta0 + xidot} gamma0_ij,
and the tau rate (killing_full: dtau = (... + s_T)/alpha) by  d = -1/2 sqrt(g) S^ij [Lie^FD - Lie^an]_ij.
Lie^FD applies the code's Dx<4> (7-point, 6th order; nghost = 4) to the background's analytic values at the stencil cells, which is
exactly what the code's Dx of the full field contributes from the background part (Dx is linear). S^ij from the prim dump and the
adm dump's gamma (no B field). Reports per snapshot the dense-band (rho >= 0.01 rho_c) integral of d dV against B1Z's own dense
Delta rate (tau budget), and per-cell maps of d/(sqrt(g) S) on the L4 cells, split by distance to a refinement-level change.
    c160_sz_split.py OUT_PREFIX RUNDIR [RUNDIR ...]   (each with bin/*.adm_3d.*.bin and bin/*.prim_3d1.*.bin, parfile.par, user.hst)"""
import glob, re, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402
from scipy.optimize import brentq   # noqa: E402

GAM = 5.0/3.0
FAC = 0.75*np.sqrt(3.0)
W7 = np.array([-1/60, 3/20, -3/4, 0.0, 3/4, -3/20, 1/60])   # finite_diff.hpp Dx<4>, offsets -3..3


def chi(R):   # areal -> isotropic radius (M = 1), TrumpetArealToIso (c159_d2alpha.py)
    t = 2.0*R - 3.0
    s1 = np.sqrt(4*R*R + 4*R + 3); s2 = np.sqrt(8*R*R + 8*R + 6)
    return 0.25*(2*R + 1 + s1)*((4 + 3*np.sqrt(2))*t/(8*R + 6 + 3*s2))**(1/np.sqrt(2))


_RT = np.geomspace(1.5 + 1e-9, 400.0, 200001)
_CT = chi(_RT)


def areal(r):   # isotropic -> areal (M = 1): the table's guess, then Newton on chi(R) = r (dchi/dR = r/(R alpha0))
    R = np.interp(np.log(r), np.log(_CT), _RT)
    for _ in range(4):
        al = np.sqrt(np.maximum(1 - 2/R + 1.6875/R**4, 1e-300))
        c = chi(R)
        R = R - (c - r)*R*al/c
    return R


def bg(x, y, z, m):
    """psi0^4 and beta0 at Cartesian offsets from the hole (TrumpetBackground, m_bh = m)"""
    r = np.sqrt(x*x + y*y + z*z + 1e-30)
    R = m*areal(r/m)
    p4 = (R/r)**2
    b = FAC/(R*(R/m)**2)
    return p4, b*x, b*y, b*z


def lie_fd(x, y, z, dx, m, xd):
    """Lie_{beta0 + xidot} gamma0 with Dx<4> on the analytic background: (6 comps xx,xy,xz,yy,yz,zz) per cell"""
    off = np.arange(-3, 4)
    dg = np.zeros((3,) + x.shape); db = np.zeros((3, 3) + x.shape)   # d_c psi0^4, d_c beta^a
    for c in range(3):
        sh = [0.0, 0.0, 0.0]
        acc_g = 0.0; acc_b = [0.0, 0.0, 0.0]
        for w, o in zip(W7, off):
            if w == 0.0:
                continue
            sh = [o*dx if c == q else 0.0 for q in range(3)]
            p4, b1, b2, b3 = bg(x + sh[0], y + sh[1], z + sh[2], m)
            acc_g = acc_g + w*p4
            acc_b = [acc_b[0] + w*b1, acc_b[1] + w*b2, acc_b[2] + w*b3]
        dg[c] = acc_g/dx
        for a in range(3):
            db[c, a] = acc_b[a]/dx   # xidot is constant: its Dx is exactly 0
    p4, b1, b2, b3 = bg(x, y, z, m)
    be = [b1 + xd[0], b2 + xd[1], b3 + xd[2]]
    return lie_from(p4, be, dg, db)


def lie_an(x, y, z, m, xd):
    """the same, analytic: d_c psi0^4 = 4 psi0^3 psi0' x_c/r, d_c beta0^a = b delta_ca + b' x_c x_a / r"""
    r = np.sqrt(x*x + y*y + z*z + 1e-30)
    R = m*areal(r/m)
    psi = np.sqrt(R/r)
    rrs = R/m; rrs3 = rrs**3
    al = np.sqrt(np.maximum(1 - 2/rrs + 1.6875/(rrs3*rrs), 0.0))
    dpsi = -psi*(1.0 - 0.84375/rrs3)/(rrs*r*(1.0 + al))     # cfc_puncture.hpp dpsi0
    dRdr = R*al/r                                              # dR/dr_iso
    b = FAC/(R*rrs*rrs); dbdr = -3.0*b/R*dRdr
    X = [x, y, z]
    dg = np.array([4*psi**3*dpsi*X[c]/r for c in range(3)])
    db = np.array([[(b if a == c else 0.0) + dbdr*X[c]*X[a]/r for a in range(3)] for c in range(3)])
    be = [b*x + xd[0], b*y + xd[1], b*z + xd[2]]
    return lie_from(psi**4, be, dg, db)


def lie_from(p4, be, dg, db):
    """gamma0 = p4 delta: Lie_ab = beta^c d_c p4 delta_ab + p4 (d_a beta^b + d_b beta^a)"""
    bdg = be[0]*dg[0] + be[1]*dg[1] + be[2]*dg[2]
    out = {}
    for a, b_ in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        out[(a, b_)] = (bdg if a == b_ else 0.0) + p4*(db[a, b_] + db[b_, a])
    return out


def par(d, k, dflt=0.0):
    m = re.search(rf'^\s*{k}\s*=\s*([-+.\deE]+)', open(f'{d}/parfile.par').read(), re.M)
    return float(m.group(1)) if m else dflt


def hst_at(d, t, names):
    f = glob.glob(f'{d}/*.user.hst')[0]
    hdr = open(f).readlines()[1]
    h = np.loadtxt(f, ndmin=2); i = np.argmin(np.abs(h[:, 0] - t))
    return [h[i, int(re.search(rf'\[(\d+)\]={n}\b', hdr).group(1)) - 1] if re.search(rf'\]={n}\b', hdr) else 0.0
            for n in names]


def main():
    pre, runs = sys.argv[1], sys.argv[2:]
    rows = ['run,t,rho_dense_cells,int_d_dense,int_absd_dense,int_sqrtgS_dense,ratio_d_over_S,int_d_L4_near_levelchange,'
            'n_near,int_d_L4_interior,xbh,ybh,zbh,xidot_x,xidot_y,xidot_z']
    for d in runs:
        rhoc, k0, m = par(d, 'rhoc'), par(d, 'kappa'), par(d, 'puncture_mass', 1.0)
        p0 = np.array([par(d, f'puncture_x{a}') for a in (1, 2, 3)])
        for fa in sorted(glob.glob(f'{d}/bin/*.adm_3d.*.bin')):
            A = bc.read_binary(fa)
            t = float(A['time'])
            fp = [f for f in glob.glob(f'{d}/bin/*.prim_3d1.*.bin') if abs(bc.read_binary(f)['time'] - t) < 1e-9][0]
            Pm = bc.read_binary(fp)
            disp = hst_at(d, t, ['frame_disp_x1', 'frame_disp_x2', 'frame_disp_x3'])
            xd = hst_at(d, t, ['frame_vel_x1', 'frame_vel_x2', 'frame_vel_x3'])
            xb = p0 - np.array(disp)
            g = np.asarray(A['mb_geometry'], dtype=float); lev = np.asarray(A['mb_logical'])[:, 3]
            n = A['nx1_out_mb']; qa = A['mb_data']; qp = Pm['mb_data']
            lmax = lev.max()
            tot = dict(d=0.0, ad=0.0, S=0.0, dn=0.0, di=0.0, nn=0); nd = 0
            maps = []
            for bk in range(len(g)):
                rho = np.asarray(qp['dens'][bk], dtype=float)
                dense = rho >= 0.01*rhoc
                if not dense.any():
                    continue
                dx = (g[bk, 1] - g[bk, 0])/n
                c = (np.arange(n) + 0.5)*dx
                zz, yy, xx = np.meshgrid(g[bk, 4] + c, g[bk, 2] + c, g[bk, 0] + c, indexing='ij')
                x, y, z = xx - xb[0], yy - xb[1], zz - xb[2]
                Lf = lie_fd(x, y, z, dx, m, xd); La = lie_an(x, y, z, m, xd)
                gd = {k: np.asarray(qa[f'adm_g{k}'][bk], dtype=float) for k in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz')}
                G = np.array([[gd['xx'], gd['xy'], gd['xz']], [gd['xy'], gd['yy'], gd['yz']], [gd['xz'], gd['yz'], gd['zz']]])
                Gm = np.moveaxis(G, (0, 1), (-2, -1)); Gu = np.linalg.inv(Gm); detg = np.linalg.det(Gm); vol = np.sqrt(detg)
                pr = np.asarray(qp['press'][bk], dtype=float)
                u = np.stack([np.asarray(qp[k][bk], dtype=float) for k in ('velx', 'vely', 'velz')], -1)
                usq = np.einsum('...a,...ab,...b->...', u, Gm, u); W = np.sqrt(1 + usq)
                H = rho + GAM/(GAM - 1)*pr   # ideal gas: rho h
                Suu = H[..., None, None]*u[..., :, None]*u[..., None, :] + pr[..., None, None]*Gu
                dl = 0.0
                for (a, b_), v in Lf.items():
                    dlab = v - La[(a, b_)]
                    dl = dl + (1.0 if a == b_ else 2.0)*Suu[..., a, b_]*dlab
                dd = -0.5*vol*dl            # tau rate change per unit coordinate volume
                S = np.einsum('...ab,...ab->...', Gm, Suu)
                dv = dx**3
                # near a level change: the block touches (or is within 3 cells of) a block of another level
                near = False
                if lev[bk] == lmax:
                    lo, hi = g[bk, 0::2] - 3.5*dx, g[bk, 1::2] + 3.5*dx
                    ov = np.all((g[:, 0::2] < hi) & (g[:, 1::2] > lo), axis=1)
                    near = bool((lev[ov] != lev[bk]).any())
                tot['d'] += (dd*dense).sum()*dv; tot['ad'] += (np.abs(dd)*dense).sum()*dv
                tot['S'] += (vol*S*dense).sum()*dv; nd += int(dense.sum())
                if near:
                    tot['dn'] += (dd*dense).sum()*dv; tot['nn'] += int(dense.sum())
                else:
                    tot['di'] += (dd*dense).sum()*dv
                maps.append(np.stack([xx[dense], yy[dense], zz[dense], rho[dense]/rhoc, dd[dense]/np.maximum(vol*S, 1e-300)[dense],
                                      np.full(dense.sum(), lev[bk]), np.full(dense.sum(), float(near))], 1))
            mp = np.concatenate(maps)
            np.save(f'{pre}_map_{d.rstrip("/").split("/")[-1]}_t{t:.1f}.npy', mp.astype(np.float32))
            rows.append(f'{d},{t},{nd},{tot["d"]},{tot["ad"]},{tot["S"]},{tot["d"]/tot["S"]},{tot["dn"]},{tot["nn"]},{tot["di"]},'
                        f'{xb[0]},{xb[1]},{xb[2]},{xd[0]},{xd[1]},{xd[2]}')
            print(rows[-1], flush=True)
            q = np.abs(mp[:, 4])
            print(f'   per-cell |d|/(sqrt(g) S): median {np.median(q):.3e}, 99% {np.quantile(q, 0.99):.3e}, max {q.max():.3e}; '
                  f'near a level change: {int(mp[:, 6].sum())} of {len(mp)} dense cells', flush=True)
    open(f'{pre}.csv', 'w').write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
