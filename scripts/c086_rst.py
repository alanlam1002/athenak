#!/usr/bin/env python3
"""C-086 items 8 and 9 from 3D restarts.

Item 8: the C-049 offline replica of the dual-energy shock flag (scripts/c049_divflag.py: block(),
validated in C-049 against the stored dediag_xy flag; that replica follows ComputeShockFlags,
src/eos/primitive_solver_hyd.hpp:1589-1757), evaluated on cells BELOW dual_energy_rho_switch.
Adaptations of c049_divflag.block (everything else is the same code path):
  - blocks are kept if any interior cell has D > 1e-9 rho_c (c049 kept only blocks with dense cells);
  - per interior cell it returns D (= cons(IDN)/sqrt(gamma), the switch variable, C++ 1977/2252-2253),
    dM = cons(IDN) dV (tde_dmde.cells), c_s, psi_v, coord div v, the final flag, and the block level;
  - a counterfactual flag with c_s from the initial adiabat, P = K0 rho^Gamma (the cold limit: the
    entropy branch keeps K, and a shock can only raise it), through the same pre-flag / dilation path;
  - the validation (z = 0 layer vs stored dediag_xy de_shock at the nearest dump) is restricted to the
    stored below-switch cells (de_branch == 0) with D > 1e-9 rho_c.
Coordinate divergence only (the R6/R7 decks have no dual_energy_div line; default coord, C++ 1346-1354).

Item 9: bound = BH-frame E = -u_t - 1 < 0 over r_BH >= 3 (tde_dmde.cells, rcut = 3), remnant =
remnant.bound_mass (self-bound clump), bound debris = bound and not remnant (c051_s2.fallback).
D = rho W with W from tde_dmde's u_i and v^i (u_i v^i = W - 1/W).

    c086_rst.py [--out DIR] [--nproc N] [--runs R6,R7]
"""
import argparse
import glob
import re
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import remnant as RM                              # noqa: E402
import c049_divflag as F                          # noqa: E402
from tde_ensemble_plots import bc                 # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
B = 'cfc_tde_wd_imbh_parabolic_tracker'
RHOC, K0, MSTAR = 3.24e-4, 1.2346626297e-2, 7.47e-5
LO = 1e-9 * RHOC
RUNS = {'R6': ['R6_rp10_comov_L4_fix', 'R6x_rp10_comov_L4_t300'],
        'R7': ['R7_rp18_comov_L4_fix', 'R7x_rp18_comov_L4_t500']}


def flags(psiv, psip, divc, cs, p):
    """C++ 1663-1671 (sensor 2), 1757 (dilation, de_dilate passes), final == 2; as c049_divflag.block"""
    thr = p['vcsh'] * cs
    shk_ = (psiv > thr) | (psip > p['psijp'])
    weak = (psiv > p['wk'] * thr) | (psip > p['wk'] * p['psijp'])
    f = np.where(divc <= 0, np.where(shk_, 2, np.where(weak, 1, 0)), 0)
    for _ in range(p['dil']):
        inner = f[1:-1, 1:-1, 1:-1]
        f = np.where((inner == 2) | ((inner == 1) & F.nb_any2(f)), 2, inner)
    k0 = 4 - 1 - p['dil']
    return f[k0:k0 + 16, k0:k0 + 16, k0:k0 + 16] == 2


def block(rec, nscal, dx, p):
    nx, ng = 16, 4
    no = nx + 2 * ng; nmhd = 5 + nscal; ncc = no**3; bsz = (no + 1) * no * no * 3
    cons = rec[:nmhd * ncc].reshape(nmhd, no, no, no)
    adm = rec[nmhd * ncc + bsz:].reshape(17, no, no, no)
    gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
    det = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz)
    sg = np.sqrt(np.maximum(det, 1e-300))
    I = slice(ng, ng + nx)
    Dd = cons[0] / sg
    if not (Dd[I, I, I] > LO).any():
        return None
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
    gam = p['gam']
    rr = np.maximum(rho, 1e-300)
    cs = np.sqrt(np.maximum(gam * P / (rr * (1 + gam / (gam - 1) * P / rr)), 1e-300))
    P0 = K0 * rr**gam                              # counterfactual: the initial adiabat
    cs0 = np.sqrt(np.maximum(gam * P0 / (rr * (1 + gam / (gam - 1) * P0 / rr)), 1e-300))
    R = slice(1, no - 1)
    def sh(a, q, s):
        sl = [R, R, R]; sl[2 - a] = slice(1 + s, no - 1 + s)
        return q[tuple(sl)]
    divc = sum((sh(a, v[a], 1) - sh(a, v[a], -1)) / (2 * dx) for a in range(3))
    psiv = np.max([np.abs(sh(a, v[a], 1) - 2 * sh(a, v[a], 0) + sh(a, v[a], -1)) for a in range(3)], axis=0)
    Pp = [(sh(a, P, 1), sh(a, P, 0), sh(a, P, -1)) for a in range(3)]
    psip = np.max([np.abs(pp - 2 * p0 + pm) / np.maximum(pp + 2 * p0 + pm, 1e-300) for pp, p0, pm in Pp], axis=0)
    # psi_P for the counterfactual (P0 = K0 rho^Gamma); irrelevant while psi_jump = 1e30, kept for completeness
    Pq = [(sh(a, P0, 1), sh(a, P0, 0), sh(a, P0, -1)) for a in range(3)]
    psip0 = np.max([np.abs(pp - 2 * p0 + pm) / np.maximum(pp + 2 * p0 + pm, 1e-300) for pp, p0, pm in Pq], axis=0)
    csR, cs0R = cs[R, R, R], cs0[R, R, R]
    f_act = flags(psiv, psip, divc, csR, p)
    f_cold = flags(psiv, psip0, divc, cs0R, p)
    J = slice(ng - 1, ng - 1 + nx)
    return dict(D=Dd[I, I, I], dM=cons[0][I, I, I] * dx**3, cs=cs[I, I, I], cs0=cs0[I, I, I],
                psiv=psiv[J, J, J], divdx=divc[J, J, J] * dx, f=f_act, f0=f_cold,
                ratio=np.maximum(psiv / (p['vcsh'] * csR), psip / p['psijp'])[J, J, J])


EDG = RHOC * np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
BLAB = ['1e-9..1e-8', '1e-8..1e-7', '1e-7..1e-6', '1e-6..1e-5', '1e-5..sw']


def one(job):
    lab, fn = job
    t = D.header(fn)
    p = F.deck(t['par'])
    if p['iso'] or p['dil'] > 2 or p['sensor'] != 'vcurv' or p['pers'] != 1 or \
            re.search(r'^\s*dual_energy_div\s*=\s*(?!coord)', t['par'], re.M):
        raise SystemExit(f'{fn}: deck not covered: {p}')
    edges = np.append(EDG, p['rho_sw'])
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(t['nmb'], t['dsize'] // 8))
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min]); lo = np.array([x1min, x2min, x3min])
    lev = t['ll'][:, 3]
    xbh, _ = D.frame_state(t['par'], t['time'])
    nb = len(BLAB)
    acc = {k: np.zeros(nb) for k in ('n', 'nf', 'm', 'mf', 'nf0', 'mf0', 'ndiv', 'mdiv')}
    dn = np.zeros(4)
    samp = []              # flagged below-switch cells: bin, log psi_v, log c_s, log c_s0, -div dx / c_s, level
    samp_all = []          # all below-switch cells (subsampled): bin, log psi_v, log c_s, -div dx/c_s, level, flag
    lvl = {}
    sl_mine = {}
    rng = np.random.default_rng(0)
    ic = np.arange(16) + 0.5
    for b in range(t['nmb']):
        ext = w / 2.0**lev[b]; dx = ext[0] / 16
        o = block(mm[b], 1, dx, p)
        if o is None:
            continue
        bmin = lo + t['ll'][b, :3] * ext
        Z, Y, X = np.meshgrid(bmin[2] + ic * dx, bmin[1] + ic * dx, bmin[0] + ic * dx, indexing='ij')
        rbh = np.sqrt((X - xbh[0])**2 + (Y - xbh[1])**2 + (Z - xbh[2])**2)
        keep = rbh >= 3.0
        Dd = o['D']
        bi = np.digitize(Dd, edges) - 1                  # 0..nb-1 inside [1e-9 rho_c, rho_sw]
        bi = np.where((Dd > edges[0]) & (Dd <= edges[-1]) & keep, bi, -1)
        L = lev[b] - t['root']
        for k in range(nb):
            s = bi == k
            if not s.any():
                continue
            m = o['dM'][s]
            acc['n'][k] += s.sum(); acc['m'][k] += m.sum()
            acc['nf'][k] += o['f'][s].sum(); acc['mf'][k] += m[o['f'][s]].sum()
            acc['nf0'][k] += o['f0'][s].sum(); acc['mf0'][k] += m[o['f0'][s]].sum()
            dv = o['divdx'][s] <= 0
            acc['ndiv'][k] += dv.sum(); acc['mdiv'][k] += m[dv].sum()
            a = lvl.setdefault(L, np.zeros((nb, 4)))
            a[k] += (s.sum(), o['f'][s].sum(), m.sum(), m[o['f'][s]].sum())
        sf = (bi >= 0) & o['f']
        if sf.any():
            samp.append(np.stack([bi[sf], np.log10(np.maximum(o['psiv'][sf], 1e-30)), np.log10(o['cs'][sf]),
                                  np.log10(o['cs0'][sf]), -o['divdx'][sf] / o['cs'][sf], np.full(sf.sum(), L)]).astype(np.float32))
        sa = (bi >= 0) & (rng.random(bi.shape) < 0.1)
        if sa.any():
            samp_all.append(np.stack([bi[sa], np.log10(np.maximum(o['psiv'][sa], 1e-30)), np.log10(o['cs'][sa]),
                                      -o['divdx'][sa] / o['cs'][sa], np.full(sa.sum(), L), o['f'][sa],
                                      np.log10(o['cs0'][sa]), o['f0'][sa]]).astype(np.float32))
        de = (Dd > p['rho_sw']) & keep
        dn += (de.sum(), (de & o['f']).sum(), o['dM'][de].sum(), o['dM'][de & o['f']].sum())
        if abs(bmin[2]) < 1e-9:
            key = (round(bmin[0], 6), round(bmin[1], 6), round(bmin[0] + ext[0], 6), round(bmin[1] + ext[1], 6))
            sl_mine[key] = (o['f'][0], o['ratio'][0], Dd[0])
    # validation vs the nearest stored dediag_xy (z = 0 layer), below-switch band cells
    d = os.path.dirname(os.path.dirname(fn))
    fs = sorted(glob.glob(f'{d}/bin/{B}.dediag_xy.*.bin'))
    nums = np.array([int(f.split('.')[-2]) for f in fs])
    q = bc.read_binary(fs[int(np.argmin(np.abs(nums - t['time'])))])
    g = np.asarray(q['mb_geometry'])
    sk, sr, br = (np.asarray(q['mb_data'][k])[:, 0] for k in ('de_shock', 'de_sensor', 'de_branch'))
    nm = n1 = m1 = a11 = a00 = n0 = 0; lr = []
    for i in range(g.shape[0]):
        key = tuple(round(x, 6) for x in (g[i, 0], g[i, 2], g[i, 1], g[i, 3]))
        if key not in sl_mine:
            continue
        mf, mr, md = sl_mine[key]
        both = (np.rint(br[i]) == 0) & (md > LO) & (md <= p['rho_sw'])
        nm += both.sum(); st = sk[i] > 0.5
        n1 += (st & both).sum(); n0 += (~st & both).sum(); m1 += (mf & both).sum()
        a11 += (st & mf & both).sum(); a00 += (~st & ~mf & both).sum()
        okr = both & (sr[i] > 0.1) & (mr > 0)
        lr.append(np.abs(np.log10(mr[okr] / sr[i][okr])))
    lr = np.concatenate(lr) if lr else np.zeros(0)
    val = dict(tdump=q['time'], nm=nm, st1=n1 / max(nm, 1), my1=m1 / max(nm, 1), agree=(a11 + a00) / max(nm, 1),
               p11=a11 / max(n1, 1), p00=a00 / max(n0, 1), lr=np.median(lr) if lr.size else np.nan)
    # item 9
    c = D.cells(fn, rcut=3.0)
    Mb, xc, vc, rem = RM.bound_mass(c)
    s = c['ux'] * c['vx'] + c['uy'] * c['vy'] + c['uz'] * c['vz']   # = W - 1/W
    Wc = 0.5 * (s + np.sqrt(s * s + 4))
    Dc = c['rho'] * Wc
    bnd = c['E'] < 0
    deb = bnd & ~rem
    dM = c['dM']
    i9 = {}
    for name, sel in (('all', np.ones_like(bnd)), ('bound', bnd), ('debris', deb), ('remnant', rem)):
        M = dM[sel].sum()
        i9[name] = (M / MSTAR, dM[sel & (Dc < p['rho_sw'])].sum() / max(M, 1e-300), dM[sel & (Dc < LO)].sum() / max(M, 1e-300),
                    dM[sel & (Dc < 1e-6 * RHOC)].sum() / max(M, 1e-300))
    # how far D and rho differ for the switch
    i9['n_rho_vs_D'] = int(((c['rho'] < p['rho_sw']) != (Dc < p['rho_sw'])).sum())
    S = np.concatenate(samp, axis=1) if samp else np.zeros((6, 0), np.float32)
    SA = np.concatenate(samp_all, axis=1) if samp_all else np.zeros((8, 0), np.float32)
    return dict(lab=lab, t=t['time'], fn=fn, acc=acc, dense=dn, lvl=lvl, S=S, SA=SA, val=val, i9=i9, p=p)


def pct(a, q=(10, 50, 90)):
    return '/'.join(f'{x:+.2f}' for x in np.percentile(a, q)) if a.size else 'n/a'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--runs', default='R6,R7')
    ap.add_argument('--samples', default=None, help='write the per-cell samples npz here (~100 MB; keep off /plots)')
    a = ap.parse_args()
    jobs = []
    for lab in a.runs.split(','):
        for d in RUNS[lab]:
            fs = sorted(glob.glob(f'{C}/{d}/rst/*.rst'))
            jobs += [(lab, f) for f in fs if D.header(f)['time'] > 1.0]
    with Pool(a.nproc) as pool:
        res = pool.map(one, jobs, chunksize=1)
    res.sort(key=lambda r: (r['lab'], r['t']))
    if a.samples:
      np.savez_compressed(a.samples,
                        **{f'{r["lab"]}_{r["t"]:.1f}_flagged': r['S'] for r in res},
                        **{f'{r["lab"]}_{r["t"]:.1f}_all10pct': r['SA'] for r in res})
    L = ['C-086 items 8/9 from 3D restarts (scripts/c086_rst.py; sensor replica = c049_divflag.block adapted).',
         'Item 8: cells with 1e-9 rho_c < D <= rho_sw (3.2e-8), r_BH >= 3. f_n / f_m = flagged count / mass (dM = cons(IDN) dV)',
         'fraction with the run\'s own c_s; f0_n / f0_m = flagged with the cold counterfactual c_s(K0 rho^Gamma); div<=0 = share',
         'of cells with coordinate div v <= 0 (the flag\'s necessary condition, C++ 1671).', '']
    for r in res:
        p, v = r['p'], r['val']
        L.append(f'{r["lab"]} t {r["t"]:6.1f} ({os.path.basename(os.path.dirname(os.path.dirname(r["fn"])))}): '
                 f'validation vs dediag t {v["tdump"]:.1f}: {v["nm"]} below-sw z=0 cells, stored flagged {v["st1"]:.4f}, '
                 f'replica {v["my1"]:.4f}, agree {v["agree"]:.4f} (P(rep=1|st=1) {v["p11"]:.3f}, P(rep=0|st=0) {v["p00"]:.3f}); '
                 f'median |log10 ratio rep/st| {v["lr"]:.3f}')
        A = r['acc']
        for k in range(len(BLAB)):
            n, m = max(A['n'][k], 1), max(A['m'][k], 1e-300)
            L.append(f'    {BLAB[k]:11s} N {int(A["n"][k]):9d} M/M* {A["m"][k] / MSTAR:.2e}  f_n {A["nf"][k] / n:.4f} f_m {A["mf"][k] / m:.4f}'
                     f'   cold: f0_n {A["nf0"][k] / n:.4f} f0_m {A["mf0"][k] / m:.4f}   div<=0 n {A["ndiv"][k] / n:.3f} m {A["mdiv"][k] / m:.3f}')
        dn = r['dense']
        L.append(f'    dense (D > sw) N {int(dn[0])}: f_n {dn[1] / max(dn[0], 1):.4f} f_m {dn[3] / max(dn[2], 1e-300):.4f}')
    L += ['', 'Drivers, flagged below-switch cells (run c_s), pooled over restarts per run and density bin:',
          '  log10 psi_v p10/50/90 | log10 c_s p10/50/90 | log10 c_s(K0) p10/50/90 | C = -div v dx / c_s p10/50/90 |',
          '  share with C < 0.01 (div v ~ 0: curvature-only) | ditto for all below-sw cells (10% sample): log10 psi_v, log10 c_s']
    for lab in a.runs.split(','):
        rs = [r for r in res if r['lab'] == lab]
        S = np.concatenate([r['S'] for r in rs], axis=1)
        SA = np.concatenate([r['SA'] for r in rs], axis=1)
        for k in range(len(BLAB)):
            s, sa = S[0] == k, SA[0] == k
            Cn = S[4][s]
            L.append(f'  {lab} {BLAB[k]:11s} nflag {s.sum():8d}: psi_v {pct(S[1][s])} | c_s {pct(S[2][s])} | c_s0 {pct(S[3][s])} | '
                     f'C {pct(Cn)} | C<0.01 {np.mean(Cn < 0.01) if Cn.size else np.nan:.3f}')
            L.append(f'  {"":3s} {"":11s} all (10%) {sa.sum():8d}: psi_v {pct(SA[1][sa])} | c_s {pct(SA[2][sa])} | c_s0 {pct(SA[6][sa])} '
                     f'| C {pct(SA[3][sa])}')
        # by level
        L.append(f'  {lab} flagged fraction by MeshBlock level above root (count f_n / mass f_m, all bins, all restarts):')
        lv = {}
        for r in rs:
            for Lk, arr in r['lvl'].items():
                lv[Lk] = lv.get(Lk, 0) + arr
        for Lk in sorted(lv):
            arr = lv[Lk].sum(0)
            L.append(f'      level {Lk}: N {int(arr[0]):9d} f_n {arr[1] / max(arr[0], 1):.4f} f_m {arr[3] / max(arr[2], 1e-300):.4f}')
    L += ['', 'Item 9 (3D): bound = E < 0 over r_BH >= 3 (tde_dmde.cells), remnant = remnant.bound_mass, debris = bound & ~remnant.',
          'Mass fractions of each set with D < rho_sw (3.2e-8), D < 1e-9 rho_c, and (for scale) D < 1e-6 rho_c.']
    for r in res:
        i = r['i9']
        L.append(f'  {r["lab"]} t {r["t"]:6.1f}: ' + '; '.join(
            f'{nm} {i[nm][0]:.4f} M*: <sw {i[nm][1]:.3e} <1e-9rc {i[nm][2]:.3e} <1e-6rc {i[nm][3]:.3e}'
            for nm in ('bound', 'debris', 'remnant')) + f'  (cells where rho vs D change the <sw test: {i["n_rho_vs_D"]})')
    print('\n'.join(L))
    with open(f'{a.out}/c086_rst.txt', 'w') as fo:
        fo.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
