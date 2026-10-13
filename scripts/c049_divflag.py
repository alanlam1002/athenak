#!/usr/bin/env python3
"""C-049 (R-060 item 2): offline coord vs covariant divergence in the dual-energy shock flag.

Reproduces the de_shock_flag kernel (src/eos/primitive_solver_hyd.hpp) in 3D from a restart's
conserved state (ideal-gas C2P, tde_dmde.c2p), over each MeshBlock including its ghost cells:
  v^a = Eulerian 3-velocity (= u~^a/sqrt(1 + u~^2) of the kernel), central differences;
  coord:     div = sum_a (v^a(+) - v^a(-))/(2 dx)
  covariant: div += sum_a v^a (ln sqrt(gamma)(+) - ln sqrt(gamma)(-))/(2 dx)  (dual_energy_div)
  sensor (vcurv): psi_v = max_a |v^a(+) - 2 v^a + v^a(-)|; sh = psi_v > vcurv cs, weak = psi_v >
  weak x vcurv cs (or psi_P > psi_jump / weak x psi_jump); flag = div <= 0 ? (sh ? 2 : weak ? 1 : 0) : 0;
  then the isolation filter (if on), de_dilate dilation passes, final = (flag == 2).
The deck's sensor settings are read from each restart's parameter block. dual_energy_shock_persist
> 1 needs the flag history and cannot be reproduced from one snapshot: the pre-persistence flag
is reported (both decks here have persist = 1).

Validation: the z = 0 layer (blocks with x3min = 0, k = 0, as the slice_x3 = 0 dediag_xy output)
against the stored dediag (channel 2 = final flag, 1 = sensor ratio, 0 = branch; dense = branch >
0) at the nearest dump. The stored flag comes from the last C2P call before the dump (stage-3
call, previous-stage primitives), not the restart's primitives, so agreement is not exact.

Per restart: dense cells (D > dual_energy_rho_switch), flagged fraction coord / covariant, cells
switching on->off and off->on, the share of switched cells that are marginal (psi_v within a
factor 2 of the threshold: 0.5 <= ratio <= 2), sign flips of div, and |covariant term| / |d_a v^a|
in the dense gas. Aggregated by phase t < 80, 80-120, 120-160, 160-200.

    c049_divflag.py [--out DIR] [--nproc N]
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
from tde_ensemble_plots import bc                 # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
RUNS = (('L5', 'L5c_rp10_comov_t200'), ('R6', 'R6_rp10_comov_L4_fix'))
BASE = 'cfc_tde_wd_imbh_parabolic_tracker'
PHASES = ((0, 80), (80, 120), (120, 160), (160, 200.01))


def deck(par):
    def get(k, dflt):
        m = re.search(rf'<mhd>(?:(?!<).)*?^\s*{k}\s*=\s*(\S+)', par, re.S | re.M)
        return m.group(1) if m else dflt
    return dict(rho_sw=float(get('dual_energy_rho_switch', 0)), gam=float(get('gamma', 5 / 3)),
                sensor=get('dual_energy_shock_sensor', 'vcurv'), vcsh=float(get('dual_energy_vcurv_shock', 0.05)),
                psijp=float(get('dual_energy_psi_jump', 1e30)), dil=int(get('dual_energy_shock_dilate', 1)),
                wk=float(get('dual_energy_shock_dilate_weak', 0.5)),
                iso=get('dual_energy_shock_isolated_off', 'false') in ('1', 'true'),
                pers=int(get('dual_energy_shock_persist', 1)))


def nb_any2(f):
    """for the inner (n-2)^3 cells: any face neighbour == 2"""
    c = (slice(1, -1),) * 3
    out = np.zeros(tuple(n - 2 for n in f.shape), bool)
    for ax in range(3):
        for sh in (0, 2):
            sl = list(c); sl[ax] = slice(sh, f.shape[ax] - 2 + sh)
            out |= f[tuple(sl)] == 2
    return out


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
    dense = Dd[I, I, I] > p['rho_sw']
    if not dense.any():
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
    lsg = np.log(sg)
    gam = p['gam']
    cs = np.sqrt(np.maximum(gam * P / (np.maximum(rho, 1e-300) * (1 + gam / (gam - 1) * P / np.maximum(rho, 1e-300))), 1e-300))
    R = slice(1, no - 1)                         # cells with both neighbours in every direction
    def sh(a, q, s):                             # q shifted by s along axis a, on R^3
        sl = [R, R, R]; sl[2 - a] = slice(1 + s, no - 1 + s)   # arrays are (k, j, i): x = axis 2
        return q[tuple(sl)]
    divc = sum((sh(a, v[a], 1) - sh(a, v[a], -1)) / (2 * dx) for a in range(3))
    cov = sum(sh(a, v[a], 0) * (sh(a, lsg, 1) - sh(a, lsg, -1)) / (2 * dx) for a in range(3))
    psiv = np.max([np.abs(sh(a, v[a], 1) - 2 * sh(a, v[a], 0) + sh(a, v[a], -1)) for a in range(3)], axis=0)
    Pp = [(sh(a, P, 1), sh(a, P, 0), sh(a, P, -1)) for a in range(3)]
    psip = np.max([np.abs(pp - 2 * p0 + pm) / np.maximum(pp + 2 * p0 + pm, 1e-300) for pp, p0, pm in Pp], axis=0)
    thr = p['vcsh'] * cs[R, R, R]
    shk_ = (psiv > thr) | (psip > p['psijp'])
    weak = (psiv > p['wk'] * thr) | (psip > p['wk'] * p['psijp'])
    ratio = np.maximum(psiv / thr, psip / p['psijp'])
    out = {}
    for name, div in (('f_coord', divc), ('f_cov', divc + cov)):
        f = np.where(div <= 0, np.where(shk_, 2, np.where(weak, 1, 0)), 0)
        for _ in range(p['dil']):
            inner = f[1:-1, 1:-1, 1:-1]
            f = np.where((inner == 2) | ((inner == 1) & nb_any2(f)), 2, inner)
        k0 = ng - 1 - p['dil']                   # interior offset in the shrunken array
        out[name] = (f[k0:k0 + nx, k0:k0 + nx, k0:k0 + nx] == 2)
    J = slice(ng - 1, ng - 1 + nx)
    out.update(dense=dense, ratio=ratio[J, J, J], divc=divc[J, J, J], cov=cov[J, J, J])
    return out


def one(job):
    lab, fn = job
    t = D.header(fn)
    p = deck(t['par'])
    if p['iso'] or p['dil'] > 2 or p['sensor'] != 'vcurv':
        raise SystemExit(f'{fn}: deck not covered by the replica (isolated / dilate > 2 / sensor): {p}')
    nscal = 1
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(t['nmb'], t['dsize'] // 8))
    x1min, x2min, x3min, x1max, x2max, x3max = t['size'][:6]
    w = np.array([x1max - x1min, x2max - x2min, x3max - x3min]); lo = np.array([x1min, x2min, x3min])
    lev = t['ll'][:, 3]
    acc = dict(nd=0, fc=0, fv=0, onoff=0, offon=0, chg_marg=0, sflip=0, rel=[], covpos=0)
    sl_mine = {}
    for b in range(t['nmb']):
        ext = w / 2.0**lev[b]; dx = ext[0] / 16
        o = block(mm[b], nscal, dx, p)
        if o is None:
            continue
        dn = o['dense']
        fc, fv = o['f_coord'] & dn, o['f_cov'] & dn
        acc['nd'] += dn.sum(); acc['fc'] += fc.sum(); acc['fv'] += fv.sum()
        onoff = fc & ~fv; offon = fv & ~fc
        acc['onoff'] += onoff.sum(); acc['offon'] += offon.sum()
        r = o['ratio']
        acc['chg_marg'] += ((onoff | offon) & (r >= 0.5) & (r <= 2)).sum()
        acc['sflip'] += (dn & ((o['divc'] <= 0) != (o['divc'] + o['cov'] <= 0))).sum()
        rel = np.abs(o['cov'][dn]) / np.maximum(np.abs(o['divc'][dn]), 1e-300)
        acc['rel'].append(rel)
        acc['covpos'] += (o['cov'][dn] > 0).sum()
        bmin = lo + t['ll'][b, :3] * ext
        if abs(bmin[2]) < 1e-9:
            key = (round(bmin[0], 6), round(bmin[1], 6), round(bmin[0] + ext[0], 6), round(bmin[1] + ext[1], 6))
            sl_mine[key] = (o['f_coord'][0], o['ratio'][0], dn[0])
    rel = np.concatenate(acc['rel']) if acc['rel'] else np.zeros(1)
    acc['rel'] = (np.median(rel), np.percentile(rel, 90), np.percentile(rel, 99))
    # validation against the nearest stored dediag_xy dump
    d = os.path.dirname(os.path.dirname(fn))
    fs = sorted(glob.glob(f'{d}/bin/{BASE}.dediag_xy.*.bin'))
    nums = np.array([int(f.split('.')[-2]) for f in fs])
    q = bc.read_binary(fs[int(np.argmin(np.abs(nums - t['time'])))])
    g = np.asarray(q['mb_geometry'])
    sk, sr, br = (np.asarray(q['mb_data'][k])[:, 0] for k in ('de_shock', 'de_sensor', 'de_branch'))
    m1 = s1 = a11 = a00 = n0 = n1 = nm = 0; lr = []
    for i in range(g.shape[0]):
        key = tuple(round(x, 6) for x in (g[i, 0], g[i, 2], g[i, 1], g[i, 3]))
        if key not in sl_mine:
            continue
        mf, mr, mdn = sl_mine[key]
        both = mdn & (np.rint(br[i]) > 0)
        nm += both.sum()
        st = sk[i] > 0.5
        n1 += (st & both).sum(); n0 += (~st & both).sum()
        a11 += (st & mf & both).sum(); a00 += (~st & ~mf & both).sum()
        m1 += (mf & both).sum()
        ok = both & (sr[i] > 0.1) & (mr > 0)
        lr.append(np.abs(np.log10(mr[ok] / sr[i][ok])))
    lr = np.concatenate(lr) if lr else np.zeros(1)
    val = dict(tdump=q['time'], nm=nm, st1=n1, my1=m1, p11=a11 / max(n1, 1), p00=a00 / max(n0, 1),
               agree=(a11 + a00) / max(nm, 1), lr=np.median(lr) if lr.size else np.nan)
    return lab, t['time'], p, acc, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    ap.add_argument('--nproc', type=int, default=13)
    a = ap.parse_args()
    jobs = [(lab, f) for lab, d in RUNS for f in sorted(glob.glob(f'{C}/{d}/rst/*.rst'))[1:]]
    with Pool(a.nproc) as pool:
        res = pool.map(one, jobs)
    rows = ['R-060 item 2: dual-energy shock flag, coordinate vs slice-covariant divergence (c049_divflag.py).',
            'Fractions are of dense interior cells (D > dual_energy_rho_switch). Pre-persistence flag (persist = 1 here).']
    for lab, t, p, acc, val in res:
        nd = max(acc['nd'], 1)
        rows.append(f'{lab} t {t:6.1f}: deck vcurv {p["vcsh"]} weak {p["wk"]} dilate {p["dil"]} iso {p["iso"]} persist {p["pers"]} '
                    f'rho_sw {p["rho_sw"]:.1e}; n_dense {acc["nd"]}')
        rows.append(f'   flagged coord {acc["fc"] / nd:.4f} cov {acc["fv"] / nd:.4f}; on->off {acc["onoff"] / nd:.2e} '
                    f'off->on {acc["offon"] / nd:.2e}; switched marginal {acc["chg_marg"] / max(acc["onoff"] + acc["offon"], 1):.3f}; '
                    f'div sign flips {acc["sflip"] / nd:.2e}; |cov|/|d v| median {acc["rel"][0]:.2e} p90 {acc["rel"][1]:.2e} '
                    f'p99 {acc["rel"][2]:.2e}; cov > 0 in {acc["covpos"] / nd:.3f}')
        rows.append(f'   validation vs dediag_xy t {val["tdump"]:.1f} (z = 0 layer, {val["nm"]} dense cells): stored flagged '
                    f'{val["st1"] / max(val["nm"], 1):.4f}, replica {val["my1"] / max(val["nm"], 1):.4f}; agree {val["agree"]:.4f} '
                    f'(P(rep=1|st=1) {val["p11"]:.3f}, P(rep=0|st=0) {val["p00"]:.3f}); median |log10 ratio rep/st| {val["lr"]:.3f}')
    rows.append('')
    rows.append('By phase (dense-cell weighted sums over the restarts in each window):')
    for lab, _ in RUNS:
        for t0, t1 in PHASES:
            sel = [r for r in res if r[0] == lab and t0 <= r[1] < t1]
            if not sel:
                continue
            S = {k: sum(r[3][k] for r in sel) for k in ('nd', 'fc', 'fv', 'onoff', 'offon', 'chg_marg', 'sflip')}
            nd = max(S['nd'], 1)
            rows.append(f'  {lab} t {t0:g}-{t1:.0f} ({len(sel)} restarts: ' + ', '.join(f'{r[1]:.1f}' for r in sel) + f'): flagged coord '
                        f'{S["fc"] / nd:.4f} -> cov {S["fv"] / nd:.4f}; on->off {S["onoff"] / nd:.2e}, off->on {S["offon"] / nd:.2e}; '
                        f'marginal share of switched {S["chg_marg"] / max(S["onoff"] + S["offon"], 1):.3f}')
    print('\n'.join(rows))
    with open(f'{a.out}/c049_divflag.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
