#!/usr/bin/env python3
"""Phase 2 M1, open item (R-040 2a): BH-centred constraint diagnostics for the vacuum
moving-puncture runs, from restart files (3D ADM state; layout in scripts/tde_dmde.py).

CFC metric gamma_ij = psi^4 delta_ij (psi = psi4^(1/4)); vacuum, so per cell
  H   = R + K^2 - K_ij K^ij,          R = -8 psi^-5 lap(psi)   (flat Laplacian)
  M_i = D_j K^j_i - d_i K,            D_j K^j_i = d_j K^j_i + 6 phi_k K^k_i - 2 phi_i K,
with phi = ln psi, K^j_i = psi^-4 K_ij, K = psi^-4 delta^ij K_ij (Christoffels of a
conformally flat metric). Second-order centred differences on each MeshBlock, using
interior cells only (the outermost interior layer is skipped, so no ghost or prolongated
data enters). Scales for a relative measure: |R| term-by-term, 8 psi^-5 sum_i |d_ii psi|,
and |d_j K^j_i| summed term-by-term.

K CONVENTION (measured 2026-10-11, R-040 item 2a): the stored adm vK_dd of the puncture runs
equals psi^-6 x the true K_ij, by three independent checks (the analytic trumpet K^r_r =
-2C/R^3; K_ij = (1/2 alpha) L_beta gamma_ij from the code's own alpha/beta/psi; and the momentum
constraint, which vanishes only for K^j_i = psi^2 vK_dd). The puncture background
(cfc_puncture.hpp TrumpetBackground) fills Ahat0_ij with the mixed K^i_j, not
Ahat_ij = psi^6 K^i_j, and AssembleLapseShiftK writes vK_dd = psi^-2 Ahat_ij. So the
constraints are reported both with vK_dd as K_ij ("as stored", what the hydro source uses)
and with K_ij = psi^6 vK_dd ("fixed", which measures the moving-puncture solve). Relative norms:
H / |R| and |M| / sum |d_j K^j_i|.

Norms in BH-centred shells r_BH in [3,6], [6,12], [12,24] (x_BH = puncture_x1 - frame_vel1 t
when puncture_moving), volume-weighted L2 and max. Also the ADM-mass surface integral
M(r) = -(1/2 pi) oint d_r psi dS on spheres about x_BH and about the origin (trilinear
interpolation inside the finest block that contains each point), against the analytic
static trumpet's value at the same r (finite-radius, ~ 1 + O(1/r)).

    m1_constraints.py LABEL=RUNDIR[,RUNDIR...] [...] [--out f.npz]
    m1_constraints.py --plot OUT.png A.npz [B.npz ...]     (figure from saved runs)
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as T                                  # noqa: E402
import tde_elliptical_orbit_ic as TR                  # noqa: E402

SHELLS = ((3.0, 6.0), (6.0, 12.0), (12.0, 24.0))
RS = (6.0, 10.0, 16.0)            # inside the +-64 box for |x_BH| <= 45
NX, NG = 16, 4
NO = NX + 2 * NG


def par(text, blk, key, default=0.0):
    m = re.search(rf'<{blk}>(.*?)(?=\n<|\Z)', text, re.S)
    if not m:
        return default
    mm = re.search(rf'^{key}\s*=\s*([-+.\deE]+)', m.group(1), re.M)
    return float(mm.group(1)) if mm else default


def blocks(fn):
    """yield (x0, dx, adm[17, NO, NO, NO]) per MeshBlock; x0 = lower corner of the interior"""
    h = T.header(fn)
    x1min, x2min, x3min, x1max, x2max, x3max = h['size'][:6]
    lo = np.array([x1min, x2min, x3min]); w = np.array([x1max - x1min, x2max - x2min, x3max - x3min])
    ncc = NO**3; bsz = (NO + 1) * NO * NO * 3
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=h['data_off'], shape=(h['nmb'], h['dsize'] // 8))
    ll = h['ll']
    for b in range(h['nmb']):
        ext = w / 2.0**ll[b, 3]
        x0 = lo + ll[b, :3] * ext
        adm = np.asarray(mm[b][6 * ncc + bsz:]).reshape(17, NO, NO, NO)
        yield x0, ext[0] / NX, adm
    return


def d1(f, ax, dx):
    return (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / (2 * dx)


def d2(f, ax, dx):
    return (np.roll(f, -1, ax) - 2 * f + np.roll(f, 1, ax)) / dx**2


def constraints(adm, dx, kfix=False):
    """H, |M|, and their scales on the interior cells [NG+1, NG+NX-1) (arrays in z,y,x order).
    kfix: take K_ij = psi^6 vK_dd instead of vK_dd (see the module docstring)"""
    psi4 = adm[12]
    psi = psi4**0.25
    Kl = adm[6:12]                                 # Kxx Kxy Kxz Kyy Kyz Kzz (lower)
    K = np.empty((3, 3) + psi.shape)
    idx = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}
    for (i, j), n in idx.items():
        K[i, j] = K[j, i] = Kl[n]
    # array axes are (z, y, x): spatial direction i -> array axis 2 - i
    ax = lambda i: 2 - i
    lap = sum(d2(psi, ax(i), dx) for i in range(3))
    R = -8 * psi**-5 * lap
    Km = K * psi4**(0.5 if kfix else -1.0)         # mixed K^j_i (symmetric in flat indices)
    trK = Km[0, 0] + Km[1, 1] + Km[2, 2]
    KK = sum(Km[i, j] * Km[j, i] for i in range(3) for j in range(3))
    H = R + trK**2 - KK
    sc_H = np.abs(R)
    phi = np.log(psi)
    dphi = [d1(phi, ax(k), dx) for k in range(3)]
    dK = [d1(trK, ax(i), dx) for i in range(3)]
    M = []; sc_M = 0
    for i in range(3):
        div = sum(d1(Km[j, i], ax(j), dx) for j in range(3))
        sc_M = sc_M + sum(np.abs(d1(Km[j, i], ax(j), dx)) for j in range(3))
        M.append(div + 6 * sum(dphi[k] * Km[k, i] for k in range(3)) - 2 * dphi[i] * trK - dK[i])
    Mabs = np.sqrt(sum(m**2 for m in M))
    s = slice(NG + 1, NG + NX - 1)
    return [a[s, s, s] for a in (H, sc_H, Mabs, sc_M)]


def trumpet_psi(r):
    r = np.atleast_1d(r)
    rho = np.array([TR.trumpet_iso_to_areal(c) for c in r])
    return np.sqrt(rho / r)


def analytic_mass(r, h=1e-3):
    dpsi = (trumpet_psi(r + h) - trumpet_psi(r - h)) / (2 * h)
    return float(-2.0 * r * r * dpsi[0])                 # -(1/2pi) 4 pi r^2 d_r psi


def sphere(nth=24, nph=48):
    th = (np.arange(nth) + 0.5) * np.pi / nth
    ph = np.arange(nph) * 2 * np.pi / nph
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    n = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1).reshape(-1, 3)
    dw = (np.sin(TH) * (np.pi / nth) * (2 * np.pi / nph)).ravel()
    return n, dw


def surface_mass(blks, centre, r):
    """-(1/2 pi) oint d_r psi dS on |x - centre| = r: psi trilinear, d_r psi from +-h samples"""
    n, dw = sphere()
    h = 0.05 * r / 10
    vals = []
    for s in (r + h, r - h):
        P = centre[None, :] + s * n
        v = np.full(len(P), np.nan); best = np.full(len(P), np.inf)
        for x0, dx, adm in blks:
            ins = np.all((P >= x0) & (P < x0 + NX * dx), axis=1) & (dx < best)
            if not ins.any():
                continue
            psi = adm[12]**0.25
            q = (P[ins] - x0) / dx - 0.5 + NG              # fractional index in the ghosted array
            i0 = np.floor(q).astype(int); f = q - i0
            acc = 0
            for cx in (0, 1):
                for cy in (0, 1):
                    for cz in (0, 1):
                        wgt = ((f[:, 0] if cx else 1 - f[:, 0]) * (f[:, 1] if cy else 1 - f[:, 1])
                               * (f[:, 2] if cz else 1 - f[:, 2]))
                        acc = acc + wgt * psi[i0[:, 2] + cz, i0[:, 1] + cy, i0[:, 0] + cx]
            v[ins] = acc; best[ins] = dx
        vals.append(v)
    dpsi = (vals[0] - vals[1]) / (2 * h)
    return float(-(1 / (2 * np.pi)) * np.nansum(dpsi * dw) * r * r)


def run_file(fn, xb, kfix):
    blks = list(blocks(fn))
    out = {f'{a}-{b}': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for a, b in SHELLS}   # vol, sumH2, maxH, sumM2, maxM, sumscH2, sumscM2
    for x0, dx, adm in blks:
        H, sH, M, sM = constraints(adm, dx, kfix)
        c = x0[:, None] + (np.arange(NG + 1, NG + NX - 1) - NG + 0.5)[None, :] * dx
        Z, Y, X = np.meshgrid(c[2], c[1], c[0], indexing='ij')
        r = np.sqrt((X - xb[0])**2 + (Y - xb[1])**2 + (Z - xb[2])**2)
        dv = dx**3
        for a, b in SHELLS:
            sel = (r >= a) & (r < b)
            if not sel.any():
                continue
            o = out[f'{a}-{b}']
            o[0] += sel.sum() * dv
            o[1] += (H[sel]**2).sum() * dv; o[2] = max(o[2], np.abs(H[sel]).max())
            o[3] += (M[sel]**2).sum() * dv; o[4] = max(o[4], M[sel].max())
            o[5] += (sH[sel]**2).sum() * dv; o[6] += (sM[sel]**2).sum() * dv
    res = {}
    for k, o in out.items():
        res[k] = dict(H2=np.sqrt(o[1] / o[0]), Hmax=o[2], M2=np.sqrt(o[3] / o[0]), Mmax=o[4],
                      Hrel=np.sqrt(o[1] / o[5]), Mrel=np.sqrt(o[3] / o[6]))
    if kfix:
        res['mass_bh'] = [surface_mass(blks, xb, r) for r in RS]
        res['mass_origin'] = [surface_mass(blks, np.zeros(3), r) for r in RS]
    return res


def plot(png, npzs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols = {'static': 'k', 'xd025': '#1b9e77', 'xd045': '#d95f02'}
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), constrained_layout=True)
    for fn in npzs:
        z = np.load(fn, allow_pickle=True)
        an = z['an']
        for lab in [k for k in z.files if k != 'an']:
            rows = z[lab]
            t = np.array([r[0] for r in rows], float)
            c = cols.get(lab, None)
            for n, ls in zip(range(len(SHELLS)), ('-', '--', ':')):
                ax[0].semilogy(t, [r[2][n][0] for r in rows], 'o' + ls, color=c, ms=3,
                               label=f'{lab}, r_BH ∈ [{SHELLS[n][0]:.0f},{SHELLS[n][1]:.0f})')
                ax[1].semilogy(t, [r[2][n][2] for r in rows], 'o' + ls, color=c, ms=3)
                ax[2].semilogy(t, [r[5][n][0] for r in rows], 'o' + ls, color=c, ms=3)
            for n, mk in zip(range(len(RS)), ('o', 's', '^')):
                ax[3].plot(t, [r[3][n] - an[n] for r in rows], mk + '-', color=c, ms=4,
                           label=f'{lab}, r = {RS[n]:.0f}' if lab != 'xd025' else None)
    ax[0].set_title('Hamiltonian |H| (L2), K_ij = ψ⁶ vK_dd')
    ax[1].set_title('momentum |M_i| (L2), K_ij = ψ⁶ vK_dd')
    ax[2].set_title('H / |R| with vK_dd as stored (= K_ij in the hydro source)')
    ax[3].set_title('BH-centred ADM-mass integral − analytic trumpet')
    ax[3].axhline(0, color='0.5', lw=0.5)
    ax[0].legend(fontsize=7, ncol=1)
    ax[3].legend(fontsize=7, ncol=2)
    for x in ax:
        x.set_xlabel('t [M]'); x.grid(alpha=0.3)
    fig.suptitle('M1 test 1a: BH-centred constraints, static (black) vs ξ̇ = 0.25 (green) and 0.45 (orange)')
    fig.savefig(png, dpi=95)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == '--plot':
        plot(sys.argv[2], sys.argv[3:]); return
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    an = [analytic_mass(r) for r in RS]
    print('analytic trumpet M(r) at r = ' + ', '.join(f'{r:.0f}: {m:.6f}' for r, m in zip(RS, an)), flush=True)
    store = {}
    for spec in a.runs:
        lab, ds = spec.split('=', 1)
        rows = []
        for d in ds.split(','):
            txt = open(f'{d}/parfile.par').read()
            x0 = par(txt, 'cfc', 'puncture_x1'); v = par(txt, 'cfc', 'frame_vel1')
            mov = re.search(r'^puncture_moving\s*=\s*true', txt, re.M) is not None
            for fn in sorted(glob.glob(f'{d}/rst/*.rst')):
                t = T.header(fn)['time']
                xb = np.array([x0 - (v * t if mov else 0.0), 0.0, 0.0])
                r0 = run_file(fn, xb, False)
                r = run_file(fn, xb, True)
                line = f'{lab:7s} t={t:6.1f} x_BH={xb[0]:+6.1f} as-stored: ' + '  '.join(
                    f'[{k}] H/|R| {q["Hrel"]:.2e} |M|rel {q["Mrel"]:.2e}' for k, q in r0.items()) + '\n        fixed: '
                line += '  '.join(f'[{k}] H L2 {q["H2"]:.2e} max {q["Hmax"]:.2e} rel {q["Hrel"]:.1e}; |M| L2 {q["M2"]:.2e} max {q["Mmax"]:.2e} rel {q["Mrel"]:.1e}'
                                  for k, q in r.items() if k[0].isdigit())
                line += '\n        M_ADM-M_an (BH-centred) ' + ' '.join(f'r={R:.0f}: {m - x:+.2e}' for R, m, x in zip(RS, r['mass_bh'], an))
                line += ' | (origin-centred) ' + ' '.join(f'r={R:.0f}: {m - x:+.2e}' for R, m, x in zip(RS, r['mass_origin'], an))
                print(line, flush=True)
                rows.append((t, xb[0], [[r[k][q] for q in ('H2', 'Hmax', 'M2', 'Mmax', 'Hrel', 'Mrel')]
                                        for k in r if k[0].isdigit()], r['mass_bh'], r['mass_origin'],
                             [[r0[k][q] for q in ('Hrel', 'Mrel')] for k in r0]))
        store[lab] = rows
    if a.out:
        np.savez(a.out, an=np.array(an), **{k: np.array(v, dtype=object) for k, v in store.items()})


if __name__ == '__main__':
    main()
