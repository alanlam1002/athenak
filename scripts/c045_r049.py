#!/usr/bin/env python3
"""C-045 (R-049 + R-051 Hill-sphere metric): debris mass by refinement level, dM/dE by level,
stream width in cells, and the remnant against its Hill radius, from 3D restarts.

Per restart (tde_dmde.cells, rcut = 0, frame-aware; cells with r_BH < 3 dropped here):
- remnant = remnant.bound_mass's self-bound set; debris = everything else; bound/unbound
  debris by the BH-frame Killing E = -u_t - 1 < 0 / >= 0;
- mass of bound debris, unbound debris and remnant on each refinement level;
- stream width: per cell l = D/|grad D| (centred differences on the block incl. ghosts;
  D = rho W, the conserved density), in units of the cell's dx. Reported mass-weighted
  (median, p10) over debris cells with rho > 1e-6 rho_c, and the fraction of bound-wing
  mass (E < -0.5 dE) with l/dx < 4 and < 8;
- Hill: d = |x_rem - x_BH|, r_H = d (M_rem / 3 M_BH)^(1/3), the remnant's half- and
  90%-mass radii, and the mass self-bound within r_H (remnant.bound_mass on cells within
  r_H of the remnant centre, iterated to a fixed r_H).
dM/dE by level (bound + unbound debris, remnant excluded) is saved per restart to
c045_r049_<run>.npz.

    c045_r049.py [--runs R6,R7,Q3] [--nproc 16] [--out DIR]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import remnant as RM                              # noqa: E402

C = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc'
RHOC, MSTAR = 3.24e-4, 7.47e-5
DE = 0.677 / 16.4**2
EDGES = np.linspace(-6, 6, 121)
RUNS = {'R6': ('R6_rp10_comov_L4_fix', 'R6x_rp10_comov_L4_t300'),
        'R7': ('R7_rp18_comov_L4_fix',), 'Q3': ('Q3_P2_rp18_L4',),
        'Q1c': ('D_e_Q1deck_kfix_t80', 'RQ1_bh_L4_t300_from_De')}


def ell_over_dx(fn, nscal=1):
    """per cell l/dx = D/|grad D|/dx, in tde_dmde.cells(rcut=0) order (same ok-mask)"""
    t = D.header(fn)
    nmb, nx, ng = t['nmb'], 16, 4
    no = nx + 2 * ng
    nmhd = 5 + nscal
    ncc = no**3
    bsz = (no + 1) * no * no * 3
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(nmb, t['dsize'] // 8))
    s = slice(ng, ng + nx)
    out = []
    for b in range(nmb):
        rec = mm[b]
        cons = rec[:nmhd * ncc].reshape(nmhd, no, no, no)
        adm = rec[nmhd * ncc + bsz:].reshape(17, no, no, no)
        g = adm[:6]
        det = (g[0] * (g[3] * g[5] - g[4] * g[4]) - g[1] * (g[1] * g[5] - g[4] * g[2])
               + g[2] * (g[1] * g[4] - g[3] * g[2]))
        Dd = cons[0] / np.sqrt(det)
        gz, gy, gx = np.gradient(Dd)                  # per cell index; same dx in all directions
        l = Dd / np.maximum(np.sqrt(gx**2 + gy**2 + gz**2), 1e-300)
        Di, taui = Dd[s, s, s], (cons[4] / np.sqrt(det))[s, s, s]
        ok = (Di > 0) & np.isfinite(taui)
        out.append(l[s, s, s][ok])
    return np.concatenate(out)


def hill(c, Mb, xc, xbh):
    X = np.stack([c['x'], c['y'], c['z']])
    d = np.linalg.norm(xc - xbh)
    rH = d * (Mb / 3.0)**(1 / 3)
    Mh = Mb
    for _ in range(20):                               # self-bound mass within r_H, r_H(M) fixed point
        r = np.linalg.norm(X - xc[:, None], axis=0)
        sub = {k: (v[r < rH] if isinstance(v, np.ndarray) and v.shape == r.shape else v) for k, v in c.items()}
        if sub['dM'].size < 10:
            Mh = 0.0; break
        Mh_new = RM.bound_mass(sub)[0]
        rH_new = d * (Mh_new / 3.0)**(1 / 3)
        if abs(rH_new - rH) < 1e-4 * rH:
            Mh = Mh_new; rH = rH_new; break
        Mh, rH = Mh_new, rH_new
    return d, rH, Mh


def one(fn):
    cache = f'{C}/plots/phase2_m1/c045_r049_cache/' + fn.split('/')[-3] + '_' + fn.split('.')[-2] + '.npy'
    if os.path.exists(cache):
        return np.load(cache, allow_pickle=True).item()
    r = _one(fn)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, r, allow_pickle=True)
    return r


def _one(fn):
    c = D.cells(fn, rcut=0.0)
    lx = ell_over_dx(fn)
    xbh = np.asarray(c['xbh'])
    rbh = np.sqrt((c['x'] - xbh[0])**2 + (c['y'] - xbh[1])**2 + (c['z'] - xbh[2])**2)
    k = rbh >= 3.0
    c = {kk: (v[k] if isinstance(v, np.ndarray) and v.shape == k.shape else v) for kk, v in c.items()}
    lx = lx[k]
    Mb, xc, vc, rem = RM.bound_mass(c)
    E, dM, lev = c['E'], c['dM'], c['lev'].astype(int)
    deb = ~rem
    levels = np.arange(0, 7)
    per = np.array([[dM[deb & (E < 0) & (lev == L)].sum(), dM[deb & (E >= 0) & (lev == L)].sum(),
                     dM[rem & (lev == L)].sum()] for L in levels]) / MSTAR
    H = np.array([np.histogram(E[deb & (lev == L)] / DE, bins=EDGES, weights=dM[deb & (lev == L)] / MSTAR)[0]
                  for L in levels])
    st = deb & (c['rho'] > 1e-6 * RHOC)
    w = dM[st]; v = lx[st]
    if w.size:
        o = np.argsort(v); cw = np.cumsum(w[o]) / w.sum()
        med, p10 = v[o][min(np.searchsorted(cw, 0.5), v.size - 1)], v[o][min(np.searchsorted(cw, 0.1), v.size - 1)]
    else:
        med = p10 = np.nan                            # no debris yet (star intact)
    wing = st & (E < -0.5 * DE)
    f4 = dM[wing & (lx < 4)].sum() / max(dM[wing].sum(), 1e-300)
    f8 = dM[wing & (lx < 8)].sum() / max(dM[wing].sum(), 1e-300)
    wl = {L: dM[wing & (lev == L)].sum() / max(dM[wing].sum(), 1e-300) for L in levels}
    X = np.stack([c['x'], c['y'], c['z']]); r = np.linalg.norm(X[:, rem] - xc[:, None], axis=0)
    o = np.argsort(r); cm = np.cumsum(dM[rem][o]) / dM[rem].sum()
    R50, R90 = r[o][np.searchsorted(cm, 0.5)], r[o][np.searchsorted(cm, 0.9)]
    d, rH, Mh = hill(c, Mb, xc, xbh)
    return dict(t=c['t'], per=per, H=H, med=med, p10=p10, f4=f4, f8=f8, wl=wl, Mb=Mb / MSTAR,
                R50=R50, R90=R90, d=d, rH=rH, Mh=Mh / MSTAR, Lmax=lev.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='R6,R7,Q3')
    ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--out', default=f'{C}/plots/phase2_m1')
    a = ap.parse_args()
    labs = a.runs.split(',')
    files = {}
    for lab in labs:
        byt = {}
        for s in RUNS[lab]:
            for f in sorted(glob.glob(f'{C}/{s}/rst/*.rst')):
                byt[round(D.header(f)['time'], 1)] = f        # later segments win
        files[lab] = [byt[t] for t in sorted(byt) if t > 0]
    jobs = [f for v in files.values() for f in v]
    with Pool(a.nproc) as p:
        res = dict(zip(jobs, p.map(one, jobs)))
    rows = []
    for lab in labs:
        R = [res[f] for f in files[lab]]
        np.savez(f'{a.out}/c045_r049_{lab}.npz', edges=EDGES, t=[r['t'] for r in R],
                 H=np.array([r['H'] for r in R]), per=np.array([r['per'] for r in R]))
        rows.append(f'== {lab}: debris mass by level, bound / unbound / remnant [M*]; stream l/dx; Hill')
        for r in R:
            lv = ' '.join(f'L{L}:{r["per"][L, 0]:.3f}/{r["per"][L, 1]:.3f}/{r["per"][L, 2]:.3f}'
                          for L in range(7) if r['per'][L].sum() > 1e-4)
            wl = ' '.join(f'L{L} {v:.2f}' for L, v in r['wl'].items() if v > 0.005)
            rows.append(f'  t={r["t"]:6.1f} {lv} | l/dx med {r["med"]:.1f} p10 {r["p10"]:.1f}; '
                        f'wing(E<-0.5dE) l/dx<4 {r["f4"]:.2f} <8 {r["f8"]:.2f}; wing by level {wl} | '
                        f'remnant {r["Mb"]:.4f} M* R50 {r["R50"]:.2f} R90 {r["R90"]:.2f} d {r["d"]:.1f} '
                        f'r_H {r["rH"]:.2f} r_H/R90 {r["rH"] / r["R90"]:.2f} M(<r_H, bound) {r["Mh"]:.4f}')
    with open(f'{a.out}/c045_r049.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
