#!/usr/bin/env python3
"""C-049 (R-059 item 2): offline check of R-058 §3's tier-3 activation rule.

Single-tier zone always on (r_s = min(6, 0.5 r_p), rho < 1e-10 rho_c, rho reset; c047_s0b's
'single'). Tier 3 (r_s <= r < 2 r_s, rho < 1e-16 rho_c, velocity-only grid rest, cells within
3 R_90 of the dense-gas centroid excluded) acts only if BOTH
  - |d| > 2 r_s + 3 R_90, d = x_c - x_BH, x_c the mass-weighted centroid of rho > 1e-3 rho_max
    and R_90 the radius about x_c enclosing 90% of that mass;
  - receding, d . ddot > 0, with ddot = <alpha v - beta'>_dense + xidot: the dense gas's
    mass-weighted coordinate velocity on the comoving grid minus the BH's (x_BH = puncture_x
    - xi(t), so dx_BH/dt = -xidot; equivalently <alpha v - beta_BH> with beta_BH = beta' - xidot).
Per restart: d, R_90, d.ddot, active?, dt for baseline / single / single + tier 3 (also shown
when inactive, as 'would be'), the limiter, and the tier-3 velocity-modified mass per E bin as
a fraction of the grid's mass in that bin (r_BH >= 3). Pass (R-058 §3): <= 1e-6 in every
E > 0 bin whenever active; inactive at R6 129.6 and 144.8.

    c049_t3rule.py [--out DIR] [--only R6]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tde_dmde as D                              # noqa: E402
import c046_s0 as S                               # noqa: E402
import c047_s0b as B47                            # noqa: E402
from c045_dt import rst_at, RHOC                  # noqa: E402

MSTAR, DE = B47.MSTAR, B47.DE
EB = B47.EB
SHELL3 = 1e-16
S.RCAND = 12.0


def alpha_beta(fn, nscal=1):
    """alpha and beta'^i per cell, same order and mask as tde_dmde.cells(fn, rcut=0)"""
    t = D.header(fn)
    nx, ng = 16, 4
    no = nx + 2 * ng; nmhd = 5 + nscal; ncc = no**3; bsz = (no + 1) * no * no * 3
    mm = np.memmap(fn, dtype=np.float64, mode='r', offset=t['data_off'], shape=(t['nmb'], t['dsize'] // 8))
    s = slice(ng, ng + nx)
    out = [[], [], [], []]
    for b in range(t['nmb']):
        rec = mm[b]
        cons = rec[:nmhd * ncc].reshape(nmhd, no, no, no)[:, s, s, s]
        adm = rec[nmhd * ncc + bsz:].reshape(17, no, no, no)[:, s, s, s]
        gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
        det = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) + gxz * (gxy * gyz - gyy * gxz)
        sg = np.sqrt(det)
        ok = (cons[0] / sg > 0) & np.isfinite(cons[4] / sg)
        for i, q in enumerate((adm[13], adm[14], adm[15], adm[16])):
            out[i].append(q[ok])
    return [np.concatenate(o) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{S.C}/plots/phase2_m1')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    rows = ['R-059 item 2: tier-3 activation rule (R-058 §3), offline (c049_t3rule.py). Masses in M*.',
            f'tier 3: r_s <= r < 2 r_s, rho < {SHELL3:g} rho_c, velocity-only grid rest, 3 R_90 exclusion; '
            'active iff |d| > 2 r_s + 3 R_90 and d.ddot > 0.',
            'E bins (E/dE) edges ' + ' '.join(f'{e:g}' for e in EB) + '; "frac" = tier-3 modified mass / grid mass (r_BH >= 3) per bin.']
    verdict = []
    for lab, d, rs, ts in B47.CASES:
        if a.only and lab not in a.only.split(','):
            continue
        for tt in ts:
            fn = rst_at(d, tt)
            t, B, Cc = S.load(fn)
            c = D.cells(fn, rcut=0.0)
            al, b0, b1, b2 = alpha_beta(fn)
            assert al.size == c['dM'].size == B['r'].size
            cand = B['r'] < S.RCAND
            assert cand.sum() == Cc['r'].size and np.allclose(c['dM'][cand], Cc['dM'])
            xb = np.asarray(c['xbh'])
            _, xdot = D.frame_state(c['par'], c['t'])
            Cc.update(E=c['E'][cand], X=c['x'][cand], Y=c['y'][cand], Z=c['z'][cand], L=np.zeros(cand.sum()),
                      bidx=np.nonzero(cand)[0])
            dense = c['rho'] > 1e-3 * c['rho'].max()
            w = c['dM'][dense]
            P = np.stack([c['x'][dense], c['y'][dense], c['z'][dense]])
            xc = (P * w).sum(1) / w.sum()
            rr = np.linalg.norm(P - xc[:, None], axis=0)
            o = np.argsort(rr); R90 = np.interp(0.9, np.cumsum(w[o]) / w.sum(), rr[o])
            vel = np.stack([al[dense] * c['vx'][dense] - b0[dense], al[dense] * c['vy'][dense] - b1[dense],
                            al[dense] * c['vz'][dense] - b2[dense]])
            ddot = (vel * w).sum(1) / w.sum() + np.asarray(xdot)
            dv = xc - xb
            dist = np.linalg.norm(dv); ddd = float(dv @ ddot)
            thr = 2 * rs + 3 * R90
            active = dist > thr and ddd > 0
            far3 = np.sqrt((c['x'] - xb[0])**2 + (c['y'] - xb[1])**2 + (c['z'] - xb[2])**2) >= 3.0
            ref = np.histogram(c['E'][far3] / DE, EB, weights=c['dM'][far3])[0] / MSTAR
            r, rho, exc = Cc['r'], Cc['rho'], Cc['exc']
            single = (r < rs) & (rho < B47.THR * RHOC) & ~exc
            away = np.sqrt((Cc['X'] - xc[0])**2 + (Cc['Y'] - xc[1])**2 + (Cc['Z'] - xc[2])**2) > 3 * R90
            t3 = (r >= rs) & (r < 2 * rs) & (rho < SHELL3 * RHOC) & away & ~exc
            dt0 = B['dt'].min()
            L1 = B47.dt_with(B, Cc, single)
            L3 = B47.dt_with(B, Cc, single | t3)
            rows.append(f'== {lab} t = {t:.1f} (r_s {rs:g}): d {dist:.2f}, R_90 {R90:.3f}, 2 r_s + 3 R_90 {thr:.2f}, '
                        f'd.ddot {ddd:+.3e} ({"receding" if ddd > 0 else "approaching"}), ddot {np.linalg.norm(ddot):.4f} '
                        f'-> tier 3 {"ACTIVE" if active else "inactive"} (far: {dist > thr}, receding: {ddd > 0})')
            fmt = lambda L: (f'dt {L["dt"]:.4f} (x{L["dt"] / dt0:.2f}) lim [{L["where"]}, L{L["lev"]}, r_BH {L["r"]:.2f}, '
                             f'rho {L["rho"]:.1e}]')
            rows.append(f'  baseline dt {dt0:.4f}; single {fmt(L1)}')
            rows.append(f'  single + tier 3 {"" if active else "(would be; rule keeps it off) "}{fmt(L3)}; '
                        f'x{L3["dt"] / L1["dt"]:.2f} over single')
            if t3.any():
                rr_, eb, fr, _ = B47.binned(t3, Cc, ref)
                rows.append(f'  tier-3 v-modified {t3.sum()} cells, M {eb.sum():.1e}, max rho {rho[t3].max() / RHOC:.1e} rho_c')
                rows.append('    E bins ' + ' '.join(f'{x:.1e}' for x in eb))
                rows.append('    frac   ' + ' '.join(f'{x:.1e}' for x in fr))
                fpos = np.nanmax(np.where(EB[:-1] >= 0, np.nan_to_num(fr, nan=0.0), 0.0))
            else:
                fpos = 0.0
                rows.append('  tier-3 mask empty')
            rows.append(f'  max frac over E > 0 bins {fpos:.1e}')
            req = lab == 'R6' and tt in (129.6, 144.8)
            v = f'{lab} {t:.1f}: ' + ('ACTIVE' if active else 'inactive')
            if req:
                v += ' (must be inactive) -> ' + ('FAIL' if active else 'PASS')
            if active:
                v += f'; max E>0 frac {fpos:.1e} (<= 1e-6) -> {"PASS" if fpos <= 1e-6 else "FAIL"}'
            verdict.append(v)
            print('\n'.join(rows[-7:]), flush=True)
    rows += ['', 'Verdict:'] + ['  ' + v for v in verdict]
    print('\n'.join(rows[-len(verdict) - 1:]))
    if not a.only:
        with open(f'{a.out}/c049_t3rule.txt', 'w') as f:
            f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
