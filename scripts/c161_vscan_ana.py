#!/usr/bin/env python3
"""c161_vscan (R-201 addendum 3): the ISO_A speed scan at rms 1e-8 + mg_min_vcycles 1 (L4, CFL 0.4, DE off, t = 0 -> 120).
Arms ISOB (v = 0), V1/V2/V4 (Valencia, v = 0.1/0.2/0.4), K1/K2/K4 (tau_source_K = keff).  Per arm:
  - the dense-band drift: rho dV-weighted K/K0 - 1 over rho >= 0.01 and >= 0.1 rho_c at the prim_3d dumps nearest t = 30/60/120;
  - R and |T|_coord: TRD step means (c160_trd_ana.trd_lines) over t in [10,30), [30,60), [60,120), bands b0 and b1;
  - cost (c161_tol_ana.cost) and MGV skip stats (c161_tol_ana.mgv).
Then keff - Valencia per v, and the fitted power p in |drift(v) - drift(ISOB)| ~ v^p (least squares in log-log over v = 0.1/0.2/0.4,
plus the two pairwise slopes; a sign change across v is flagged and makes the fit meaningless).
    c161_vscan_ana.py SCANDIR OUT.csv"""
import glob, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_trd_ana import trd_lines   # noqa: E402
from c161_tol_ana import cost, mgv, RC, K0, G   # noqa: E402

TS = (30.0, 60.0, 120.0)
WIN = ((10, 30), (30, 60), (60, 120))
VS = (0.1, 0.2, 0.4)
NM = ('1', '2', '4')


def drift(d):
    o = {}
    fs = sorted(glob.glob(f'{d}/bin/*prim_3d.*.bin'))
    ts = np.array([bc.read_binary(f)['time'] for f in fs])
    for tt in TS:
        i = int(np.argmin(np.abs(ts - tt)))
        P = bc.read_binary(fs[i]); g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']
        r = np.array([np.asarray(P['mb_data']['dens'][b], float) for b in range(len(g))])
        p = np.array([np.asarray(P['mb_data']['press'][b], float) for b in range(len(g))])
        dv = (((g[:, 1] - g[:, 0])/n)**3)[:, None, None, None]; K = p/r**G/K0; w = r*dv
        o[f't_dump{tt:.0f}'] = float(ts[i])
        for th, lab in ((0.01, 'd01'), (0.1, 'd1')):
            m = r >= th*RC
            o[f'{lab}_t{tt:.0f}'] = float((w[m]*(K[m] - 1)).sum()/w[m].sum())
    return o


def trd(d):
    o = {}
    _, steps = trd_lines(d)
    S = {k: np.array([s[k] for s in steps]) for k in steps[0]}
    for lo, hi in WIN:
        m = (S['t'] >= lo) & (S['t'] < hi)
        for b in (0, 1):
            o[f'R_b{b}_{lo}_{hi}'] = float(S[f'b{b}_R'][m].mean()) if m.any() else np.nan
            o[f'T_b{b}_{lo}_{hi}'] = float(S[f'b{b}_T'][m].mean()) if m.any() else np.nan
    return o


def fit(y):
    y = np.asarray(y); lv = np.log(VS)
    if np.any(y == 0) or len(set(np.sign(y))) > 1:
        return np.nan, np.nan, np.nan, 'sign change'
    ly = np.log(np.abs(y))
    p = np.polyfit(lv, ly, 1)[0]
    return p, (ly[1] - ly[0])/(lv[1] - lv[0]), (ly[2] - ly[1])/(lv[2] - lv[1]), ''


def main():
    sd, out = sys.argv[1], sys.argv[2]
    rows = {}
    for a in ('ISOB', 'V1', 'K1', 'V2', 'K2', 'V4', 'K4'):
        o = {'arm': a}
        for f in (drift, trd, cost, mgv):
            o.update(f(f'{sd}/{a}'))
        rows[a] = o
        print(f"{a:4s} d01 t30/60/120 {100*o['d01_t30']:+.4f}/{100*o['d01_t60']:+.4f}/{100*o['d01_t120']:+.4f}% "
              f"d1 {100*o['d1_t30']:+.4f}/{100*o['d1_t60']:+.4f}/{100*o['d1_t120']:+.4f}% | "
              + ' '.join(f"[{lo},{hi}) R b0/b1 {o[f'R_b0_{lo}_{hi}']:+.2e}/{o[f'R_b1_{lo}_{hi}']:+.2e} "
                         f"|T| {o[f'T_b0_{lo}_{hi}']:.2e}/{o[f'T_b1_{lo}_{hi}']:.2e}" for lo, hi in WIN)
              + f" | {o['cycles']} cyc {o['wall_per_cycle']:.3f} s/cyc psi skip {o.get('psi_skip', np.nan):.2f} vc/cyc {o.get('vc_per_cycle', np.nan):.1f}",
              flush=True)
    print('\nkeff - Valencia (percentage points; R in 1/M):')
    for i, v in enumerate(VS, 1):
        V, K = rows[f'V{NM[i-1]}'], rows[f'K{NM[i-1]}']
        print(f"v={v}: " + ' '.join(f"{lab} t{tt:.0f} {100*(K[f'{lab}_t{tt:.0f}'] - V[f'{lab}_t{tt:.0f}']):+.4f}"
                                     for lab in ('d01', 'd1') for tt in TS)
              + ' | ' + ' '.join(f"R b1 [{lo},{hi}) {K[f'R_b1_{lo}_{hi}'] - V[f'R_b1_{lo}_{hi}']:+.2e}" for lo, hi in WIN))
    print('\nfitted power p, |q(v) - q(ISOB)| ~ v^p (lsq; pairwise 0.1->0.2, 0.2->0.4):')
    fits = []
    for sch in ('V', 'K'):
        qs = [f'{lab}_t{tt:.0f}' for lab in ('d01', 'd1') for tt in TS] + [f'R_b{b}_{lo}_{hi}' for b in (0, 1) for lo, hi in WIN]
        for q in qs:
            y = [rows[f'{sch}{n}'][q] - rows['ISOB'][q] for n in NM]
            p, p1, p2, fl = fit(y)
            fits.append((sch, q, *y, p, p1, p2, fl))
            print(f"{sch} {q:14s} dq {y[0]:+.3e} {y[1]:+.3e} {y[2]:+.3e}  p {p:5.2f} ({p1:5.2f}, {p2:5.2f}) {fl}")
    keys = []
    for r in rows.values():
        keys += [k for k in r if k not in keys]
    with open(out, 'w') as fo:
        fo.write(','.join(keys) + '\n')
        for r in rows.values():
            fo.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
    with open(out.replace('.csv', '_fit.csv'), 'w') as fo:
        fo.write('scheme,quantity,dq_v0.1,dq_v0.2,dq_v0.4,p_lsq,p_01_02,p_02_04,flag\n')
        for f in fits:
            fo.write(','.join(str(x) for x in f) + '\n')


if __name__ == '__main__':
    main()
