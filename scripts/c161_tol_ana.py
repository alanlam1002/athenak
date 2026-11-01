#!/usr/bin/env python3
"""c161_mgv (R-201 item 1 with addenda 2-3): the CFC tolerance study on ISO_A (Valencia, DE off, L4, CFL 0.4, t = 0 -> 60).
Per run directory:
  - the dense-band drift: rho dV-weighted K/K0 - 1 over rho >= 0.01 and >= 0.1 rho_c at the prim_3d dumps nearest t = 30 and 60;
  - R and |T|_coord: the TRD step means (c160_trd_ana.trd_lines: dtm-weighted over each step's 3 stages) over t in [10, 30) and
    [30, 60), bands b0 (rho >= 0.5 rho_c) and b1 (>= 0.1 rho_c);
  - the skip pattern (MGV lines, <cfc> mg_log_vcycles): per solver (X, psi, alpha psi, beta), the share of the per-stage solves
    that did no V-cycle and the mean V-cycles per solve; for psi, the share of each RK stage (1, 2, 3) on which it did >= 1
    V-cycle (the stages on which psi can change: a solve with no V-cycle returns the warm start unchanged);
  - the cost: cycles, wall (out.log "cpu time used" = the run's wall clock per rank), wall per cycle.
    c161_tol_ana.py OUT.csv LABEL=RUNDIR [LABEL=RUNDIR ...]"""
import glob, re, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_trd_ana import trd_lines   # noqa: E402

RC, K0, G = 3.24e-4, 1.2346626297e-2, 5.0/3.0
SOL = ('X', 'psi', 'ap', 'beta')


def drift(d):
    o = {}
    fs = sorted(glob.glob(f'{d}/bin/*prim_3d.*.bin'))
    ts = np.array([bc.read_binary(f)['time'] for f in fs])
    for tt in (30.0, 60.0):
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


def mgv(d):
    rows = []
    for l in open(f'{d}/out.log'):
        if l.startswith('MGV '):
            h = re.search(r'cycle= (\d+) stage= (\d+)', l)
            v = dict((k, int(n)) for k, n in re.findall(r' (\w+)= (-?\d+),', l))
            rows.append((int(h.group(1)), int(h.group(2)), v))
    if not rows:
        return {}
    rows = [r for r in rows if r[0] > 0]   # cycle 0 = the initial solve
    o = {'mgv_lines': len(rows)}
    for s in SOL:
        n = np.array([r[2][s] for r in rows if r[2][s] >= 0])
        o[f'{s}_skip'] = float((n == 0).mean()); o[f'{s}_vc'] = float(n.mean())
    for st in (1, 2, 3):
        n = np.array([r[2]['psi'] for r in rows if r[1] == st and r[2]['psi'] >= 0])
        o[f'psi_moves_s{st}'] = float((n > 0).mean()) if len(n) else np.nan
    o['vc_per_cycle'] = float(sum(max(r[2][s], 0) for r in rows for s in SOL)/len(set(r[0] for r in rows)))
    return o


def cost(d):
    s = open(f'{d}/out.log').read()
    cyc = [int(x) for x in re.findall(r'cycle=(\d+) time=', s)]
    w = re.findall(r'cpu time used\s*=\s*(\S+)', s)
    o = {'cycles': max(cyc) if cyc else -1, 'wall': float(w[-1]) if w else np.nan}
    o['wall_per_cycle'] = o['wall']/o['cycles'] if o['cycles'] > 0 else np.nan
    return o


def trd(d):
    o = {}
    _, steps = trd_lines(d)
    S = {k: np.array([s[k] for s in steps]) for k in steps[0]}
    for lo, hi in ((10, 30), (30, 60)):
        m = (S['t'] >= lo) & (S['t'] < hi)
        for b in (0, 1):
            o[f'R_b{b}_{lo}_{hi}'] = float(S[f'b{b}_R'][m].mean())
            o[f'T_b{b}_{lo}_{hi}'] = float(S[f'b{b}_T'][m].mean())
    return o


def main():
    out = sys.argv[1]
    rows = []
    for a in sys.argv[2:]:
        lab, d = a.split('=', 1)
        o = {'arm': lab, 'dir': d}
        for f in (drift, trd, cost, mgv):
            o.update(f(d))
        rows.append(o)
        print(f"{lab:6s} d01 t30 {100*o['d01_t30']:+.3f}% t60 {100*o['d01_t60']:+.3f}% | d1 t60 {100*o['d1_t60']:+.3f}% "
              f"| R b1 [30,60) {o['R_b1_30_60']:+.3e} b0 {o['R_b0_30_60']:+.3e} | |T| b1 {o['T_b1_30_60']:.3e} b0 {o['T_b0_30_60']:.3e} "
              f"| {o['cycles']} cyc {o['wall']:.0f} s {o['wall_per_cycle']:.3f} s/cyc"
              + (f" | skip X/psi/ap/beta {o['X_skip']:.2f}/{o['psi_skip']:.2f}/{o['ap_skip']:.2f}/{o['beta_skip']:.2f} "
                 f"vc/solve {o['X_vc']:.2f}/{o['psi_vc']:.2f}/{o['ap_vc']:.2f}/{o['beta_vc']:.2f} vc/cycle {o['vc_per_cycle']:.1f} "
                 f"psi moves s1/s2/s3 {o['psi_moves_s1']:.2f}/{o['psi_moves_s2']:.2f}/{o['psi_moves_s3']:.2f}" if 'X_skip' in o else ''),
              flush=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out, 'w') as fo:
        fo.write(','.join(keys) + '\n')
        for r in rows:
            fo.write(','.join(str(r.get(k, '')) for k in keys) + '\n')


if __name__ == '__main__':
    main()
