#!/usr/bin/env python3
"""c160_trd (R-197 item 1 and addendum): the trace mismatch R = 6 d_t ln psi - D_k beta'^k and the TT remainder
T_ij = (L beta')_ij - 2 alpha psi^-6 Ahat_ij on ISO_A (wip/trace-diag; definitions in dyn_grmhd.hpp kTrN / kTrRed).
A. TRD lines (every stage): rho dV-weighted band means (bands rho >= 0.5, 0.1, 0.01 rho_c) of R, |T|_coord, 6 d_t ln psi,
   D_k beta'^k, and the band integrals of the two tau-source parts. Per stage, and per step as the dtm-weighted mean over the
   step's three stages (the psi^4 chord then telescopes: one chord between consecutive tau = 1/2 states, over dt).
B. mhd_trace_diag dumps (the step's last stage) with the prim_3d dump at the same time: per band, R and |T| mean / rms /
   5-50-95 percentiles (rho dV weights), the rho dV-weighted correlation of R with |T| and with 1/2 psi^4 S^ij T_ij-free
   measures, the trace delta^ij T_ij (max |.| / max |T|), and the divergence d_j T_ij (central differences inside each
   MeshBlock, cells >= 1 from its faces), relative to |T| / dx.
C. The window R per cell between consecutive dumps (the per-stage chord moves in jumps: a warm-started MG solve whose initial
   defect is below mg_threshold does no V-cycle, multigrid_driver.cpp:960-969): Rw = 6 ln(psi(t2)/psi(t1))/(t2 - t1) - the mean
   of D_k beta'^k at t1 and t2 (adm psi4, tr_divb), same MeshBlocks at both ends; per band (rho at t2) mean / rms / percentiles,
   and its rho dV-weighted correlation with the mean |T|.
    c160_trd_ana.py OUT.csv RUNDIR[,RUNDIR...] [--rhoc 3.24e-4]"""
import glob, os, re, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'
BANDS = (0.5, 0.1, 0.01)
NAMES = ['w', 'wR', 'wR2', 'wT', 'wT2', 'wRT', 'w6dt', 'wdivb', 'sig_tt', 'sig_r']


def trd_lines(d):
    st, sp = [], {}
    for l in open(f'{d}/out.log'):
        if not l.startswith('TRD '):
            continue
        h = dict(re.findall(r'(\w+)=(\S+)', l))
        r = {'cycle': int(h['cycle']), 'stage': int(h['stage']), 't': float(h['time']), 'tst': float(h['tstage']),
             'dtm': float(h['dtm']), 'valid': int(h['valid'])}
        for b in range(len(BANDS)):
            v = [float(x) for x in h[f'b{b}'].split(',')]
            r.update({f'b{b}_{n}': x for n, x in zip(NAMES, v)})
        st.append(r)
    # per step: the dtm-weighted mean over its stages (all valid)
    for r in st:
        sp.setdefault(r['cycle'], []).append(r)
    steps = []
    for c, rs in sorted(sp.items()):
        if len(rs) != 3 or not all(x['valid'] for x in rs):
            continue
        dt = sum(x['dtm'] for x in rs)
        o = {'cycle': c, 't': rs[0]['t'], 'dt': dt}
        for b in range(len(BANDS)):
            for q in ('R', '6dt', 'divb', 'T'):
                k = {'R': 'wR', '6dt': 'w6dt', 'divb': 'wdivb', 'T': 'wT'}[q]
                o[f'b{b}_{q}'] = sum(x['dtm']*x[f'b{b}_{k}']/x[f'b{b}_w'] for x in rs)/dt if q != 'T' else \
                    np.mean([x[f'b{b}_wT']/x[f'b{b}_w'] for x in rs])
            for s in (1, 2, 3):
                x = [y for y in rs if y['stage'] == s][0]
                o[f'b{b}_R_s{s}'] = x[f'b{b}_wR']/x[f'b{b}_w']
            o[f'b{b}_sig_tt'] = np.mean([x[f'b{b}_sig_tt'] for x in rs])
            o[f'b{b}_sig_r'] = np.mean([x[f'b{b}_sig_r'] for x in rs])
        steps.append(o)
    return st, steps


def cells(P):
    g = np.asarray(P['mb_geometry'], dtype=float); n = P['nx1_out_mb']
    return g, n


def dumps(d, rhoc):
    out = []
    ft = {round(bc.read_binary(f)['time'], 6): f for f in sorted(glob.glob(f'{d}/bin/{B}.trd_3d.*.bin'))}
    fp = {round(bc.read_binary(f)['time'], 6): f for f in sorted(glob.glob(f'{d}/bin/{B}.prim_3d.*.bin'))}
    prev = None
    A2 = {round(bc.read_binary(f)['time'], 6): f for f in sorted(glob.glob(f'{d}/bin/{B}.adm_3d.*.bin'))}
    for t in sorted(set(ft) & set(fp) & set(A2)):
        T, P = bc.read_binary(ft[t]), bc.read_binary(fp[t])
        g, n = cells(T)
        q = T['mb_data']
        A = lambda k: np.array([np.asarray(q[k][b], dtype=float) for b in range(len(g))])
        rho = np.array([np.asarray(P['mb_data']['dens'][b], dtype=float) for b in range(len(g))])
        R, Tn, tr, val = A('tr_R'), A('tr_Tnorm'), A('tr_Ttrace'), A('tr_valid')
        Tc = {c: A(f'tr_T{c}') for c in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz')}
        dx = (g[:, 1] - g[:, 0])/n
        dv = (dx**3)[:, None, None, None]*np.ones_like(rho)
        # d_j T_ij, central, cells >= 1 from the block faces (axis order k, j, i)
        def d_(f, ax):
            o = np.full_like(f, np.nan)
            sl = [slice(None)]*4; sp = list(sl); sm = list(sl); sl[ax + 1] = slice(1, -1)
            sp[ax + 1] = slice(2, None); sm[ax + 1] = slice(None, -2)
            o[tuple(sl)] = (f[tuple(sp)] - f[tuple(sm)])/(2*dx[:, None, None, None])
            return o
        X, Y, Z = 2, 1, 0
        div = [d_(Tc['xx'], X) + d_(Tc['xy'], Y) + d_(Tc['xz'], Z),
               d_(Tc['xy'], X) + d_(Tc['yy'], Y) + d_(Tc['yz'], Z),
               d_(Tc['xz'], X) + d_(Tc['yz'], Y) + d_(Tc['zz'], Z)]
        dn = np.sqrt(sum(x**2 for x in div))
        o = {'t': t, 'valid_frac': float(val.mean())}
        for bi, f in enumerate(BANDS):
            m = rho >= f*rhoc
            w = (rho*dv)[m]; r = R[m]; tn = Tn[m]
            W = w.sum()
            mr, mt = (w*r).sum()/W, (w*tn).sum()/W
            o[f'b{bi}_R_mean'] = mr
            o[f'b{bi}_R_rms'] = np.sqrt((w*r*r).sum()/W)
            o[f'b{bi}_R_p05'], o[f'b{bi}_R_p50'], o[f'b{bi}_R_p95'] = np.percentile(r, [5, 50, 95])
            o[f'b{bi}_T_mean'] = mt
            o[f'b{bi}_T_p95'] = np.percentile(tn, 95)
            cv = (w*(r - mr)*(tn - mt)).sum()/W
            o[f'b{bi}_corr_R_T'] = cv/np.sqrt((w*(r - mr)**2).sum()/W*(w*(tn - mt)**2).sum()/W)
            o[f'b{bi}_corr_absR_T'] = np.corrcoef(np.abs(r), tn)[0, 1]
            mm = m & np.isfinite(dn)
            o[f'b{bi}_divT_over_T_dx'] = float(np.median(dn[mm]*np.broadcast_to(dx[:, None, None, None], dn.shape)[mm] /
                                                         np.maximum(Tn[mm], 1e-300)))
            o[f'b{bi}_trace_over_T'] = float(np.abs(tr[m]).max()/max(np.abs(tn).max(), 1e-300))
        # C: the window R against the previous common dump
        if t in A2 and prev is not None and prev[0] in A2 and np.array_equal(prev[1], g):
            p2 = np.array([np.asarray(bc.read_binary(A2[t])['mb_data']['adm_psi4'][b], dtype=float) for b in range(len(g))])
            p1 = np.array([np.asarray(bc.read_binary(A2[prev[0]])['mb_data']['adm_psi4'][b], dtype=float) for b in range(len(g))])
            dvb = A('tr_divb')
            rw = 1.5*np.log(p2/p1)/(t - prev[0]) - 0.5*(dvb + prev[2])
            tm = 0.5*(Tn + prev[3])
            o['win_t1'] = prev[0]
            for bi, f in enumerate(BANDS):
                m = rho >= f*rhoc
                w = (rho*dv)[m]; r = rw[m]; tn = tm[m]; W = w.sum()
                mr, mt = (w*r).sum()/W, (w*tn).sum()/W
                o[f'b{bi}_Rw_mean'] = mr
                o[f'b{bi}_Rw_rms'] = np.sqrt((w*r*r).sum()/W)
                o[f'b{bi}_Rw_p05'], o[f'b{bi}_Rw_p50'], o[f'b{bi}_Rw_p95'] = np.percentile(r, [5, 50, 95])
                o[f'b{bi}_corr_Rw_T'] = (w*(r - mr)*(tn - mt)).sum()/W/np.sqrt((w*(r - mr)**2).sum()/W*(w*(tn - mt)**2).sum()/W)
        prev = (t, g.copy(), A('tr_divb'), Tn)
        out.append(o)
    return out


def main():
    out, runs = sys.argv[1], sys.argv[2].split(',')
    rhoc = float(sys.argv[sys.argv.index('--rhoc') + 1]) if '--rhoc' in sys.argv else 3.24e-4
    rows = []
    for d in runs:
        arm = os.path.basename(d.rstrip('/'))
        st, steps = trd_lines(d)
        if steps:
            S = {k: np.array([s[k] for s in steps]) for k in steps[0]}
            with open(out.replace('.csv', f'_{arm}_steps.csv'), 'w') as fo:
                ks = list(steps[0].keys()); fo.write(','.join(ks) + '\n')
                for s in steps:
                    fo.write(','.join(f'{s[k]:.8e}' for k in ks) + '\n')
            for lo, hi in ((10, 30), (30, 60)):
                m = (S['t'] >= lo) & (S['t'] < hi)
                if m.any():
                    rows.append({'arm': arm, 'kind': f'TRD steps t in [{lo},{hi})', 'n': int(m.sum()),
                                 **{f'{k}': float(np.mean(S[k][m])) for k in S if k.startswith('b')}})
        for o in dumps(d, rhoc):
            rows.append({'arm': arm, 'kind': f'dump t={o["t"]:g}', 'n': 1, **o})
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out, 'w') as fo:
        fo.write(','.join(keys) + '\n')
        for r in rows:
            fo.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
    for r in rows:
        if r['kind'].startswith('TRD'):
            print(f"{r['arm']:5s} {r['kind']:24s} n={r['n']:5d} | R b0 {r['b0_R']:+.3e} b1 {r['b1_R']:+.3e} b2 {r['b2_R']:+.3e} "
                  f"| stages b1 {r['b1_R_s1']:+.2e} {r['b1_R_s2']:+.2e} {r['b1_R_s3']:+.2e} | 6dt b1 {r['b1_6dt']:+.3e} divb {r['b1_divb']:+.3e} "
                  f"| |T| b1 {r['b1_T']:.3e} | sig_tt b2 {r['b2_sig_tt']:+.3e} sig_r {r['b2_sig_r']:+.3e}", flush=True)
        else:
            print(f"{r['arm']:5s} {r['kind']:24s} R b1 mean {r['b1_R_mean']:+.3e} rms {r['b1_R_rms']:.3e} p05/50/95 {r['b1_R_p05']:+.2e}/"
                  f"{r['b1_R_p50']:+.2e}/{r['b1_R_p95']:+.2e} | |T| {r['b1_T_mean']:.3e} | corr(R,|T|) {r['b1_corr_R_T']:+.2f} "
                  f"| |divT| dx/|T| {r['b1_divT_over_T_dx']:.2e} | trace {r['b1_trace_over_T']:.1e}" +
                  (f" | Rw b1 mean {r['b1_Rw_mean']:+.3e} rms {r['b1_Rw_rms']:.3e} corr(Rw,|T|) {r['b1_corr_Rw_T']:+.2f}" if 'b1_Rw_mean' in r else ''),
                  flush=True)


if __name__ == '__main__':
    main()
