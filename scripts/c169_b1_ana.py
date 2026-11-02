#!/usr/bin/env python3
"""c169 (R-202 item 1; R-204 item 4): the B.1 readout at converged solves (c161_b1 VB10 / Be10 / Bc10, t = 41.78 -> 300,
t_p = 135.9; rms 1e-8 + mg_min_vcycles 1; trace_diag with the split; binary caf6487e).
  dumps   c169_b1_ana.py dumps OUT.csv LABEL=RUNDIR ...   per 3D dump (prim_3d, prim_3d_peri):
     the dense bands (rho >= 0.01, 0.1 rho_c): mass, rho dV-weighted K/K0 - 1, rho_max/rho_c;
     the bound remnant ~ the Hill sphere about the rho-centroid x_c of rho >= 0.01 rho_c: r_H = |x_c - x_BH| (M_rem/(3 M_BH))^(1/3),
       M_rem = the mass inside r_H (3 fixed-point iterations from M(rho >= 0.01 rho_c)); its mass, K/K0 - 1, floored mass share;
     the stream: 1e-6 <= rho/rho_c < 0.01 outside r_H: mass, rho dV-weighted K/K0 - 1, floored mass share (T <= 1.0001 tfloor),
       KE/e_int = sum 1/2 rho |v - v_c|^2 dV / sum P/(Gamma - 1) dV (v = the dump's velocity, v_c = the dense mass's mean;
       a Newtonian coordinate measure of how cold the stream is, not a covariant energy);
     x_BH = puncture_x - frame_disp (user history; c160_coldmax.xbh).
  logs    c169_b1_ana.py logs OUT.txt LABEL=RUNDIR ...    from out.log:
     TRD (16 sums per band, wip/trace-split): per 6 M window the stage means of R = sum wR / sum w per band, and the deposit
       split (sqrt(g) dV sums): trace part (R S/3), trace-free total (-1/2 psi^4 S^ij T_ij) = T_self + T_cross + B parts;
     TRR (10 radial bins about x_BH, rho >= 0.01 rho_c): per window the w-weighted |T_self|, |T_cross|, |B| and the deposits;
     MGV: per window and solver, the V-cycles per solve (distribution 1, 2, 3, 4-5, > 5) and the share of solves whose warm
       start already met 1e-8 (R-204 item 4);
     cost: wall per cycle (PHASE_TIMERS)."""
import glob, re, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh, par   # noqa: E402
GAM = 5.0/3.0
TP = 135.9


def dumps(out, specs):
    rows = []
    for s in specs:
        lab, d = s.split('=', 1)
        rhoc, k0, tfl, mbh = par(d, 'rhoc'), par(d, 'kappa'), par(d, 'tfloor'), par(d, 'puncture_mass', 1.0)
        fs = sorted(glob.glob(f'{d}/bin/*.prim_3d.0*.bin') + glob.glob(f'{d}/bin/*.prim_3d_peri.0*.bin'),
                    key=lambda f: bc.read_binary(f)['time'])
        for f in fs:
            P = bc.read_binary(f); g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']; q = P['mb_data']; t = P['time']
            cat = lambda k: np.concatenate([np.asarray(q[k][b]).ravel() for b in range(len(g))]).astype(float)
            rho, p, T = cat('dens'), cat('press'), cat('temperature')
            v = np.stack([cat('velx'), cat('vely'), cat('velz')])
            h = (g[:, 1] - g[:, 0])/n
            dv = np.repeat(h**3, n**3)
            ax = lambda lo: (lo[:, None] + (np.arange(n) + 0.5)[None, :]*h[:, None])
            X, Y, Z = ax(g[:, 0]), ax(g[:, 2]), ax(g[:, 4])
            x = np.repeat(X[:, None, None, :], n, 1).repeat(n, 2).ravel() if False else \
                np.broadcast_to(X[:, None, None, :], (len(g), n, n, n)).ravel()
            y = np.broadcast_to(Y[:, None, :, None], (len(g), n, n, n)).ravel()
            z = np.broadcast_to(Z[:, :, None, None], (len(g), n, n, n)).ravel()
            K = p/rho**GAM/k0 - 1.0; fl = T <= 1.0001*tfl; m = rho*dv
            d01, d1 = rho >= 0.01*rhoc, rho >= 0.1*rhoc
            xc = np.array([(m[d01]*c[d01]).sum()/m[d01].sum() for c in (x, y, z)])
            vc = np.array([(m[d01]*c[d01]).sum()/m[d01].sum() for c in v])
            xb = xbh(d, t); D = np.linalg.norm(xc - xb)
            rr = np.sqrt((x - xc[0])**2 + (y - xc[1])**2 + (z - xc[2])**2)
            mrem = m[d01].sum()
            for _ in range(3):
                rh = D*(mrem/(3.0*mbh))**(1.0/3.0); inh = rr < rh; mrem = m[inh].sum()
            st = (~inh) & (rho >= 1e-6*rhoc) & (rho < 0.01*rhoc)
            ke = 0.5*rho*((v - vc[:, None])**2).sum(axis=0)*dv; ei = p/(GAM - 1.0)*dv
            wm = lambda sel, a: (m[sel]*a[sel]).sum()/m[sel].sum() if m[sel].sum() > 0 else np.nan
            r = dict(arm=lab, t=t, d01_M=m[d01].sum(), d01_K=wm(d01, K), d1_M=m[d1].sum(), d1_K=wm(d1, K), rhomax=rho.max()/rhoc,
                     D_bh=D, r_H=rh, rem_M=mrem, rem_K=wm(inh, K), rem_fl=m[inh & fl].sum()/mrem,
                     st_M=m[st].sum(), st_K=wm(st, K), st_fl=m[st & fl].sum()/max(m[st].sum(), 1e-300),
                     st_KE_eint=ke[st].sum()/max(ei[st].sum(), 1e-300), all_M=m.sum())
            rows.append(r)
            print(f"{lab} t={t:7.2f} D={D:7.2f} r_H={rh:.3f} M01={r['d01_M']:.4e} K01={r['d01_K']:+.4f} K1={r['d1_K']:+.4f} "
                  f"rho_max={r['rhomax']:.3f} | rem M={mrem:.4e} K={r['rem_K']:+.4f} fl={r['rem_fl']:.4f} | stream M={r['st_M']:.4e} "
                  f"K={r['st_K']:+.4f} fl={r['st_fl']:.4f} KE/eint={r['st_KE_eint']:.3e}", flush=True)
    keys = list(rows[0].keys())
    with open(out, 'w') as fo:
        fo.write(','.join(keys) + '\n')
        for r in rows:
            fo.write(','.join(str(r[k]) for k in keys) + '\n')


def logs(out, specs, win=6.0):
    o = open(out, 'w')
    def p(s=''):
        print(s); o.write(s + '\n')
    for s in specs:
        lab, d = s.split('=', 1)
        trd, trr, mgv, pt = [], [], [], []
        for l in open(f'{d}/out.log'):
            if l.startswith('TRD '):
                h = dict(re.findall(r'(\w+)=(\S+)', l.split(' b0=')[0]))
                b = [np.array([float(x) for x in bb.split(',')]) for bb in re.findall(r' b\d=(\S+)', l)]
                trd.append((float(h['tstage']), int(h['valid']), b))
            elif l.startswith('TRR '):
                tm = float(re.search(r' time=(\S+)', l).group(1))
                rb = [np.array([float(x) for x in bb.split(',')]) for bb in re.findall(r' r\d=(\S+)', l)]
                trr.append((tm, rb))
            elif l.startswith('MGV '):
                tm = float(re.search(r'time= (\S+)', l).group(1))
                v = dict((k, (int(a), float(b0), float(b1))) for k, a, b0, b1 in re.findall(r' (\w+)= (-?\d+),(\S+?),(\S+?)(?= |$)', l.strip()))
                mgv.append((tm, v))
            elif l.startswith('PHASE_TIMERS '):
                pt.append(dict((k, float(x)) for k, x in re.findall(r'(\w+)=([-+.\deE]+)', l)))
        p(f'===== {lab} ({d}): TRD stages {len(trd)}, TRR {len(trr)}, MGV {len(mgv)}')
        q = [x for x in pt if x.get('cycles', 0) >= 50]
        if q:
            p(f'cost: wall {sum(x["wall"] for x in q)/sum(x["cycles"] for x in q):.3f} s/cycle, mg {sum(x["mg"] for x in q)/sum(x["cycles"] for x in q):.3f} s/cycle '
              f'({sum(x["cycles"] for x in q):.0f} cycles)')
        t0, t1 = trd[0][0], trd[-1][0]
        edges = np.arange(np.floor(t0/win)*win, t1 + win, win)
        p('TRD per window (valid stages): t | R b0 b1 b2 (M^-1) | deposit sqrt(g)dV-sums, band b2 (rho >= 0.01 rho_c): trace(RS/3) '
          'TF_total  TF_self  TF_cross  TF_B | |T_self| |T_cross| |B| (w-mean, b2)')
        for lo, hi in zip(edges[:-1], edges[1:]):
            S = [x for x in trd if lo <= x[0] < hi and x[1] == 1]
            if not S:
                continue
            R = [np.mean([x[2][b][1]/x[2][b][0] for x in S if x[2][b][0] > 0]) for b in range(3)]
            bb = np.mean([x[2][2] for x in S], axis=0)
            p(f'{lo:6.0f}-{hi:4.0f} | {R[0]:+.2e} {R[1]:+.2e} {R[2]:+.2e} | {bb[9]:+.3e} {bb[8]:+.3e} {bb[13]:+.3e} {bb[14]:+.3e} '
              f'{bb[15]:+.3e} | {bb[10]/bb[0]:.2e} {bb[11]/bb[0]:.2e} {bb[12]/bb[0]:.2e}')
        if trr:
            rmin, rmax = 2.0, 200.0
            e = np.exp(np.linspace(np.log(rmin), np.log(rmax), 9))
            lab_b = ['<2'] + [f'{e[i]:.1f}-{e[i+1]:.1f}' for i in range(8)] + ['>200']
            p('TRR per window: per radial bin about x_BH (rho >= 0.01 rho_c): mass-weight w | |B|/|T_self| | dep_B/(dep_self+dep_cross)')
            for lo, hi in zip(edges[:-1], edges[1:]):
                S = [x for x in trr if lo <= x[0] < hi]
                if not S:
                    continue
                A = np.mean([np.array(x[1]) for x in S], axis=0)
                cells = []
                for i in range(10):
                    w = A[i][0]
                    if w <= 0:
                        continue
                    ts, tb = A[i][2]/w, A[i][4]/w
                    cells.append(f'{lab_b[i]}: w {w:.2e} B/Ts {tb/max(ts, 1e-300):.2e} depB/(s+c) {A[i][7]/(A[i][5] + A[i][6]) if (A[i][5] + A[i][6]) != 0 else np.nan:+.2e}')
                p(f'{lo:6.0f}-{hi:4.0f} | ' + ' ; '.join(cells))
        p('MGV per window: solver: mean V-cycles, shares n=1/2/3/4-5/>5, share whose warm start met 1e-8 (def0 <= 1e-8)')
        for lo, hi in zip(edges[:-1], edges[1:]):
            S = [x[1] for x in mgv if lo <= x[0] < hi]
            if not S:
                continue
            seg = []
            for sv in ('X', 'psi', 'ap', 'beta'):
                nn = np.array([x[sv][0] for x in S if sv in x and x[sv][0] >= 0]); d0 = np.array([x[sv][1] for x in S if sv in x and x[sv][0] >= 0])
                hb = [(nn == 1).mean(), (nn == 2).mean(), (nn == 3).mean(), ((nn >= 4) & (nn <= 5)).mean(), (nn > 5).mean()]
                seg.append(f'{sv} {nn.mean():.2f} [' + '/'.join(f'{x:.2f}' for x in hb) + f'] met {(d0 <= 1e-8).mean():.2f}')
            p(f'{lo:6.0f}-{hi:4.0f} | ' + ' | '.join(seg))
    o.close()


if __name__ == '__main__':
    {'dumps': dumps, 'logs': logs}[sys.argv[1]](sys.argv[2], sys.argv[3:])
