#!/usr/bin/env python3
"""c160 (R-177 item 3 + addendum 3): the P18W per-shell analysis (c155_shell/P18W, P18_r82 t_p + 800 -> 850, wip/shell-counters
924a7ca1, SHELL lines about every 1 M; mass_closure.cpp's header lists every item).
A. The budget. Per shell (0.01 M* in m about the remnant's centre, re-partitioned at every SHELL output), over the window:
   - the fixed-m change of the tracer content, MY0(last partition)/MY0(first) - 1 (each partition puts 0.01 M* in every shell,
     so this is dY/Y at fixed m; dK/K = Gamma dY/Y, Y = (K/kmax)^(1/Gamma));
   - the channels summed over the intervals between those two partitions: the C2P's (flagged-dense resync up / down, non-dense
     resync, cold recovery, ...), the gap between cycles (the regrid C2P; c155 gate: the shells' gap = YCLOSE floor_init), and
     transport + re-partition = the rest (the advected DY and the edges' motion; the follow / excess split of transport is not
     used: its excess is relative to donor-cell Y and does not survive the re-partition);
   - the flag events' sensor-ratio and theta histograms (counts, RK-weighted dDY).
   The rho_sw crossing counts are not used: sh_den_ is reset only when its extent changes (mass_closure.cpp:1034-1037), so after
   a regrid that keeps nmb it compares other MeshBlocks' cells.
B. The addendum-3 proxy (the counters lack it): the dediag slices every 2 M (de_branch, de_ken_ktr = K_en/K_tr - 1 for every
   dense, non-bad cell, de_sensor). Per slice cell inside the remnant (r about the nearest partition's centre < its m = 0.2
   edge; shell from its edges): K_en/K_tr - 1 by branch -- 4 (energy branch: resynced, K_en > K_tr (1 - 1e-3)), 2 (the guard
   declined, K_en <= K_tr (1 - 1e-3)), 1 (unflagged entropy branch: the energy solution's noise, applied nowhere). Snapshots of
   the flagged cells' guard decisions, not every release event; the xy plane (z = 0) only (the xz plane misses the remnant).
    c160_p18w_shell.py RUNDIR OUTPREFIX"""
import glob, re, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402

GAM = 5.0/3.0
B = 'cfc_tde_wd_imbh_parabolic_tracker'
CH = ['flag_up', 'flag_dn', 'resync_nondense', 'crec', 'floor', 'zone', 'exc', 'dense_chk', 'upd_nc', 'rk', 'gap']
HB = ['<0.25', '0.25-0.5', '0.5-1', '1-2', '2-5', '>=5']
TB = ['<0.01', '0.01-0.03', '0.03-0.1', '0.1-0.3', '0.3-1', '>=1']
KB = np.array([-1.0, -0.1, -0.03, -0.01, -0.003, -0.001, 0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 1e9])


def parse(log):
    sh, ed = [], []
    for l in open(log):
        if l.startswith('SHELL ') or l.startswith('SHELL_EDGES'):
            d = {}
            for k, v in re.findall(r'(\w+)=(\S+)', l):
                d[k] = np.array([float(x) for x in v.split(',')]) if ',' in v else float(v)
            (sh if l.startswith('SHELL ') else ed).append(d)
    return sh, ed


def budget(sh, out):
    t0 = sorted(set(r['t0'] for r in sh))
    ns = 1 + max(int(r['shell']) for r in sh)
    lines = [f'# P18W window {t0[0]} -> {t0[-1]} ({t0[-1] - t0[0]:.2f} M), {len(t0)} partitions; fractions of MY0(first partition)',
             'shell,m0,m1,MY0,dY_fixm,dK_fixm,transport_repart,' + ','.join(CH) + ',resync_net,n_flag,n_guard,' +
             ','.join(f'flag_ratio_n{b}' for b in HB) + ',' + ','.join(f'flag_theta_n{b}' for b in TB) + ',' +
             ','.join(f'flag_theta_dY{b}' for b in TB)]
    for s in range(ns):
        rs = [r for r in sh if int(r['shell']) == s]
        if len(rs) != len(t0) or rs[0]['MY0'] <= 0.0:
            continue
        m = rs[0]['MY0']
        fix = rs[-1]['MY0']/m - 1.0
        body = rs[:-1]   # the intervals between the first and the last partition
        c = {k: sum(r[k] for r in body)/m for k in CH}
        tr = fix - sum(c.values())
        hr = sum(r['ratio_n'] for r in body); ht = sum(r['theta_n'] for r in body); hy = sum(r['theta_dY'] for r in body)/m
        mm = rs[0]['m']
        lines.append(f"{s},{mm[0]},{mm[1]},{m:.6e},{fix:.4e},{GAM*fix:.4e},{tr:.4e}," + ','.join(f'{c[k]:.4e}' for k in CH) +
                     f",{c['flag_up'] + c['flag_dn']:.4e},{sum(r['n_flag'] for r in body):.0f},{sum(r['n_guard'] for r in body):.0f}," +
                     ','.join(f'{x:.0f}' for x in hr) + ',' + ','.join(f'{x:.0f}' for x in ht) + ',' + ','.join(f'{x:.3e}' for x in hy))
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def slices(d, ed, out):
    te = np.array([e['t'] for e in ed])
    fx = sorted(glob.glob(f'{d}/bin/{B}.dediag_xy.*.bin'))
    fp = sorted(glob.glob(f'{d}/bin/{B}.prim_xy.*.bin'))
    rows = []
    H = {b: np.zeros((len(KB) - 1, 2)) for b in (1, 2, 4)}   # (count, mass) per K_en/K_tr - 1 bin
    for f, g in zip(fx, fp):
        S, P = bc.read_binary(f), bc.read_binary(g)
        t = float(S['time'])
        e = ed[int(np.argmin(np.abs(te - t)))]
        gs = np.asarray(S['mb_geometry'], dtype=float); n = S['nx1_out_mb']
        cat = lambda Q, k: np.concatenate([np.asarray(Q['mb_data'][k][b]).ravel() for b in range(len(gs))]).astype(float)
        br, kk, sn, rho = cat(S, 'de_branch'), cat(S, 'de_ken_ktr'), cat(S, 'de_sensor'), cat(P, 'dens')
        xs, ys, da = [], [], []
        for b in range(len(gs)):
            h = (gs[b, 1] - gs[b, 0])/n; c = (np.arange(n) + 0.5)*h
            yy, xx = np.meshgrid(gs[b, 2] + c, gs[b, 0] + c, indexing='ij')
            xs.append(xx.ravel()); ys.append(yy.ravel()); da.append(np.full(xx.size, h*h))
        xs, ys, da = np.concatenate(xs), np.concatenate(ys), np.concatenate(da)
        xc = e['xc']; re_ = e['r_edges']
        r = np.sqrt((xs - xc[0])**2 + (ys - xc[1])**2 + xc[2]**2)
        inr = r < re_[-1]
        shell = np.searchsorted(re_, r, side='right') - 1
        w = rho*da
        row = {'t': t, 'dt_partition': float(e['t'] - t), 'z_offset': float(xc[2]), 'n_rem': int(inr.sum())}
        for b in (1, 2, 4):
            m = inr & (np.abs(br - b) < 0.5)
            row[f'n_br{b}'] = int(m.sum())
            row[f'mass_br{b}'] = float(w[m].sum()/w[inr].sum()) if inr.any() else np.nan
            if m.any():
                q = kk[m]
                row[f'ken_mean_br{b}'] = float((w[m]*q).sum()/w[m].sum())
                row[f'ken_med_br{b}'] = float(np.median(q))
                row[f'ken_p05_br{b}'], row[f'ken_p95_br{b}'] = (float(x) for x in np.percentile(q, [5, 95]))
                row[f'sensor_med_br{b}'] = float(np.median(sn[m]))
                row[f'shell_med_br{b}'] = float(np.median(shell[m]))
                i = np.clip(np.searchsorted(KB, q, side='right') - 1, 0, len(KB) - 2)
                np.add.at(H[b][:, 0], i, 1.0); np.add.at(H[b][:, 1], i, w[m])
            else:
                for k in ('ken_mean', 'ken_med', 'ken_p05', 'ken_p95', 'sensor_med', 'shell_med'):
                    row[f'{k}_br{b}'] = np.nan
        rows.append(row)
        print(f"t={t:.1f} rem cells {row['n_rem']} | br4 {row['n_br4']} ({100*row['mass_br4']:.2f}% m) ken med "
              f"{row['ken_med_br4']:+.2e} | br2 {row['n_br2']} ken med {row['ken_med_br2']:+.2e} | br1 {row['n_br1']} ken p05/med/p95 "
              f"{row['ken_p05_br1']:+.1e}/{row['ken_med_br1']:+.1e}/{row['ken_p95_br1']:+.1e}", flush=True)
    keys = list(rows[0].keys())
    with open(out + '_slices.csv', 'w') as fo:
        fo.write(','.join(keys) + '\n')
        for r in rows:
            fo.write(','.join(str(r[k]) for k in keys) + '\n')
    with open(out + '_kenhist.csv', 'w') as fo:
        fo.write('# K_en/K_tr - 1 over the remnant cells of all xy slices, by branch: count, rho dA (mass share in the branch)\n')
        fo.write('lo,hi,' + ','.join(f'n_br{b},mfrac_br{b}' for b in (1, 2, 4)) + '\n')
        for i in range(len(KB) - 1):
            fo.write(f'{KB[i]},{KB[i + 1]},' + ','.join(f'{H[b][i, 0]:.0f},{H[b][i, 1]/max(H[b][:, 1].sum(), 1e-300):.4e}'
                                                        for b in (1, 2, 4)) + '\n')
    print(open(out + '_kenhist.csv').read())


def main():
    d, out = sys.argv[1], sys.argv[2]
    sh, ed = parse(f'{d}/out.log')
    budget(sh, out + '_budget.csv')
    slices(d, ed, out)


if __name__ == '__main__':
    main()
