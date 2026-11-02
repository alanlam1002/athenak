#!/usr/bin/env python3
"""c167 (R-204 item 2): P18W (plain rms 1e-8, c155_shell/P18W) vs P18W2 (rms 1e-8 + mg_min_vcycles 1, c167_p18w2/P18W2), both
P18_r82 from t_p + 800 = 1291.1, 96 ranks, DE on; SHELL lines (mass_closure.cpp shell_counters) cut at the last partition both
runs share (P18W2's final partial interval was lost in its end-of-run hang). Reports:
  M_Hill(t) = M_rH/M* (SHELL_EDGES) and its least-squares loss rate;
  K_Hill = <Y>^Gamma inside r_H (Y = sum MY / sum MD over shells 0-20 at each interval's end; K = Y^Gamma for dual_energy_kmax 1;
    the K of the mean tracer, not a mass-weighted K);
  K(m): per shell (MY1/MD)^Gamma at the first and the last common interval; the fixed-m change MY0(last partition)/MY0(first) - 1
    (C-159's measure; x Gamma for K) and its channels (C2P items, the gap between cycles; transport + re-partition = the rest),
    fractions of MY0(first);
  the CFC cost (PHASE_TIMERS mg / cycles) of both runs and P18W2's V-cycle counts (MGV lines; P18W has none);
  P18W2's SHELL_REL distributions (applied / declined / release) by shell group.
    c167_p18w_compare.py P18W_DIR P18W2_DIR OUTPREFIX"""
import re, sys
import numpy as np
GAM = 5.0/3.0
CH = ['flag_up', 'flag_dn', 'resync_nondense', 'crec', 'floor', 'zone', 'exc', 'dense_chk', 'upd_nc', 'rk', 'gap']
KB = ['<-0.1', '-0.1..-0.03', '-0.03..-0.01', '-0.01..-0.003', '-0.003..-0.001', '-0.001..0', '0..0.001', '0.001..0.003',
      '0.003..0.01', '0.01..0.03', '0.03..0.1', '>=0.1']
TB = ['<0.01', '0.01-0.03', '0.03-0.1', '0.1-0.3', '0.3-1', '>=1']
GROUPS = (('core m<0.04', range(0, 4)), ('m 0.04-0.17', range(4, 17)), ('m 0.17-0.20', range(17, 20)),
          ('beyond 0.2 in r_H', [20]), ('outside r_H', [21]))


def kv(l):
    d = {}
    for k, v in re.findall(r'(\w+)=(\S+)', l):
        try:
            d[k] = np.array([float(x) for x in v.split(',')]) if ',' in v else float(v)
        except ValueError:
            d[k] = v
    return d


def parse(d):
    sh, ed, rel, mgv, pt = [], [], [], [], []
    for l in open(f'{d}/out.log'):
        if l.startswith('SHELL '):
            sh.append(kv(l))
        elif l.startswith('SHELL_EDGES'):
            e = kv(l); e['M_rH/M*'] = float(re.search(r'M_rH/M\*=(\S+)', l).group(1)); ed.append(e)
        elif l.startswith('SHELL_REL'):
            rel.append(kv(l))
        elif l.startswith('MGV '):
            mgv.append(l)
        elif l.startswith('PHASE_TIMERS '):
            pt.append(kv(l))
    return sh, ed, rel, mgv, pt


def intervals(sh):
    out = {}
    for d in sh:
        out.setdefault(round(d['t0'], 3), {})[int(d['shell'])] = d
    return [out[k] for k in sorted(out)]   # partitions in order (output times drift with dt between runs)


def main():
    A, B, pre = sys.argv[1], sys.argv[2], sys.argv[3]
    ra, rb = parse(A), parse(B)
    ia, ib = intervals(ra[0]), intervals(rb[0])
    npair = min(len(ia), len(ib))   # pair the partitions by order
    dts = [abs(ia[q][0]['t0'] - ib[q][0]['t0']) for q in range(npair)]
    common = list(range(npair))
    t_first, t_last = 0, npair - 1
    o = open(pre + '.txt', 'w')
    def p(s=''):
        print(s); o.write(s + '\n')
    p(f'paired partitions {npair} (P18W {len(ia)}, P18W2 {len(ib)}): t0 {ia[0][0]["t0"]:.3f} .. P18W {ia[-1 + npair][0]["t0"]:.3f} '
      f'/ P18W2 {ib[-1 + npair][0]["t0"]:.3f}; max |t0 offset| {max(dts):.3f} M')
    # M_Hill
    for lab, r, iv in (('P18W', ra, ia), ('P18W2', rb, ib)):
        ta, tb = iv[0][0]['t0'], iv[npair - 1][0]['t0']
        e = [x for x in r[1] if ta - 1e-3 <= x['t'] <= tb + 1e-3]
        t = np.array([x['t'] for x in e]); m = np.array([x['M_rH/M*'] for x in e])
        sl = np.polyfit(t, m, 1)[0]
        p(f'{lab}: M_Hill/M* {m[0]:.6f} -> {m[-1]:.6f} over t {t[0]:.2f}..{t[-1]:.2f} ({len(t)} partitions); lsq rate {sl:+.4e} M*/M '
          f'({sl/m.mean():+.4e} /M relative); r_H {e[0]["r_H"]:.4f} -> {e[-1]["r_H"]:.4f}')
    # K_Hill and K(m)
    def kprof(iv, t0, key='MY1'):
        y = np.array([iv[t0][b][key]/iv[t0][b]['MD'] if iv[t0][b]['MD'] > 0 else np.nan for b in range(22)])
        yh = sum(iv[t0][b][key] for b in range(21))/sum(iv[t0][b]['MD'] for b in range(21))
        return y**GAM, yh**GAM
    p('\nK_Hill = <Y>^Gamma inside r_H at the interval ends (first common, last common):')
    for lab, iv in (('P18W', ia), ('P18W2', ib)):
        k0, kh0 = kprof(iv, t_first); k1, kh1 = kprof(iv, t_last)
        p(f'  {lab}: {kh0:.6e} -> {kh1:.6e} ({kh1/kh0 - 1:+.4%})')
    p('\nper shell: K(m) at the first / last common interval end, P18W | P18W2; fixed-m dY (MY0 last partition / first - 1) and '
      'channels as fractions of MY0(first)')
    hdr = 'shell m      K0_W       K1_W       K0_W2      K1_W2      dY_W       dY_W2      gap_W     gap_W2    resync_W  resync_W2 transp_W  transp_W2'
    p(hdr)
    rows = []
    for b in range(22):
        r = [b]
        for iv in (ia, ib):
            r += [kprof(iv, t_first)[0][b], kprof(iv, t_last)[0][b]]
        for iv in (ia, ib):
            my0 = iv[t_first][b]['MY0']
            if my0 == 0.0:   # an empty shell at the window's start (m edges beyond the gas)
                r += [np.nan]*4; continue
            dy = iv[t_last][b]['MY0']/my0 - 1
            ch = {c: sum(iv[t][b][c] for t in common[:-1])/my0 for c in CH}
            res = ch['flag_up'] + ch['flag_dn']
            tr = dy - sum(ch.values())
            r += [dy, ch['gap'], res, tr]
        rows.append(r)
        m = f'{0.01*b:.2f}' if b < 20 else ('>0.2' if b == 20 else 'out')
        p(f'{b:3d} {m:5s} {r[1]:.4e} {r[2]:.4e} {r[3]:.4e} {r[4]:.4e} {r[5]:+.3e} {r[9]:+.3e} {r[6]:+.2e} {r[10]:+.2e} '
          f'{r[7]:+.2e} {r[11]:+.2e} {r[8]:+.2e} {r[12]:+.2e}')
    np.savetxt(pre + '_shells.csv', np.array(rows), delimiter=',', header='shell,K0_W,K1_W,K0_W2,K1_W2,dY_W,gap_W,resync_W,transp_W,'
               'dY_W2,gap_W2,resync_W2,transp_W2', fmt='%.6e')
    sel = range(4, 17)
    for lab, i0 in (('P18W', 5), ('P18W2', 9)):
        dy = np.array([rows[b][i0] for b in sel]); gp = np.array([rows[b][i0 + 1] for b in sel])
        rs = np.array([rows[b][i0 + 2] for b in sel]); tr = np.array([rows[b][i0 + 3] for b in sel])
        p(f'{lab} m 0.04-0.17: fixed-m dY {dy.min():+.3%}..{dy.max():+.3%} (sum {dy.sum():+.3%}); shares of the summed rise: '
          f'gap {gp.sum()/dy.sum():.0%}, resync {rs.sum()/dy.sum():.0%}, transport+repartition {tr.sum()/dy.sum():.0%}')
    # cost and V-cycles
    p('\nCFC cost (PHASE_TIMERS lines with cycles >= 50: mg s per cycle; stagen s per cycle):')
    for lab, r in (('P18W', ra), ('P18W2', rb)):
        q = [d for d in r[4] if d.get('cycles', 0) >= 50]
        mg = sum(d['mg'] for d in q)/sum(d['cycles'] for d in q); st = sum(d['stagen'] for d in q)/sum(d['cycles'] for d in q)
        p(f'  {lab}: mg {mg:.3f} s/cycle, stagen {st:.3f} s/cycle ({sum(d["cycles"] for d in q):.0f} cycles)')
    rows = []
    for l in rb[3]:
        v = dict((k, (int(n), float(a), float(b))) for k, n, a, b in re.findall(r' (\w+)= (-?\d+),(\S+?),(\S+?)(?= |$)', l.strip()))
        rows.append(v)
    p(f'P18W2 MGV lines {len(rows)} (cycle 10085 = the restart\'s first solve included):')
    for s in ('X', 'psi', 'ap', 'beta'):
        n = np.array([r[s][0] for r in rows if s in r and r[s][0] >= 0])
        d0 = np.array([r[s][1] for r in rows if s in r and r[s][0] >= 0])
        h = np.bincount(n, minlength=6)
        p(f'  {s:4s}: V-cycles per solve mean {n.mean():.2f}, distribution n=1..5,>5: {h[1:6].tolist()} {int((n > 5).sum())}; '
          f'initial defect (warm start) above 1e-8 in {(d0 > 1e-8).mean():.1%} of solves (median {np.median(d0):.2e})')
    # SHELL_REL
    p('\nP18W2 SHELL_REL (K_en/K_tr - 1 per C2P event; summed over the whole window):')
    rel = rb[2]
    with open(pre + '_rel.csv', 'w') as f:
        f.write('group,class,n,mean_dk,frac_pos,' + ','.join('n' + k for k in KB) + ',' + ','.join('th' + t for t in TB) + '\n')
        for g, bs in GROUPS:
            for c in ('applied', 'declined', 'release'):
                kn = np.zeros(12); ks = np.zeros(12); kt = np.zeros((12, 6))
                for d in rel:
                    if int(d['shell']) in bs and d['class'] == c:
                        kn += d['k_n']; ks += d['k_sum']; kt += np.asarray(d['k_theta_n']).reshape(12, 6)
                n = kn.sum()
                if n == 0:
                    p(f'  {g:18s} {c:8s} n=0'); continue
                pos = kn[6:].sum()/n
                th = kt.sum(axis=0)
                p(f'  {g:18s} {c:8s} n={n:9.0f} mean {ks.sum()/n:+.3e} frac(>=0) {pos:.3f} | K bins '
                  + ' '.join(f'{x/n:.3f}' for x in kn) + ' | theta ' + ' '.join(f'{x/n:.3f}' for x in th))
                f.write(f'{g},{c},{n:.0f},{ks.sum()/n:.6e},{pos:.6f},' + ','.join(f'{x:.0f}' for x in kn) + ',' +
                        ','.join(f'{x:.0f}' for x in th) + '\n')
    o.close()


if __name__ == '__main__':
    main()
