#!/usr/bin/env python3
"""C-086: compact tables from c086_rst.txt / c086_rst_samples.npz (written by c086_rst.py)."""
import re
import numpy as np
P = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/plots/phase2_m1/'
txt = open(P + 'c086_rst.txt').read().splitlines()
L = ['C-086 compact tables (scripts/c086_summary.py from c086_rst.txt, c086_rst_samples.npz)', '',
     'Item 8 (3D replica): flagged fraction of below-switch cells, f_m (f_n), per D/rho_c bin; r_BH >= 3',
     'cold = same with c_s from P = K0 rho^Gamma (all bins pooled, mass); val = z=0 agreement with stored de_shock',
     f'{"run":3s} {"t":>6s} | {"1e-9..1e-8":>15s} {"1e-8..1e-7":>15s} {"1e-7..1e-6":>15s} {"1e-6..1e-5":>15s} {"1e-5..sw":>15s} | all f_m  cold f_m | val']
i = 0
while i < len(txt):
    m = re.match(r'(R\d) t +([\d.]+) .*agree ([\d.]+)', txt[i])
    if m:
        cells, nm, mm, mf, mf0 = [], 0, 0, 0, 0
        for k in range(5):
            s = txt[i + 1 + k]
            n = float(re.search(r'N +(\d+)', s).group(1)); M = float(re.search(r'M/M\* ([\d.e+-]+)', s).group(1))
            fn = float(re.search(r'f_n ([\d.]+)', s).group(1)); fm = float(re.search(r' f_m ([\d.]+)', s).group(1))
            f0 = float(re.search(r'f0_m ([\d.]+)', s).group(1))
            cells.append(f'{fm:.3f} ({fn:.3f})'); mm += M; mf += fm * M; mf0 += f0 * M
        L.append(f'{m.group(1):3s} {float(m.group(2)):6.1f} | ' + ' '.join(f'{c:>15s}' for c in cells)
                 + f' | {mf / mm:.3f}    {mf0 / mm:.3f}   | {float(m.group(3)):.3f}')
        i += 7
    else:
        i += 1
L += ['', 'Item 9 (3D): mass fraction with D < rho_sw | D < 1e-9 rho_c, of: bound (E<0) | bound debris (bound & ~remnant)',
      f'{"run":3s} {"t":>6s} | {"M_bound/M*":>10s} {"<sw":>9s} {"<1e-9rc":>9s} | {"M_deb/M*":>8s} {"<sw":>9s} {"<1e-9rc":>9s}']
for s in txt:
    m = re.match(r'  (R\d) t +([\d.]+): bound ([\d.]+) M\*: <sw (\S+) <1e-9rc (\S+) .*; debris ([\d.]+) M\*: <sw (\S+) <1e-9rc (\S+)', s)
    if m:
        g = m.groups()
        deb = float(g[5])
        d1, d2 = (f'{float(g[6]):9.2e}', f'{float(g[7]):9.2e}') if deb > 1e-3 else ('      n/a', '      n/a')
        L.append(f'{g[0]:3s} {float(g[1]):6.1f} | {float(g[2]):10.4f} {float(g[3]):9.2e} {float(g[4]):9.2e} | {deb:8.4f} {d1} {d2}')
import sys
z = np.load(sys.argv[1] if len(sys.argv) > 1 else P + 'c086_rst_samples.npz')   # c086_rst.py --samples FILE
L += ['', 'Drivers (3D, 10% random sample of all below-switch band cells, all restarts pooled):']
for lab in ('R6', 'R7'):
    SA = np.concatenate([z[k] for k in z.files if k.startswith(lab + '_') and k.endswith('all10pct')], axis=1)
    b, lp, lc, C, lv, f, lc0, f0 = SA
    r, r0 = lp - lc - np.log10(0.05), lp - lc0 - np.log10(0.05)
    dv = C >= 0
    L.append(f'  {lab}: n {SA.shape[1]}; log10[psi_v/(0.05 c_s)] p10/50/90 ' + '/'.join(f'{x:+.2f}' for x in np.percentile(r, [10, 50, 90]))
             + f' (>1 in {np.mean(r > 0):.3f}); cold c_s: ' + '/'.join(f'{x:+.2f}' for x in np.percentile(r0, [10, 50, 90]))
             + f' (>1 in {np.mean(r0 > 0):.4f})')
    L.append(f'      div v <= 0 in {dv.mean():.3f}; flagged {f.mean():.3f}; cold-flagged {f0.mean():.3f}; flagged among div<=0 {f[dv].mean():.3f}')
    for k in (2, 3, 4):
        s = lv == k
        L.append(f'      level {k}: n {s.sum():8d} median log10 psi_v {np.median(lp[s]):+.2f}  log10 c_s {np.median(lc[s]):+.2f}  flagged {f[s].mean():.3f}')
print('\n'.join(L))
open(P + 'c086_summary.txt', 'w').write('\n'.join(L) + '\n')
