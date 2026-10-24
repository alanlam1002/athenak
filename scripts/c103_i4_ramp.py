"""R-112 item 4 ramp validation (c103_i4): the R7 comoving-exit ramp arms against ref7's R7
u/s1/s2 ensemble. Per history quantity, at the ramp run's output times: the ensemble mean and
sigma_diff (RMS of the three pairwise differences, n = 3), and the ramp arm's deviation
|x - mean|/sigma_diff (max and at t_end). Frame-independent rows (rho-max, the rest mass, the
entropy tracer total) should stay within the spread; the frame rows change by design. Also the
zone's capped-target count (ATMZONE n_cap: cells whose grid-rest target exceeded
atmosphere_zone_vmax) before the ramp, during it and after it.

    c103_i4_ramp.py RUNDIR REFDIR OUT.csv
"""
import re
import sys

import numpy as np

D, REF, CSV = sys.argv[1], sys.argv[2], sys.argv[3]
BN = 'cfc_tde_wd_imbh_parabolic_tracker'


def hst(d, f):
    hdr = [l for l in open(f'{d}/{BN}.{f}.hst') if l.startswith('#')][-1]
    names = [x.split('=')[-1] for x in hdr.replace('#', '').split()]
    a = np.loadtxt(f'{d}/{BN}.{f}.hst')
    _, i = np.unique(a[:, 0], return_index=True)   # the terminal row repeats a time
    return names, a[np.sort(i)]


def ncap(d):
    t, n = [], []
    for l in open(f'{d}/out.log'):
        if l.startswith('ATMZONE cycle='):
            t.append(float(re.search(r' time=(\S+)', l).group(1)))
            n.append(float(re.search(r' n_cap=(\S+)', l).group(1)))
    return np.array(t), np.array(n)


def tswitch(d):
    for l in open(f'{d}/out.log'):
        if 'CFC frame controller: switch to ramp' in l:
            return float(re.search(r' at t = (\S+)', l).group(1))
    return np.nan


rows = ['arm,file,quantity,max_dev_sigma,end_dev_sigma,end_value,end_ref_mean,end_sigma_diff']
for arm in ('rampA', 'rampB'):
    for f, qs in (('user', ('rho-max', 'alpha-min', 'frame_vel_x1', 'frame_vel_x2')),
                  ('mhd', ('mass', 'scal-0', 'tot-E'))):
        names, a = hst(f'{D}/{arm}', f)
        t = a[:, 0]
        refs = []
        for m in ('u', 's1', 's2'):
            rn, r = hst(f'{REF}/R7_{m}', f)
            refs.append((rn, r))
        for q in qs:
            c = names.index(q)
            ens = np.array([np.interp(t, r[:, 0], r[:, rn.index(q)]) for rn, r in refs])
            sig = np.sqrt(np.mean([(ens[0] - ens[1])**2, (ens[0] - ens[2])**2,
                                   (ens[1] - ens[2])**2], axis=0))
            dev = np.abs(a[:, c] - ens.mean(0))/np.maximum(sig, 1e-300)
            rows.append(f'{arm},{f},{q},{dev.max():.3g},{dev[-1]:.3g},{a[-1, c]:.10g},'
                        f'{ens.mean(0)[-1]:.10g},{sig[-1]:.3g}')
    ts = tswitch(f'{D}/{arm}')
    tc, nc = ncap(f'{D}/{arm}')
    pre, dur, post = nc[tc < ts], nc[(tc >= ts) & (tc < ts + 20.0)], nc[tc >= ts + 20.0]
    m = lambda x: f'{x.mean():.3g}' if len(x) else 'nan'
    rows.append(f'{arm},ATMZONE,n_cap_mean(pre|ramp|post; t_sw={ts:.3f}),{m(pre)},{m(dur)},'
                f'{m(post)},,')
open(CSV, 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
