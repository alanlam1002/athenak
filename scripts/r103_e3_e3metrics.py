import glob, sys, numpy as np
D = sys.argv[1]
GAM, K0 = 5/3, 0.01 / 1.0**(5/3)
def tabs(d, kind):
    out = []
    for fn in sorted(glob.glob(f'{d}/tab/*{kind}*.tab')):
        t = float(open(fn).readline().split('time=')[1].split()[0])
        out.append((t, np.loadtxt(fn, ndmin=2)))
    return out
def krh(M):
    p = (2*GAM*M**2 - (GAM-1))/(GAM+1); r = (GAM+1)*M**2/((GAM-1)*M**2 + 2); return p/r**GAM, r
def slab(x, rho, rc):
    m = rho > (1 + 0.5*(rc - 1)) * np.median(rho)    # shocked (compressed) gas
    i = np.nonzero(m)[0]
    return x[i.min()], x[i.max()]
rows = []
for M in ('1.2', '1.5', '2', '3'):
    kr, rc = krh(float(M))
    for vb in ('+0.0', '+0.3', '-0.3'):
        base = f'{D}/col_M{M}_vb{vb}'
        t, a = tabs(base + '_off', 'mhd_w_bcc')[-1]
        x = a[:, 2]; dx = x[1] - x[0]
        K = a[:, 7] / a[:, 3]**GAM / K0
        xl, xr = slab(x, a[:, 3], rc); c = 0.5*(xl + xr); hw = 0.5*(xr - xl)
        win = (np.abs(x - c) > 0.04) & (np.abs(x - c) < min(0.10, hw - 4*dx))
        koff = np.median(K[win]) - 1
        line = f'M={M:3s} vb={vb}: slab {xl:.3f}-{xr:.3f} (hw {hw:.3f}); energy-only K-1 {koff:.4e} (RH {kr-1:.4e})'
        for v in ('coord', 'covariant'):
            t, b = tabs(f'{base}_{v}', 'mhd_w_bcc')[-1]
            Kv = b[:, 7] / b[:, 3]**GAM / K0
            cap = (np.median(Kv[win]) - 1) / koff
            # flag location over all dumps
            at = inside = outside = 0
            for (tw, w), (td, dg) in zip(tabs(f'{base}_{v}', 'mhd_w_bcc'), tabs(f'{base}_{v}', 'mhd_de_diag')):
                if tw == 0: continue
                f = dg[:, 5] > 0.5
                try:
                    l, r = slab(w[:, 2], w[:, 3], rc)
                except ValueError:
                    continue
                d = np.minimum(np.abs(x - l), np.abs(x - r))
                at += int((f & (d <= 4*dx)).sum())
                inside += int((f & (d > 4*dx) & (x > l) & (x < r)).sum())
                outside += int((f & (d > 4*dx) & ((x < l) | (x > r))).sum())
            line += f' | {v}: capture {cap:.3f}, flagged at shocks {at}, slab interior {inside}, outside {outside}'
        rows.append(line)
for vb in ('+0.0', '+0.3', '-0.3'):
    base = f'{D}/smooth_vb{vb}'
    t, a = tabs(base + '_off', 'mhd_w_bcc')[-1]
    Ko = a[:, 7] / a[:, 3]**GAM / K0
    line = f'smooth vb={vb}: energy-only K/K0-1 range {Ko.min()-1:+.2e}..{Ko.max()-1:+.2e}'
    for v in ('coord', 'covariant'):
        nfl = sum(int((dg[:, 5] > 0.5).sum()) for td, dg in tabs(f'{base}_{v}', 'mhd_de_diag') if td > 0)
        t, b = tabs(f'{base}_{v}', 'mhd_w_bcc')[-1]
        Kv = b[:, 7] / b[:, 3]**GAM / K0
        line += f' | {v}: flagged cell-dumps {nfl}, K/K0-1 {Kv.min()-1:+.2e}..{Kv.max()-1:+.2e}'
    rows.append(line)
print('\n'.join(rows))
