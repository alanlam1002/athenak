#!/usr/bin/env python3
"""c247 (R-247 item 1): gates for dual_energy_resync_gate (wip/resync-gate) from c247_gate/cpu (mk247c.py decks, t247c.sh).
  default bitwise: every out/def/<sub>/<deck>_af vs _rsg tab file byte for byte; p18 Pd_af vs Pd_rsg hst byte for byte.
  ghost/owner:     out/ghost_v3/mb1_<g> vs mb79_<g> (x1 [19, 38.75], nx1 5056: dx 1/256 exact; g = cnd control, cgc, cgq): the final prim and de_diag tabs, value by value.
  mass:            p18 Pc / Pq mhd.hst: max |M(t) - M(0)|/M(0) (column 'mass').
  restart:         p18 Pc4 vs Pc4r (restarted from the cycle-2 rst): the hst rows both have, column by column.
  regressions:     shocked-slab f_RH (c190's slab), hom K range, building-shock entropy excess X(t), each gate (cgc, cgq)
                   vs the plain conditioning arm (cnd); plus the code-18 counts.
    c247_gate_metrics.py CPUDIR"""
import filecmp, glob, os, re, sys
import numpy as np
GAM = 5.0/3.0
D = sys.argv[1]
O = f'{D}/out'
H = 'cfc_tde_wd_imbh_parabolic_tracker'


def tabs(d, kind):
    out = []
    for fn in sorted(glob.glob(f'{d}/tab/*{kind}*.tab')):
        t = float(open(fn).readline().split('time=')[1].split()[0])
        out.append((t, np.loadtxt(fn, ndmin=2)))
    return out


def prim(d, k=-1):
    t, a = tabs(d, 'mhd_w_bcc')[k]
    return t, a[:, 2], a[:, 3], a[:, 7]


def krh(M):
    p = (2*GAM*M**2 - (GAM - 1))/(GAM + 1)
    r = (GAM + 1)*M**2/((GAM - 1)*M**2 + 2)
    return p/r**GAM, r


def slab_K(d, M):
    t, x, rho, P = prim(d); _, _, rho0, P0 = prim(d, 0)
    K0 = np.median(P0/rho0**GAM); r0 = np.median(rho0); _, rr = krh(M)
    idx = np.nonzero(rho > r0*(1 + 0.5*(rr - 1)))[0]
    if len(idx) < 5:
        return np.nan
    lo, hi = idx.min(), idx.max(); w = hi - lo
    sel = np.arange(lo + int(0.2*w), hi - int(0.2*w) + 1)
    return np.median(P[sel]/rho[sel]**GAM)/K0 - 1.0


def codes(d):
    T = tabs(d, 'mhd_de_diag')
    return {int(v): int(n) for v, n in zip(*np.unique(T[-1][1][:, 3], return_counts=True))} if T else {}


# ---- default bitwise
nf = nd = 0; bad = []
for a in sorted(glob.glob(f'{O}/def/*/*_af')):
    b = a[:-3] + '_rsg'
    fa = sorted(os.path.relpath(f, a) for f in glob.glob(f'{a}/tab/*.tab'))
    fb = sorted(os.path.relpath(f, b) for f in glob.glob(f'{b}/tab/*.tab'))
    nd += 1
    if fa != fb or not fa:
        bad.append(f'{a}: file lists differ ({len(fa)} vs {len(fb)})'); continue
    for f in fa:
        nf += 1
        if not filecmp.cmp(f'{a}/{f}', f'{b}/{f}', shallow=False):
            bad.append(f'{a}/{f}')
print(f'== default bitwise (1D): {nd} deck pairs, {nf} tab files compared, {len(bad)} differ')
for x in bad[:10]:
    print('   DIFF', x)
for h in ('mhd', 'user'):
    fa, fb = f'{O}/p18/Pd_af/{H}.{h}.hst', f'{O}/p18/Pd_rsg/{H}.{h}.hst'
    if os.path.exists(fa) and os.path.exists(fb):
        print(f'== default bitwise (p18 Pd {h}.hst): {"IDENTICAL" if filecmp.cmp(fa, fb, shallow=False) else "DIFFER"}')
    else:
        print(f'== p18 Pd {h}.hst missing')

# ---- ghost/owner
for g in ('cnd', 'cgc', 'cgq'):   # cnd = the gate-off control
    a, b = f'{O}/ghost_v3/mb1_{g}', f'{O}/ghost_v3/mb79_{g}'
    for kind in ('mhd_w_bcc', 'mhd_de_diag'):
        A, B = tabs(a, kind), tabs(b, kind)
        if not A or not B:
            print(f'== ghost/owner {g} {kind}: missing'); continue
        # columns 0-1 are gid and the block-local i (differ by construction); 2 is x1v (per-block geometry); 3+ the data
        n = min(len(A), len(B)); md = 0.0; same = True; mx = 0.0
        for k in range(n):
            x, y = A[k][1], B[k][1]
            if x.shape != y.shape:
                same = False; md = np.inf; continue
            if not np.array_equal(x[:, 3:], y[:, 3:]):
                same = False; md = max(md, float(np.nanmax(np.abs(x[:, 3:] - y[:, 3:]))))
            mx = max(mx, float(np.max(np.abs(x[:, 2] - y[:, 2]))))
        print(f'== ghost/owner {g} {kind}: {n} dumps, data columns {"IDENTICAL" if same else f"DIFFER max|d| {md:.3e}"}'
              f' (x1v max|d| {mx:.1e})')


# ---- mass and restart
def hst(f):
    L = [l for l in open(f)]
    hd = [l for l in L if l.startswith('#')][-1]
    names = [re.sub(r'^\[\d+\]=', '', c) for c in hd[1:].split()]
    return names, [l.split() for l in L if not l.startswith('#')]


for a in ('Pd_rsg', 'Pc', 'Pq'):
    f = f'{O}/p18/{a}/{H}.mhd.hst'
    if not os.path.exists(f):
        print(f'== mass {a}: missing'); continue
    nm, R = hst(f); c = nm.index('mass') if 'mass' in nm else 2
    m = np.array([float(r[c]) for r in R])
    print(f'== mass {a}: {len(m)} rows, max |M - M0|/M0 = {np.max(np.abs(m - m[0]))/m[0]:.3e}')
for h in ('mhd', 'user'):
    fa, fb = f'{O}/p18/Pc4/{H}.{h}.hst', f'{O}/p18/Pc4r/{H}.{h}.hst'
    if not (os.path.exists(fa) and os.path.exists(fb)):
        print(f'== restart {h}: missing'); continue
    _, A = hst(fa); _, B = hst(fb)
    ca = {r[1]: r for r in A}; cb = {r[1]: r for r in B}
    common = sorted(set(ca) & set(cb), key=float)
    diff = [c for c in common if ca[c] != cb[c]]
    print(f'== restart {h}.hst: cycles in both {len(common)} ({common[:1]}..{common[-1:]}), rows differing {len(diff)}')

# ---- regressions
print('== shocks: f_RH = (K - 1)/(K_RH - 1) in the shocked slab at t_end | code counts (cgc, cgq; 18 = gated)')
for d in sorted(glob.glob(f'{O}/gate/*/*_cnd')):
    base, nm = d[:-4], d.split('/')[-1][:-4]
    m = re.search(r'_M([\d.]+)', nm)
    if not m or '/build/' in d:
        continue
    kr = krh(float(m.group(1)))[0] - 1
    v = {}
    for a in ('cnd', 'cgc', 'cgq'):
        try:
            v[a] = slab_K(f'{base}_{a}', float(m.group(1)))/kr
        except Exception:   # noqa
            v[a] = np.nan
    print(f'{nm:34s} cnd {v["cnd"]:7.4f} cgc {v["cgc"]:7.4f} cgq {v["cgq"]:7.4f} | cgc {codes(base + "_cgc")} | cgq {codes(base + "_cgq")}')
for d in sorted(glob.glob(f'{O}/gate/flat/hom_c*_cnd')):
    base, out = d[:-4], []
    for a in ('cnd', 'cgc', 'cgq'):
        t, x, rho, P = prim(f'{base}_{a}'); _, _, r0, P0 = prim(f'{base}_{a}', 0)
        K0 = np.median(P0/r0**GAM); dn = rho > 0.3*rho.max(); K = P[dn]/rho[dn]**GAM/K0 - 1
        out.append(f'{a} [{K.min():+.2e}, {K.max():+.2e}]')
    print(f'{base.split("/")[-1]:12s} ' + '  '.join(out))
for d in sorted(glob.glob(f'{O}/gate/flat/sod_cnd')):
    _, _, ro, Po = prim(d)
    for a in ('cgc', 'cgq'):
        _, _, r, P = prim(d[:-4] + '_' + a)
        print(f'sod {a} vs cnd: max |dP|/P {np.max(np.abs(P - Po)/Po):.2e}')
print('== building shock: X = sum rho (K/K0 - 1) dx at t = 100/200/300/400')
for d in sorted(glob.glob(f'{O}/gate/build/*_cnd')):
    base = d[:-4]; row = {}
    for a in ('cnd', 'cgc', 'cgq'):
        T = tabs(f'{base}_{a}', 'mhd_w_bcc')
        a0 = T[0][1]; K0 = np.median(a0[:, 7]/a0[:, 3]**GAM); dx = a0[1, 2] - a0[0, 2]
        row[a] = {round(t): (a1[:, 3]*(a1[:, 7]/a1[:, 3]**GAM/K0 - 1)).sum()*dx for t, a1 in T}
    for tt in (100, 200, 300, 400):
        print(f'{base.split("/")[-1]:18s} t={tt:3d}: ' + ' '.join(f'{a} {row[a].get(tt, np.nan):+.4e}' for a in ('cnd', 'cgc', 'cgq'))
              + f' | cgc {codes(base + "_cgc")}')
