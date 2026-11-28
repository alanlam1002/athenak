#!/usr/bin/env python3
"""c316 (R-282 / RL-043, K0; offline, read-only): the frozen-metric arms (c310 K08, K04) against the reference K0R, from the logs,
the histories and the orbital-plane slices:
  A. per window (approach 120-441.1, C0->P 441.1-491.1, P->C1 491.1-541.1): MG solves and V-cycles per M per field (MGLOG lines:
     'MGLOG <field> cyc N t T it IT ...'), the wall node-h per M (runs.log 'nh' per segment), MCLOSE PASS/FAIL count and the max
     |R/M*|, M_BH - M0 at the end ('### CFC accretion ... M_BH-M0=').
  B. user.hst: x_BH(t) (c160_coldmax.xbh: puncture_x - frame_disp), the frame velocity, fc_offset, rho_max; the arm minus K0R
     interpolated onto the arm's times: max |dx_BH|, max |d rho_max|/rho_max per window.
  C. prim_xy + adm_xy every --every slices: the star's centre in the plane (rho >= 1e-6 rho_c centroid) and its offset from K0R's at
     the same t; psi (= psi4^(1/4)) and alpha at the star's centre and at 4 M from the hole towards the star (nearest cell).
    c316_k0.py OUTPREFIX K0R_DIR K08_DIR [K04_DIR ...] [--every 4]"""
import argparse, glob, os, re, sys, types, json
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('dirs', nargs='+'); ap.add_argument('--every', type=int, default=4)
a = ap.parse_args()
RC = 3.24e-4
WIN = [(120.0, 441.1), (441.1, 491.1), (491.1, 541.1)]


def logs(d):
    d = os.path.abspath(d); name = os.path.basename(d); E = os.path.dirname(d)
    mg = {}
    acc = None; mc = []
    for l in open(f'{d}/out.log', errors='replace'):
        if l.startswith('MGLOG '):
            p = l.split(); f = p[1]; t = float(p[5]); it = int(p[7]); w = float(p[9])
            mg.setdefault(f, []).append((t, it, w))
        elif l.startswith('### CFC accretion'):
            m = re.search(r'M_BH-M0=([-0-9.e+]+)', l); acc = float(m.group(1)) if m else acc
        elif l.startswith('MCLOSE cycles'):
            m = re.search(r'R/M\*=([-0-9.e+]+)', l); mc.append((float(m.group(1)) if m else np.nan, 'PASS' in l))
    out = {'MBH_minus_M0_end': acc, 'mclose_n': len(mc), 'mclose_pass': int(sum(p for _, p in mc)),
           'mclose_max_RoverMs': float(np.nanmax(np.abs([r for r, _ in mc]))) if mc else None}
    for (t0, t1) in WIN:
        key = f'{t0:g}-{t1:g}'
        for f, v in mg.items():
            v = np.array(v); s = (v[:, 0] >= t0) & (v[:, 0] < t1)
            out[f'mg_{f}_solves_perM_{key}'] = float(s.sum()/(t1 - t0))
            out[f'mg_{f}_vcyc_perM_{key}'] = float(v[s, 1].sum()/(t1 - t0))
            out[f'mg_{f}_wall_perM_{key}'] = float(v[s, 2].sum()/(t1 - t0))
    # wall: runs.log segments of this arm (start t from the restart's stage, end t from 'last=... time=')
    nh = {}
    segs = [l for l in open(f'{E}/runs.log') if l.startswith(f'{name} rc=')]
    tprev = 120.0
    for l in segs:
        m = re.search(r'time=([0-9.e+]+)', l); n = re.search(r' nh ([0-9.]+)', l)
        if m and n:
            t = float(m.group(1)); nh[f'{tprev:g}-{t:g}'] = float(n.group(1)); tprev = t
    out['nh_segments'] = nh
    out['nh_total'] = float(sum(nh.values()))
    out['nh_perM_total'] = out['nh_total']/(tprev - 120.0) if tprev > 120 else None
    return out


def hst(d):
    f = glob.glob(f'{d}/*.user.hst')[0]
    A = np.array([[float(x) for x in l.split()] for l in open(f) if l.strip() and not l.startswith('#')])
    _, i = np.unique(A[:, 0], return_index=True); A = A[i]   # later rows win on duplicate times are rare here; unique keeps the first
    xb = np.array([xbh(d, t) for t in A[:, 0]])
    return A[:, 0], xb, A[:, 2], A[:, 14]   # t, x_BH, rho_max, fc_offset


def stime(f):
    with open(f, 'rb') as fh: h = fh.read(4000)
    i = h.find(b'time='); return float(h[i + 5:h.index(b'\n', i)])


def plane(d):
    fp = sorted(glob.glob(f'{d}/bin/*.prim_xy.*.bin'))[::a.every]
    rows = []
    for f in fp:
        fa = f.replace('prim_xy', 'adm_xy')
        if not os.path.exists(fa): continue
        P = bc.read_binary(f); Q = bc.read_binary(fa)
        g = np.asarray(P['mb_geometry'], float); n = P['nx1_out_mb']; h = (g[:, 1] - g[:, 0])/n
        rho = np.stack([np.asarray(P['mb_data']['dens'][b])[0] for b in range(len(g))]).astype(float)
        X = g[:, 0][:, None, None] + (np.arange(n) + 0.5)[None, None, :]*h[:, None, None] + 0*rho
        Y = g[:, 2][:, None, None] + (np.arange(n) + 0.5)[None, :, None]*h[:, None, None] + 0*rho
        s = rho >= 1e-6*RC; w = (rho*h[:, None, None]**2)[s]
        xc = np.array([(X[s]*w).sum()/w.sum(), (Y[s]*w).sum()/w.sum()])
        xb = xbh(d, P['time'])[:2]
        def at(x, y, var):
            bb = np.where((g[:, 0] <= x) & (x < g[:, 1]) & (g[:, 2] <= y) & (y < g[:, 3]))[0]
            if len(bb) == 0: return np.nan
            b = bb[np.argmax(g[bb, 1] - g[bb, 0] == (g[bb, 1] - g[bb, 0]).min())]
            i = min(int((x - g[b, 0])/h[b]), n - 1); j = min(int((y - g[b, 2])/h[b]), n - 1)
            return float(np.asarray(Q['mb_data'][var][b])[0, j, i])
        u = (xc - xb)/np.linalg.norm(xc - xb); xn = xb + 4.0*u
        rows.append(dict(t=P['time'], xc=xc[0], yc=xc[1], xbh=xb[0], ybh=xb[1],
                         psi_star=at(*xc, 'adm_psi4')**0.25, alpha_star=at(*xc, 'adm_alpha'),
                         psi_hole4=at(*xn, 'adm_psi4')**0.25, alpha_hole4=at(*xn, 'adm_alpha')))
    return rows


R = {}
for d in a.dirs:
    nm = os.path.basename(d.rstrip('/'))
    R[nm] = dict(logs=logs(d)); t, xb, rm, fo = hst(d); R[nm]['hst'] = (t, xb, rm, fo); R[nm]['plane'] = plane(d)
    print(nm, 'done', flush=True)
ref = os.path.basename(a.dirs[0].rstrip('/'))
T0, XB0, RM0, _ = R[ref]['hst']; P0 = R[ref]['plane']
summary = {}
for nm, v in R.items():
    S = dict(v['logs'])
    t, xb, rm, fo = v['hst']
    for (t0, t1) in WIN:
        s = (t >= t0) & (t < t1) & (t <= T0[-1])
        if s.any():
            dx = max(np.max(np.abs(xb[s, q] - np.interp(t[s], T0, XB0[:, q]))) for q in range(3))
            drm = np.max(np.abs(rm[s] - np.interp(t[s], T0, RM0))/np.interp(t[s], T0, RM0))
            S[f'max_dxBH_{t0:g}-{t1:g}'] = float(dx); S[f'max_drhomax_rel_{t0:g}-{t1:g}'] = float(drm)
    pl = v['plane']; pt = {round(r['t'], 3): r for r in P0}
    diffs = []
    for r in pl:
        q = min(P0, key=lambda z: abs(z['t'] - r['t']))
        if abs(q['t'] - r['t']) > 0.3: continue
        diffs.append((r['t'], np.hypot(r['xc'] - q['xc'], r['yc'] - q['yc']), r['psi_star'] - q['psi_star'], r['alpha_star'] - q['alpha_star'],
                      r['psi_hole4'] - q['psi_hole4'], r['alpha_hole4'] - q['alpha_hole4'], r['psi_star'], r['alpha_star'], r['psi_hole4'], r['alpha_hole4']))
    diffs = np.array(diffs) if diffs else np.zeros((0, 10))
    for (t0, t1) in WIN:
        s = (diffs[:, 0] >= t0) & (diffs[:, 0] < t1) if len(diffs) else np.zeros(0, bool)
        if s.any():
            for k, c in (('dstar_centre', 1), ('dpsi_star', 2), ('dalpha_star', 3), ('dpsi_hole4', 4), ('dalpha_hole4', 5)):
                S[f'max_{k}_{t0:g}-{t1:g}'] = float(np.max(np.abs(diffs[s, c])))
            S[f'psi_star_range_{t0:g}-{t1:g}'] = [float(diffs[s, 6].min()), float(diffs[s, 6].max())]
            S[f'alpha_hole4_range_{t0:g}-{t1:g}'] = [float(diffs[s, 9].min()), float(diffs[s, 9].max())]
    summary[nm] = S
json.dump(summary, open(a.out + '.json', 'w'), indent=1, default=float)
for nm, S in summary.items():
    print('==', nm); [print(f'  {k}: {v}') for k, v in S.items()]
