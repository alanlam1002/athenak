#!/usr/bin/env python3
"""c311b (R-283 item 2(b); user-approved 2026-11-25; offline, read-only): where does the hot dilute class come from?
Backward tracing in the orbital-plane slices (prim_xy + dediag_xy, every 2 M): N parcels sampled mass-weighted from the class
H (rho < 1e-10, P/rho > 0.01, rho > 1e-22) in the slice nearest T are stepped back slice by slice, x <- x - dt v_c(x), with
v_c = W v / W + s xidot (W v^i = velx..; the coordinate velocity alpha v - beta ~ v + s xidot at r > 20 M, alpha ~ 1; the
frame-shift sign s in {-1, 0, +1} is calibrated per run by the dense gas: the s whose one-step back-advection of the
rho >= 1e-6 rho_c cells lands on the most similar rho, argued). 2D: vz is dropped (argued; the slice is z = +dz/2).
Along each path: rho, P/rho, the DE branch code, de_shock, de_sensor, the distance to dense gas (rho >= 1e-6 rho_c) and r_BH.
Per parcel the HEATING EVENT is the latest step back where P/rho was < 0.01 (the next later step is >= 0.01): its rho jump
(compression: rho_after/rho_before), whether de_shock was set at either end, the distance to dense gas there, and the cold rho.
Parcels hot over the whole span report 'hot at start'. Also: the fraction that was dense (rho >= 1e-6 rho_c) at any step.
    c311b_origin.py LABEL RUNDIR T [--back 120] [--n 4000] [--out PREFIX]"""
import argparse, glob, json, os, sys, types
import numpy as np
from scipy.spatial import cKDTree
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh, par   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('label'); ap.add_argument('rundir'); ap.add_argument('t', type=float)
ap.add_argument('--back', type=float, default=120.0); ap.add_argument('--n', type=int, default=4000)
ap.add_argument('--out', default=None); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--rbhmin', type=float, default=0.0); ap.add_argument('--tmin', type=float, default=0.01); ap.add_argument('--rhomin', type=float, default=1e-22)
a = ap.parse_args()
rhoc = par(a.rundir, 'rhoc'); RD = 1e-6*rhoc


def stime(f):
    with open(f, 'rb') as fh: h = fh.read(4000)
    i = h.find(b'time='); return float(h[i + 5:h.index(b'\n', i)])


def slices(kind):
    fs = sorted(glob.glob(f'{a.rundir}/bin/*.{kind}.*.bin'))
    return {round(stime(f), 6): f for f in fs}


SP, SD = slices('prim_xy'), slices('dediag_xy')
ts = sorted(t for t in SP if a.t - a.back - 1e-6 <= t <= a.t + 1.0)
ts = [t for t in ts if t <= min(SP, key=lambda q: abs(q - a.t)) + 1e-9]
ts = ts[::-1]   # backwards from the slice nearest T
hst = np.loadtxt(glob.glob(f'{a.rundir}/*.user.hst')[0])
def fvel(t): return np.array([np.interp(t, hst[:, 0], hst[:, 4 + q]) for q in range(3)])


class Slice:
    def __init__(self, t):
        P = bc.read_binary(SP[t]); self.t = P['time']
        self.g = np.asarray(P['mb_geometry'], float); self.n = P['nx1_out_mb']
        self.v = {k: np.stack([np.asarray(P['mb_data'][k][b])[0] for b in range(len(self.g))]).astype(float)
                  for k in ('dens', 'velx', 'vely', 'press')}
        self.d = None
        if t in SD:
            D = bc.read_binary(SD[t])
            self.d = {k: np.stack([np.asarray(D['mb_data'][k][b])[0] for b in range(len(self.g))]).astype(float)
                      for k in ('de_branch', 'de_shock', 'de_sensor')}
        h = (self.g[:, 1] - self.g[:, 0])/self.n; self.h = h
        xs = self.g[:, 0][:, None] + (np.arange(self.n) + 0.5)[None, :]*h[:, None]
        ys = self.g[:, 2][:, None] + (np.arange(self.n) + 0.5)[None, :]*h[:, None]
        self.X = np.broadcast_to(xs[:, None, :], (len(self.g), self.n, self.n))
        self.Y = np.broadcast_to(ys[:, :, None], (len(self.g), self.n, self.n))
        dn = self.v['dens'] >= RD
        self.tree = cKDTree(np.stack([self.X[dn], self.Y[dn]], 1)) if dn.any() else None
        # block lookup: sort by finest first not needed (leaves do not overlap)

    def locate(self, x, y):
        b = np.full(len(x), -1); i = np.zeros(len(x), int); j = np.zeros(len(x), int)
        for m, g in enumerate(self.g):
            s = (x >= g[0]) & (x < g[1]) & (y >= g[2]) & (y < g[3]) & (b < 0)
            if s.any():
                b[s] = m; i[s] = ((x[s] - g[0])/self.h[m]).astype(int).clip(0, self.n - 1)
                j[s] = ((y[s] - g[2])/self.h[m]).astype(int).clip(0, self.n - 1)
        return b, j, i

    def sample(self, x, y):
        b, j, i = self.locate(x, y); ok = b >= 0; bb = np.where(ok, b, 0)
        o = {k: np.where(ok, v[bb, j, i], np.nan) for k, v in self.v.items()}
        if self.d is not None:
            o.update({k: np.where(ok, v[bb, j, i], np.nan) for k, v in self.d.items()})
        o['dist'] = self.tree.query(np.stack([x, y], 1))[0] if self.tree is not None else np.full(len(x), np.nan)
        o['h'] = np.where(ok, self.h[bb], np.nan)
        return o


S0 = Slice(ts[0]); rng = np.random.default_rng(a.seed)
r, P = S0.v['dens'], S0.v['press']; T = np.where(r > 0, P/np.where(r > 0, r, 1), 0)
XB = xbh(a.rundir, S0.t)
cls = (r > a.rhomin) & (r < 1e-10) & (T > a.tmin) & (np.hypot(S0.X - XB[0], S0.Y - XB[1]) >= a.rbhmin)
if not cls.any():
    print(json.dumps(dict(label=a.label, t=S0.t, n=0, note='no class cells in the slice'))); sys.exit(0)
w = (r*S0.h[:, None, None]**2)[cls]; idx = rng.choice(w.size, size=min(a.n, w.size), p=w/w.sum())
x = S0.X[cls][idx].copy(); y = S0.Y[cls][idx].copy(); path_x0, path_y0 = x.copy(), y.copy()
print(f'{a.label}: slice t = {S0.t:.3f}; class cells in the slice {cls.sum()}, parcels {len(x)}; steps back {len(ts) - 1}', flush=True)

# calibrate s on the dense gas between the first two slices
S1 = Slice(ts[1]); dt = S0.t - S1.t; dn = S0.v['dens'] >= RD
sel = rng.choice(dn.sum(), size=min(20000, dn.sum()), replace=False)
xd, yd = S0.X[dn][sel], S0.Y[dn][sel]; o = S0.sample(xd, yd)
best = None
for s in (-1.0, 0.0, 1.0):
    fv = fvel(S0.t); W = np.sqrt(1 + o['velx']**2 + o['vely']**2)
    xb = xd - dt*(o['velx']/W + s*fv[0]); yb = yd - dt*(o['vely']/W + s*fv[1])
    ob = S1.sample(xb, yb); e = np.nanmedian(np.abs(np.log10(ob['dens']/o['dens'])))
    print(f'  calibration s = {s:+.0f}: median |dlog rho| one step back {e:.3f}', flush=True)
    if best is None or e < best[1]: best = (s, e)
s = best[0]

keys = ('dens', 'press', 'de_branch', 'de_shock', 'de_sensor', 'dist')
path = {k: [] for k in keys}; tl = []
cur = S0
for k in range(len(ts)):
    o = cur.sample(x, y)
    for q in keys: path[q].append(o.get(q, np.full(len(x), np.nan)))
    tl.append(cur.t)
    if k + 1 == len(ts): break
    nxt = Slice(ts[k + 1]); dt = cur.t - nxt.t
    fv = fvel(cur.t); W = np.sqrt(1 + o['velx']**2 + o['vely']**2)
    x = x - dt*(o['velx']/W + s*fv[0]); y = y - dt*(o['vely']/W + s*fv[1])
    cur = nxt
path = {q: np.array(v) for q, v in path.items()}   # [step, parcel], step 0 = T
Tp = path['press']/path['dens']
ev = []
for p in range(Tp.shape[1]):
    col = Tp[:, p]; cold = np.where(col < 0.01)[0]   # heating event: below the class's base threshold 0.01
    if len(cold) == 0: ev.append(None); continue
    k = cold[0]   # the most recent cold step (k steps back); k - 1 is hot
    ev.append(k)
ev_ok = np.array([e is not None for e in ev])
rbh_now = np.hypot(path_x0 - XB[0], path_y0 - XB[1]); kk = np.array([e if e is not None else 0 for e in ev])
P_ = np.arange(Tp.shape[1])
def at(q, off): return path[q][np.clip(kk + off, 0, Tp.shape[0] - 1), P_]
comp = at('dens', -1)/at('dens', 0)
shock = (np.nan_to_num(at('de_shock', -1)) > 0) | (np.nan_to_num(at('de_shock', 0)) > 0)
wasdense = (path['dens'] >= RD).any(0)
R = dict(rbhmin=a.rbhmin, tmin=a.tmin, rhomin=a.rhomin, label=a.label, run=a.rundir, t=S0.t, s_frame=s, calib=best[1], n=int(len(P_)), steps=len(tl), t_back=tl[-1],
         f_event=float(ev_ok.mean()), f_hot_whole_span=float((~ev_ok).mean()),
         event_age_q=np.percentile([tl[0] - tl[e] for e in kk[ev_ok]], [10, 50, 90]).tolist() if ev_ok.any() else [],
         event_compression_q=np.nanpercentile(comp[ev_ok], [10, 50, 90]).tolist() if ev_ok.any() else [],
         f_event_compressive=float(np.nanmean(comp[ev_ok] > 1.5)) if ev_ok.any() else np.nan,
         f_event_shockflag=float(shock[ev_ok].mean()) if ev_ok.any() else np.nan,
         event_dist_dense_q=np.nanpercentile(at('dist', 0)[ev_ok], [10, 50, 90]).tolist() if ev_ok.any() else [],
         event_rho_cold_q=np.nanpercentile(at('dens', 0)[ev_ok], [10, 50, 90]).tolist() if ev_ok.any() else [],
         event_T_after_q=np.nanpercentile(Tp[np.clip(kk - 1, 0, None), P_][ev_ok], [10, 50, 90]).tolist() if ev_ok.any() else [],
         f_was_dense=float(wasdense.mean()), now_rbh_q=np.percentile(rbh_now, [10, 50, 90]).tolist(),
         now_rho_q=np.nanpercentile(path['dens'][0], [10, 50, 90]).tolist(), now_T_q=np.nanpercentile(Tp[0], [10, 50, 90]).tolist(),
         now_dist_dense_q=np.nanpercentile(path['dist'][0], [10, 50, 90]).tolist(),
         now_codes={str(int(c)): float(np.mean(path['de_branch'][0] == c)) for c in np.unique(path['de_branch'][0][~np.isnan(path['de_branch'][0])])})
print(json.dumps(R, default=float))
if a.out:
    json.dump(R, open(a.out + '.json', 'w'), indent=1, default=float)
    np.savez_compressed(a.out + '_paths.npz', t=np.array(tl), **path)
