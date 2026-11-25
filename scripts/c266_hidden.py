#!/usr/bin/env python3
"""c266 (R-261, user-approved 2026-11-12): the hidden-energy budget of D-ent's tracer cells, offline from the co-located
prim_3d + dediag_3d dumps of one run lineage (no new runs).

Source (proj/tde 629d6725, src/eos/primitive_solver_hyd.hpp): in a tracer cell (codes 11, 12, 13; dec == 1, :2700-2715) the
stored (D, S, tau, DY) are kept bitwise and the primitives (so the prim_3d dump) are the tracer's (rho_t, P_t; :2703-2704);
de_diag slot 3 = dken = K_en/K_tr - 1 from the cell's own energy C2P (:2687, :2743), slot 4 = ddk = K_resync/K_tr - 1 at a
resync (code 14, :2720-2721, :2744), slot 0 = the branch code (:2741). Code 13 (energy C2P floored/failed or tau <= 0) does
not form dken (:2675-2677, left 0), so its hidden energy is not measurable from the dumps (its mass is reported).
Per tracer cell (11, 12):
  Delta E_hid = sqrt(gamma) (tau - tau_tr) dV = sqrt(gamma) rho_t eps_t dken dV
(at fixed D, S: dtau = rho deps + O(eps drho); checked against the exact entropy inversion of :1406-1418 to <= 6e-4 for
v <= 0.5, eps 1e-6..1e-4, dken -0.3..16). The class's thermal energy E_th = sum sqrt(gamma) rho_t eps_t dV over its tracer
cells, so H/E_th is the eps-weighted mean dken. sqrt(gamma) = (1 + 1/(2 r_BH))^6 (puncture far field; within 3.5e-3 of a
restart's psi^6 for r_BH >= 3, L32T 00017); cells with r_BH < 3 are left out. dM = sqrt(gamma) rho W dV, W = sqrt(1 + psi4 u^2).
Classes (C-193's, c193_code_k.py): core rho >= 0.1 rho_c; outer 0.01-0.1 rho_c; remnant = inside the self-consistent Hill
sphere r_H about x_c (centroid of rho >= 0.01 rho_c, else of rho >= 0.01 rho_max: c253 --seed rel), rho >= rho_sw; stream =
outside r_H, 1e-6 <= rho/rho_c < 0.01.
Outputs (OUT_*.csv):
  budget      per dump x class: tracer mass by code, H_hot (dken > 0), H_cold (< 0), E_th, H/E_th, the code-13 mass
  quant       per dump x class: mass-weighted 5/25/50/75/95% quantiles of dken over codes 11 + 12
  trans       per interval x class(at the first dump) x transition (11/12/13 -> 14, 15, 0, 16/17), x r/R* bin: mass, and
              the hidden energy carried (from the first dump; 13 carries mass only). Cells are followed by
              x + (alpha u/W) dt + dx_BH (straight line over the interval; alpha = (1 - 1/2r)/(1 + 1/2r); the frame from the
              user.hst frame_disp; the hole's own shift left out) and the code at the second dump is read where they land.
  resync      per dump x class: the in-situ release at resyncs in the dump's C2P, sum sqrt(gamma) rho eps ddk/(1+ddk) dV over
              code 14, and its rate x 3/dt (rk3: a resync each stage; dt = the mean step between the dumps)
    c266_hidden.py OUT_PREFIX TP RUNDIR[:TMAX] [RUNDIR[:TMAX] ...] [--every 1] [--rstar 0.677]"""
import argparse, csv, glob, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
import bin_convert as bc   # noqa: E402
from c160_coldmax import xbh, par   # noqa: E402

GAM = 5.0 / 3.0
RBINS = np.array([0.0, 0.5, 0.8, 1.0, 1.5, 3.0, 10.0, np.inf])
TRC = (11, 12, 13)
DEST = {14: '14', 15: '15', 0: '0', 16: '16/17', 17: '16/17'}
ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('tp', type=float); ap.add_argument('runs', nargs='+')
ap.add_argument('--every', type=int, default=1); ap.add_argument('--tol', type=float, default=0.0); ap.add_argument('--rstar', type=float, default=0.677)
ap.add_argument('--nohole', type=float, default=None, help='RH: an isolated remnant (no puncture): psi = alpha = 1, no r_BH cut, remnant = r < RH about x_c')
a = ap.parse_args()


def btime(f):
    with open(f, 'rb') as fh:
        for ln in fh.read(512).split(b'\n'):
            if ln.strip().startswith(b'time='):
                return float(ln.split(b'=')[1])


def bcycle(f):
    with open(f, 'rb') as fh:
        for ln in fh.read(512).split(b'\n'):
            if ln.strip().startswith(b'cycle='):
                return int(ln.split(b'=')[1])


# the lineage's (dediag, prim) pairs in time order; a piece is used for t <= its TMAX and after the previous piece's
pairs, t0 = [], -1e9
for s in a.runs:
    d, _, tm = s.partition(':')
    tm = float(tm) if tm else 1e9
    tpr = {btime(f): f for f in glob.glob(f'{d}/bin/*.prim_3d.0*.bin') + glob.glob(f'{d}/bin/*.prim_3d_peri.0*.bin')}
    for fd in sorted(glob.glob(f'{d}/bin/*.dediag_3d.*.bin')):
        t = btime(fd)
        if not (t0 + 1e-9 < t <= tm + 1e-9):
            continue
        tq = min(tpr, key=lambda u: abs(u - t)) if tpr else None
        if tq is not None and abs(tq - t) <= a.tol:   # the same output time, or (De2) the nearest prim within --tol
            pairs.append((t, d, fd, tpr[tq]))
    t0 = tm
pairs.sort()
pairs = pairs[::a.every] if a.every > 1 else pairs


class Lookup:
    """point -> flat cell index in a dump (blocks are disjoint cubes; one key table per block size)"""
    def __init__(self, g, n):
        self.g, self.n = g, n
        self.x0 = np.array([g[:, 0].min(), g[:, 2].min(), g[:, 4].min()])
        sz = g[:, 1] - g[:, 0]
        self.tabs = []
        for s in np.unique(np.round(sz, 10)):
            b = np.where(np.abs(sz - s) < 1e-9 * s)[0]
            ijk = np.rint((g[b][:, [0, 2, 4]] - self.x0) / s).astype(np.int64)
            key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
            o = np.argsort(key)
            self.tabs.append((s, key[o], b[o]))

    def __call__(self, X):
        out = np.full(len(X), -1, np.int64)
        for s, keys, b in self.tabs:
            todo = out < 0
            if not todo.any():
                break
            ijk = np.floor((X[todo] - self.x0) / s).astype(np.int64)
            ok = (ijk >= 0).all(1) & (ijk < (1 << 21)).all(1)
            key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
            p = np.clip(np.searchsorted(keys, key), 0, len(keys) - 1)
            hit = ok & (keys[p] == key)
            idx = np.where(todo)[0][hit]
            bb = b[p[hit]]
            n = self.n
            c = np.clip(np.floor((X[idx] - self.g[bb][:, [0, 2, 4]]) / (s / n)).astype(np.int64), 0, n - 1)
            out[idx] = bb * n**3 + c[:, 2] * n * n + c[:, 1] * n + c[:, 0]
        return out


def load(t, d, fd, fp):
    Dg, P = bc.read_binary(fd), bc.read_binary(fp)
    g = np.asarray(Dg['mb_geometry'], float); n = Dg['nx1_out_mb']
    gp = np.asarray(P['mb_geometry'], float)
    coloc = Dg['time'] == P['time'] and np.array_equal(g, gp)
    catb = lambda S, k, gg: np.concatenate([np.asarray(S['mb_data'][k][b]).ravel() for b in range(len(gg))]).astype(float)
    cat = lambda S, k: catb(S, k, g)
    nb = len(g); h = (g[:, 1] - g[:, 0]) / n
    ax = lambda lo: g[:, lo][:, None] + (np.arange(n) + 0.5)[None, :] * h[:, None]
    x = np.broadcast_to(ax(0)[:, None, None, :], (nb, n, n, n)).ravel()
    y = np.broadcast_to(ax(2)[:, None, :, None], (nb, n, n, n)).ravel()
    z = np.broadcast_to(ax(4)[:, :, None, None], (nb, n, n, n)).ravel()
    X = np.stack([x, y, z], 1)
    if coloc:
        pv = {k: cat(P, k) for k in ('dens', 'press', 'velx', 'vely', 'velz')}
    else:   # --tol: the prim dump's cell at each dediag cell centre (its time differs by <= tol)
        j = Lookup(gp, P['nx1_out_mb'])(X)
        assert (j >= 0).all(), 'prim mesh does not cover the dediag mesh'
        pv = {k: catb(P, k, gp)[j] for k in ('dens', 'press', 'velx', 'vely', 'velz')}
    c = dict(t=t, cyc=bcycle(fd), d=d, g=g, n=n, X=X, dV=np.repeat(h**3, n**3), coloc=coloc, dtp=P['time'] - t,
             rho=pv['dens'], p=pv['press'], u=np.stack([pv['velx'], pv['vely'], pv['velz']], 1),
             br=cat(Dg, 'de_branch').round().astype(int), dken=cat(Dg, 'de_ken_ktr'), ddk=cat(Dg, 'de_dk_resync'))
    return c


def classify(c, rhoc):
    rho, X, dV = c['rho'], c['X'], c['dV']
    xb = xbh(c['d'], c['t']); c['xb'] = xb
    rbh = np.linalg.norm(X - xb, axis=1); c['rbh'] = rbh
    if a.nohole is not None:
        # --nohole RH (c306, the isolated remnant): no puncture, so psi = alpha = 1 (the remnant's own field is ~1e-5), no
        # r_BH cut, and the remnant class a fixed sphere of radius RH about x_c (no Hill radius without a hole)
        psi = np.ones_like(rbh)
        c['alpha'] = np.ones_like(rbh)
        out = np.ones(rbh.shape, bool)
    else:
        psi = 1.0 + 0.5 / np.maximum(rbh, 1e-3)
        c['alpha'] = (1.0 - 0.5 / np.maximum(rbh, 1e-3)) / psi
        out = rbh >= 3.0
    c['sg'] = psi**6
    c['W'] = np.sqrt(1.0 + psi**4 * (c['u']**2).sum(1))
    m = c['sg'] * rho * c['W'] * dV
    c['m'] = m
    d1 = out & (rho >= 0.01 * rhoc)
    if not d1.any():
        d1 = out & (rho >= 0.01 * np.where(out, rho, 0).max())
    xc = (m[d1, None] * X[d1]).sum(0) / m[d1].sum(); MH = m[d1].sum()
    r = np.linalg.norm(xc - xb); dc = np.linalg.norm(X - xc, axis=1)
    for _ in range(40 if a.nohole is None else 0):
        rH = r * (MH / 3.0)**(1 / 3); MHn = m[out & (dc < rH)].sum()
        if abs(MHn - MH) <= 1e-10 * max(MH, 1e-300):
            break
        MH = MHn
    if a.nohole is not None:
        rH = a.nohole; MH = m[out & (dc < rH)].sum()
    c.update(xc=xc, rH=rH, dc=dc, MH=MH)
    sw = rho >= 1e-6 * rhoc
    inh = dc < rH
    return {'core': out & (rho >= 0.1 * rhoc), 'outer': out & (rho >= 0.01 * rhoc) & (rho < 0.1 * rhoc),
            'remnant': out & inh & sw, 'stream': out & ~inh & sw & (rho < 0.01 * rhoc)}


def wq(v, w, qs):
    if len(v) == 0:
        return [np.nan] * len(qs)
    o = np.argsort(v); cw = np.cumsum(w[o]); cw /= cw[-1]
    return [float(v[o][min(np.searchsorted(cw, q), len(v) - 1)]) for q in qs]


rows = {k: [] for k in ('budget', 'quant', 'trans', 'resync')}
prev = None
for t, d, fd, fp in pairs:
    rhoc = par(d, 'rhoc')
    c = load(t, d, fd, fp)
    cls = classify(c, rhoc)
    eps = c['p'] / ((GAM - 1.0) * np.maximum(c['rho'], 1e-300))
    eth = c['sg'] * c['rho'] * eps * c['dV']            # sqrt(gamma) rho eps dV
    H = eth * c['dken']                                  # hidden energy per cell (11, 12)
    br = c['br']
    dtm = np.nan
    if prev is not None and prev['d'] == d and c['cyc'] > prev['cyc']:
        dtm = (t - prev['t']) / (c['cyc'] - prev['cyc'])
    base = dict(t=t, t_tp=t - a.tp, run=d.split('/')[-1], rH=c['rH'], M_H=c['MH'], coloc=int(c['coloc']), dt_prim=c['dtp'])
    for k, s in cls.items():
        tr = s & ((br == 11) | (br == 12))
        row = dict(base, cls=k, M=c['m'][s].sum())
        for b in (0, 11, 12, 13, 14, 15, 16, 17):
            row[f'M_{b}'] = c['m'][s & (br == b)].sum()
        row.update(H_hot=H[tr & (H > 0)].sum(), H_cold=H[tr & (H < 0)].sum(), Eth_tr=eth[tr].sum(),
                   Eth_cls=eth[s].sum(), Eth_13=eth[s & (br == 13)].sum())
        row['H_frac'] = (row['H_hot'] + row['H_cold']) / row['Eth_tr'] if row['Eth_tr'] > 0 else np.nan
        row['H_frac_cls'] = (row['H_hot'] + row['H_cold']) / row['Eth_cls'] if row['Eth_cls'] > 0 else np.nan
        rows['budget'].append(row)
        q = wq(c['dken'][tr], c['m'][tr], (0.05, 0.25, 0.5, 0.75, 0.95))
        rows['quant'].append(dict(base, cls=k, M_tr=c['m'][tr].sum(), q05=q[0], q25=q[1], q50=q[2], q75=q[3], q95=q[4],
                                  f_safeguard=c['m'][s & (br == 12)].sum() / max(c['m'][s].sum(), 1e-300)))
        r14 = s & (br == 14)
        rel = (eth * c['ddk'] / (1.0 + c['ddk']))[r14]
        rows['resync'].append(dict(base, cls=k, M_14=c['m'][r14].sum(), rel_stage_pos=rel[rel > 0].sum(),
                                   rel_stage_neg=rel[rel < 0].sum(), dt_step=dtm,
                                   rate=(rel.sum() * 3.0 / dtm) if np.isfinite(dtm) else np.nan))
    if prev is not None and prev['d'] == d:   # not across a run boundary (frame and mesh change there)
        DT = t - prev['t']
        dxb = c['xb'] - prev['xb']
        P = prev
        sel = P['trsel']
        Xp = P['X'][sel] + (P['alpha'][sel, None] * P['u'][sel] / P['W'][sel, None]) * DT + dxb
        idx = Lookup(c['g'], c['n'])(Xp)
        dest = np.where(idx >= 0, br[np.maximum(idx, 0)], -1)
        for k in P['cls']:
            ck = P['cls'][k][sel]
            for code in TRC:
                fr = ck & (P['br'][sel] == code)
                if not fr.any():
                    continue
                for lab in sorted(set(DEST.values())) + ['stay', 'lost']:
                    if lab == 'stay':
                        mv = fr & np.isin(dest, TRC)
                    elif lab == 'lost':
                        mv = fr & (dest < 0)
                    else:
                        mv = fr & np.isin(dest, [b for b, l in DEST.items() if l == lab])
                    if not mv.any():
                        continue
                    rb = np.digitize(P['dc'][sel][mv] / a.rstar, RBINS) - 1
                    Hm = P['H'][sel][mv]; Mm = P['m'][sel][mv]
                    for ib in range(len(RBINS) - 1):
                        q = rb == ib
                        if not q.any():
                            continue
                        rows['trans'].append(dict(t0=P['t'], t1=t, t_tp=0.5 * (P['t'] + t) - a.tp, cls=k, frm=code,
                                                  to=lab, rbin=f'{RBINS[ib]:g}-{RBINS[ib + 1]:g}', M=Mm[q].sum(),
                                                  H_hot=Hm[q & (Hm > 0)].sum() if code != 13 else np.nan,
                                                  H_cold=Hm[q & (Hm < 0)].sum() if code != 13 else np.nan))
    trsel = np.isin(br, TRC) & np.logical_or.reduce([v for v in cls.values()])
    prev = dict(d=d, t=t, cyc=c['cyc'], X=c['X'], u=c['u'], W=c['W'], alpha=c['alpha'], br=br, H=np.where(br == 13, np.nan, H),
                m=c['m'], dc=c['dc'], xb=c['xb'], cls=cls, trsel=trsel)
    b = [r for r in rows['budget'] if r['t'] == t]
    print(f't-tp {t - a.tp:8.2f} ' + ' | '.join(f"{r['cls']}: M {r['M']:.3g} Mtr {r['M_11'] + r['M_12']:.3g} "
                                                f"H/Eth {r['H_frac']:+.3g}" for r in b), flush=True)
    del c
for k, rs in rows.items():
    if rs:
        keys = []
        for r in rs:
            keys += [x for x in r if x not in keys]
        with open(f'{a.out}_{k}.csv', 'w') as f:
            w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rs)
