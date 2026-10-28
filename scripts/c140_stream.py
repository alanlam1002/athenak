#!/usr/bin/env python3
"""c140 (R-150 item 5, user-approved 2026-10-28): the tde_debris full-run acceptance, stream geometry and cost from
the 3D primitive dumps (prim_3d, every 12 M) of two arms at equal t (here P10_T3 vs P10_r82).
Per dump: the gas with rho > thr rho_c (default 1e-8, T3's floor) outside r_BH < 3 (x_BH from the dump's parameter
copy, tde_dmde.frame_state), binned by azimuth phi about the hole in the orbital (x, y) plane (dphi = 2 deg); per bin
the mass, the in-plane radial and the vertical mass-weighted spreads (Gaussian FWHM = 2.3548 sigma), the
mass-weighted cell size dx and N_perp = FWHM_z/dx (the stream's thin direction). The bins within +-10 deg of the
density maximum (the remnant) are left out of the stream statistics. Reported: the stream mass, its mass-weighted
median FWHM_z and FWHM_R, the stream-mass fraction with N_perp >= N_w (4), rho_max, and the blocks per level.
    c140_stream.py OUT.csv ARMDIR [ARMDIR ...] [--thr 1e-8] [--nw 4]
An argued-adequate measure for an A/B comparison of refinement criteria (the same method on both arms); not the
local runs' principal-axis width (c132_local.py), which assumes a straight stream."""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_ensemble_plots import bc                 # noqa: E402
import tde_dmde as T                              # noqa: E402

RHOC = 3.24e-4


def par_text(f):
    h = open(f, 'rb').read(1 << 20)
    return h[:h.index(b'<par_end>')].decode(errors='ignore')


def wmed(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(np.interp(0.5 * c[-1], c, x[o]))


def dump(f, thr, nw):
    P = bc.read_binary(f)
    t = float(P['time'])
    xbh, _ = T.frame_state(par_text(f), t)
    g = np.asarray(P['mb_geometry'])
    n = P['nx1_out_mb']
    q = P['mb_data']
    X, Y, Z, R, D, L = [], [], [], [], [], []
    lev = {}
    for b in range(len(g)):
        dx = (g[b, 1] - g[b, 0]) / n
        lev[dx] = lev.get(dx, 0) + 1
        c = [g[b, 2 * a] + (np.arange(n) + 0.5) * dx for a in range(3)]
        z, y, x = np.meshgrid(c[2], c[1], c[0], indexing='ij')
        rho = np.asarray(q['dens'][b])
        m = (rho > thr * RHOC) & (np.sqrt((x - xbh[0])**2 + (y - xbh[1])**2 + (z - xbh[2])**2) >= 3.0)
        if not m.any():
            continue
        X.append(x[m]); Y.append(y[m]); Z.append(z[m]); R.append(rho[m]); D.append(np.full(m.sum(), dx))
    x, y, z, rho, dx = (np.concatenate(v) for v in (X, Y, Z, R, D))
    dm = rho * dx**3
    ph = np.degrees(np.arctan2(y - xbh[1], x - xbh[0]))
    rr = np.hypot(x - xbh[0], y - xbh[1])
    i0 = np.argmax(rho)
    near = np.abs((ph - ph[i0] + 180.0) % 360.0 - 180.0) < 10.0
    out = dict(t=t, rho_max=float(rho.max()) / RHOC, M_gas=float(dm.sum()))
    edges = np.arange(-180.0, 180.0 + 1e-9, 2.0)
    k = np.digitize(ph, edges) - 1
    fz, fr, ms, np_ok = [], [], [], 0.0
    for b in range(len(edges) - 1):
        s = (k == b) & ~near
        if dm[s].sum() <= 0:
            continue
        w = dm[s]
        zc = np.average(z[s], weights=w)
        rc = np.average(rr[s], weights=w)
        Fz = 2.3548 * np.sqrt(np.average((z[s] - zc)**2, weights=w))
        Fr = 2.3548 * np.sqrt(np.average((rr[s] - rc)**2, weights=w))
        dxb = np.average(dx[s], weights=w)
        fz.append(Fz); fr.append(Fr); ms.append(w.sum())
        if Fz / dxb >= nw:
            np_ok += w.sum()
    ms = np.array(ms)
    out.update(M_stream=float(ms.sum()), FWHM_z_med=wmed(np.array(fz), ms), FWHM_R_med=wmed(np.array(fr), ms),
               frac_Nperp_ok=float(np_ok / ms.sum()))
    for dxv, cnt in sorted(lev.items(), reverse=True):
        out[f'blocks_dx{dxv:g}'] = cnt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('arms', nargs='+')
    ap.add_argument('--thr', type=float, default=1e-8)
    ap.add_argument('--nw', type=float, default=4.0)
    a = ap.parse_args()
    rows = []
    for d in a.arms:
        for f in sorted(glob.glob(f'{d}/bin/*.prim_3d.*.bin')):
            o = dump(f, a.thr, a.nw)
            o['arm'] = os.path.basename(d.rstrip('/'))
            rows.append(o)
            print(o, flush=True)
    keys = ['arm', 't', 'rho_max', 'M_gas', 'M_stream', 'FWHM_z_med', 'FWHM_R_med', 'frac_Nperp_ok']
    keys += sorted({k for r in rows for k in r if k.startswith('blocks_')}, key=lambda s: -float(s[9:]))
    with open(a.out, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(str(r.get(k, 0)) for k in keys) + '\n')


if __name__ == '__main__':
    main()
