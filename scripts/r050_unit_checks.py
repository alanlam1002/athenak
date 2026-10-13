#!/usr/bin/env python3
"""R-050 unit tests of the tde_debris criterion: analysis of the debug-queue runs
(athenak_run/cfc/r050_tests/gpu_*), see research/notes/NEXT_TASKS_R050.md.

  U2  isolated star: blocks per level at the end; blocks outside the star (max rho <=
      rho_F) above the root, per level; flags per cycle from the tde_debris diag lines.
  U3  static Gaussian stream: at r_perp = FWHM/2 (|x| < L/2 - 0.5), the realised
      ell/dx with ell = rho/|grad rho| measured on the final 3D dump (and the analytic
      H^2/r_perp), against N_H; blocks with max rho <= rho_F above the root; L5 block
      count against the geometric estimate.
  U4  advected stream: refined / de-refined blocks per cycle, hysteresis on vs off.
  U5  floor-only field: blocks requested beyond the BH box.
  U6  controller: xidot history (user.hst frame_vel), max |d xidot| per step, the ramp,
      and the restart (u6e vs u6b, cycles 60-120).

    r050_unit_checks.py [--dir DIR]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tde_ensemble_plots import bc                 # noqa: E402

RHOC = 3.24e-4
RHO_F = 1e-6*RHOC


def diag_lines(d):
    out = []
    for l in open(f'{d}/out.log'):
        if l.startswith('tde_debris'):
            m = re.search(r'cycle=(\d+).*blocks/level:([\d ]+)\|.*flags -1/0/\+1: (\d+)/(\d+)/(\d+)'
                          r' \| projected/level:([\d ]+)\|', l)
            if m:
                out.append((int(m.group(1)), [int(x) for x in m.group(2).split()],
                            (int(m.group(3)), int(m.group(4)), int(m.group(5))),
                            [int(x) for x in m.group(6).split()], l.strip()))
    return out


def last_dump(d):
    fs = sorted(glob.glob(f'{d}/bin/*prim3d*.bin'))
    return bc.read_binary(fs[-1]) if fs else None


def blocks(q):
    g = np.asarray(q['mb_geometry']); nx = q['nx1_out_mb']
    rho = np.asarray(q['mb_data']['dens'])
    dx = (g[:, 1] - g[:, 0])/nx
    lev = np.rint(np.log2(1.0/dx)).astype(int)
    return g, nx, rho, dx, lev


def u2(d, lab):
    q = last_dump(d)
    if q is None:
        print(f'U2 {lab}: no dump'); return
    g, nx, rho, dx, lev = blocks(q)
    bmax = rho.reshape(len(g), -1).max(1)
    print(f'U2 {lab}: t = {q["time"]:.3f}, blocks/level {np.bincount(lev).tolist()}; blocks with '
          f'max rho <= rho_F per level {np.bincount(lev[bmax <= RHO_F], minlength=lev.max()+1).tolist()}; '
          f'blocks holding rho > 1e-2 rho_c per level {np.bincount(lev[bmax > 1e-2*RHOC], minlength=lev.max()+1).tolist()}')
    D = diag_lines(d)
    fl = np.array([x[2] for x in D])
    late = fl[-20:]
    print(f'   flags (-1/0/+1) over the last {len(late)} diag cycles: de-refine {late[:, 0].sum()}, '
          f'refine {late[:, 2].sum()}; per cycle: ' + ' '.join(f'{a}/{c}' for a, _, c in late))


def u3(d, lab, h=0.35, half=4.0, nh=8.0):
    q = last_dump(d)
    if q is None:
        print(f'U3 {lab}: no dump'); return
    g, nx, rho, dx, lev = blocks(q)
    rho3 = rho  # (mb, k, j, i)
    rs, ratio_num, ratio_an = [], [], []
    rfw = h*np.sqrt(2*np.log(2))
    for m in range(len(g)):
        h_ = dx[m]
        x = g[m, 0] + (np.arange(nx) + .5)*h_
        y = g[m, 2] + (np.arange(nx) + .5)*h_
        z = g[m, 4] + (np.arange(nx) + .5)*h_
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        rp = np.hypot(Y, Z)
        sel = (np.abs(rp - rfw) < 0.5*h_) & (np.abs(X) < half - 0.5)
        sel[[0, -1], :, :] = False; sel[:, [0, -1], :] = False; sel[:, :, [0, -1]] = False
        if not sel.any():
            continue
        r = rho3[m]
        gx = (np.roll(r, -1, 2) - np.roll(r, 1, 2))/(2*h_)
        gy = (np.roll(r, -1, 1) - np.roll(r, 1, 1))/(2*h_)
        gz = (np.roll(r, -1, 0) - np.roll(r, 1, 0))/(2*h_)
        ell = r/np.sqrt(gx**2 + gy**2 + gz**2)
        ratio_num += list((ell/h_)[sel]); ratio_an += list((h*h/rp/h_)[sel]); rs += [lev[m]]*sel.sum()
    bmax = rho.reshape(len(g), -1).max(1)
    rn, ra = np.array(ratio_num), np.array(ratio_an)
    print(f'U3 {lab}: t = {q["time"]:.3f}, blocks/level {np.bincount(lev).tolist()}')
    if rn.size:
        print(f'   r_perp = FWHM/2 = {rfw:.3f}: {rn.size} cells, levels {sorted(set(rs))}; ell/dx measured '
              f'min {rn.min():.2f} median {np.median(rn):.2f}; analytic H^2/r/dx min {ra.min():.2f} '
              f'(N_H = {nh:g}; pass if >= N_H or at the level cap)')
    else:
        print('   no cells at r_perp = FWHM/2 inside the gate')
    print(f'   blocks with max rho <= rho_F above the root, per level: '
          f'{np.bincount(lev[(bmax <= RHO_F) & (lev > 0)], minlength=lev.max()+1).tolist()}')
    D = diag_lines(d)
    if D:
        print(f'   last diag: {D[-1][4][:260]}')


def u4(d, lab):
    D = diag_lines(d)
    if len(D) < 7:
        print(f'U4 {lab}: {len(D)} diag cycles only'); return
    fl = np.array([x[2] for x in D])[5:]           # skip the initial build-up
    print(f'U4 {lab}: cycles {len(fl)}; refined blocks/cycle mean {fl[:, 2].mean():.2f} max {fl[:, 2].max()}; '
          f'de-refined mean {fl[:, 0].mean():.2f} max {fl[:, 0].max()}; total +{fl[:, 2].sum()} / -{fl[:, 0].sum()}; '
          f'blocks at end {sum(D[-1][1])}')


def hst_user(d):
    f = glob.glob(f'{d}/*.user.hst')[0]
    hdr = open(f).readlines()[1]
    cols = {m.group(2): int(m.group(1)) - 1 for m in re.finditer(r'\[(\d+)\]=(\S+)', hdr)}
    return np.loadtxt(f), cols


def u6(dirs):
    xs = {}
    for lab, d in dirs.items():
        if not os.path.isdir(d):
            continue
        h, c = hst_user(d)
        t = h[:, 0]; xd = h[:, c['frame_vel_x1']]
        dxd = np.abs(np.diff(xd))
        xs[lab] = (t, xd, h)
        print(f'U6 {lab}: t {t[0]:.3f}-{t[-1]:.3f}, xidot_x {xd[0]:+.5f} -> {xd[-1]:+.5f}; max |d xidot| per '
              f'output step {dxd.max():.2e}; fc_mass {h[0, c["fc_mass"]]:.3e} -> {h[-1, c["fc_mass"]]:.3e}')
        sw = [l.strip() for l in open(f'{d}/out.log') if 'switch to' in l]
        for l in sw[:2]:
            print('   ', l)
    if 'u6b' in xs and 'u6e' in xs:
        tb, xb, hb = xs['u6b']; te, xe, he = xs['u6e']
        common = np.intersect1d(np.round(tb, 10), np.round(te, 10))
        ib = np.isin(np.round(tb, 10), common); ie = np.isin(np.round(te, 10), common)
        same = np.array_equal(hb[ib], he[ie])
        print(f'U6 restart: u6e (restart at cycle 60) vs u6b over {common.size} common history rows: '
              f'{"bitwise identical" if same else "max |diff| %.2e" % np.abs(hb[ib] - he[ie]).max()}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/r050_tests')
    a = ap.parse_args()
    T = a.dir
    for lab in ('u2a_core4', 'u2b_core5'):
        if os.path.isdir(f'{T}/gpu_{lab}'):
            u2(f'{T}/gpu_{lab}', lab)
    for lab in ('u3a_stream1e-4', 'u3b_stream1e-6'):
        if os.path.isdir(f'{T}/gpu_{lab}'):
            u3(f'{T}/gpu_{lab}', lab)
    for lab in ('u4_hyst_on', 'u4_hyst_off'):
        if os.path.isdir(f'{T}/gpu_{lab}'):
            u4(f'{T}/gpu_{lab}', lab)
    if os.path.isdir(f'{T}/gpu_u5_floor'):
        D = diag_lines(f'{T}/gpu_u5_floor')
        print(f'U5: {len(D)} diag cycles; last: {D[-1][4][:300] if D else "none"}')
    u6({'u6a': f'{T}/gpu_u6a_remnant', 'u6b': f'{T}/gpu_u6b_switch_bound', 'u6c': f'{T}/gpu_u6c_ramp',
        'u6e': f'{T}/gpu_u6e_restart60'})


if __name__ == '__main__':
    main()
