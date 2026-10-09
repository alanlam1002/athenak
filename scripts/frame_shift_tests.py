#!/usr/bin/env python3
"""Comoving-gauge unit tests T0a-T0c (src/pgen/tests/dyngr_frame_shift.cpp,
src/cfc/DEVELOPMENT.md item 68). Each test's final state is compared with its exact
solution, which for these choices is the initial state:
  T0a  uniform medium, xidot != 0: final == initial (round-off);
  T0b  entropy wave, v1 = 0.25, t = 8: advected by exactly 2 periods (xidot = 0) or
       static (xidot = v1), so final == initial in both gauges;
  T0c  field loop, v = (0.2, 0.1), t = 10: moved by (2, 1) periods (xidot = 0) or
       static (xidot = v), so final == initial; plus max |div B| and magnetic energy.

    frame_shift_tests.py RUNDIR
"""
import os
import sys
import types

import numpy as np

sys.modules.setdefault('h5py', types.ModuleType('h5py'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vis', 'python'))
import bin_convert as bc   # noqa: E402


def load(d, var, n):
    p = bc.read_binary(f'{d}/bin/gauge.{var}.{n:05d}.bin')
    return p['time'], {k: np.asarray(v)[:, 0] for k, v in p['mb_data'].items()}


def main(root):
    out = []
    t0, a = load(f'{root}/T0a_uniform', 'mhd_w_bcc', 0)
    t1, b = load(f'{root}/T0a_uniform', 'mhd_w_bcc', 1)
    dev = {k: float(np.max(np.abs(b[k] - a[k].mean()))) for k in ('dens', 'velx', 'vely', 'press')}
    out.append(f'T0a uniform, xidot=(0.25,-0.1), v=(0.3,0.2), t={t1:g}: max |q - q0| = ' +
               ', '.join(f'{k} {v:.2e}' for k, v in dev.items()))
    for lab in ('xd0', 'xdu'):
        d = f'{root}/T0b_wave_{lab}'
        _, a = load(d, 'mhd_w_bcc', 0)
        t1, b = load(d, 'mhd_w_bcc', 1)
        e1 = np.mean(np.abs(b['dens'] - a['dens']))
        amp0 = 0.5*(a['dens'].max() - a['dens'].min())
        amp1 = 0.5*(b['dens'].max() - b['dens'].min())
        dp = np.max(np.abs(b['press']/a['press'].mean() - 1))
        out.append(f'T0b entropy wave {lab} (xidot1 = {"0" if lab == "xd0" else "v1 = 0.25"}), '
                   f't={t1:g}: L1(rho - exact) = {e1:.3e}, amplitude {amp1/amp0:.5f} of '
                   f'initial, max |P/P0 - 1| = {dp:.2e}')
    for lab in ('xd0', 'xdu'):
        d = f'{root}/T0c_loop_{lab}'
        _, a = load(d, 'mhd_w_bcc', 0)
        t1, b = load(d, 'mhd_w_bcc', 1)
        _, dv0 = load(d, 'mhd_divb', 0)
        _, dv1 = load(d, 'mhd_divb', 1)
        b2a = a['bcc1']**2 + a['bcc2']**2
        b2b = b['bcc1']**2 + b['bcc2']**2
        e1 = np.mean(np.abs(np.sqrt(b2b) - np.sqrt(b2a)))/np.mean(np.sqrt(b2a))
        bmax0 = np.sqrt(b2a).max()
        out.append(f'T0c field loop {lab} (xidot = {"0" if lab == "xd0" else "v"}), t={t1:g}: '
                   f'magnetic energy {b2b.sum()/b2a.sum():.5f} of initial, '
                   f'L1(|B| - exact)/<|B|> = {e1:.3e}, '
                   f'max|divB|/(|B|max/dx) = {np.abs(dv1["divb"]).max()*(1/64)/bmax0:.2e} '
                   f'(t=0: {np.abs(dv0["divb"]).max()*(1/64)/bmax0:.2e})')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1])
