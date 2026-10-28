#!/usr/bin/env python3
"""c134 stage checkpoints C0-C4 (R-145 item 5; research STAGED_RUN_PLAN §1): preserve the stage-end restart
plus the output set and write a manifest (binary hash, deck, stage, t, cycle, the transition criterion's values).
The restart and the bin outputs are hard links (no extra quota; rst_keep.py may then delete RUN/rst copies), the
history files, deck and logs are copied (they keep growing).
    c134_checkpoint.py RUN STAGE TP BINARY [--crit NAME=VALUE ...]   (STAGE: C0..C4 or a hold point S<k>_tp<dt>)
RUN is a pilot directory (parfile.par, rst/, bin/, *.hst, out.log); STAGE is C0..C4; TP the geodesic periapsis
time. The checkpoint goes to RUN/checkpoints/STAGE; an existing one is never overwritten.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # this tree's tde_dmde (header fix)
import tde_dmde as D   # noqa: E402

B = 'cfc_tde_wd_imbh_parabolic_tracker'


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def hst_row(fn, t):
    """the history row at time t (the last row with time <= t), as {name: value}"""
    hdr = [l for l in open(fn) if l.startswith('#')][-1]
    names = [x.split('=')[-1] for x in hdr.replace('#', '').split()]
    a = np.loadtxt(fn)
    a = a[a[:, 0] <= t + 1e-9]
    return {n: float(v) for n, v in zip(names, a[-1])}, {n: float(v) for n, v in zip(names, a[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('stage', help='C0..C4, or a hold point such as S3_tp600 (an arm stopped before its stage criterion)')
    ap.add_argument('tp', type=float)
    ap.add_argument('binary')
    ap.add_argument('--crit', action='append', default=[], help='NAME=VALUE: a transition-criterion value')
    a = ap.parse_args()
    if not re.fullmatch(r'C[0-4]|S[0-9]_tp[0-9]+', a.stage):
        sys.exit(f'stage {a.stage}: expected C0..C4 or S<k>_tp<dt>')
    run = os.path.abspath(a.run)
    out = f'{run}/checkpoints/{a.stage}'
    if os.path.exists(f'{out}/MANIFEST.json'):
        sys.exit(f'{out} exists')
    resume = os.path.exists(out)          # an interrupted checkpoint: link/copy what is missing, then the manifest
    rst = sorted(glob.glob(f'{run}/rst/*.rst'))[-1]
    h = D.header(rst)
    t, cyc = float(h['time']), int(h['ncycle'])
    os.makedirs(f'{out}/bin', exist_ok=resume)
    if not os.path.exists(f'{out}/{os.path.basename(rst)}'):
        os.link(rst, f'{out}/{os.path.basename(rst)}')
    nb = 0
    for f in sorted(glob.glob(f'{run}/bin/*.bin')):
        if not os.path.exists(f'{out}/bin/{os.path.basename(f)}'):
            os.link(f, f'{out}/bin/{os.path.basename(f)}')
        nb += 1
    for f in glob.glob(f'{run}/*.hst') + [f'{run}/parfile.par', f'{run}/out.log', f'{run}/err.log']:
        shutil.copy2(f, out)
    u, u0 = hst_row(f'{run}/{B}.user.hst', t)
    m, m0 = hst_row(f'{run}/{B}.mhd.hst', t)
    crec = None
    for l in open(f'{run}/out.log', errors='replace'):
        g = re.match(rf'CRECOV cycle={cyc - 1} .*Mbound_all=(\S+)', l) or re.match(rf'CRECOV cycle={cyc} .*Mbound_all=(\S+)', l)
        if g:
            crec = float(g.group(1))
    deck = f'{run}/parfile.par'
    man = {
        'stage': a.stage, 'run': run, 'written': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'binary': os.path.abspath(a.binary), 'binary_sha256': sha256(a.binary),
        'deck': deck, 'deck_sha256': sha256(deck),
        'restart': os.path.basename(rst), 'restart_sha256': sha256(rst),
        't': t, 'cycle': cyc, 't_p': a.tp, 't_minus_tp': t - a.tp, 'n_bin_outputs': nb,
        'criterion': dict((kv.split('=', 1) if '=' in kv else (f'note{i}', kv)) for i, kv in enumerate(a.crit)),
        'values (measured; hst rows at t)': {
            'rho-max': u['rho-max'], 'rho-max/rho-max(0)': u['rho-max'] / u0['rho-max'],
            'alpha-min': u['alpha-min'], 'fc_mass': u.get('fc_mass'),
            'fc_mass/fc_mass(0)': u['fc_mass'] / u0['fc_mass'] if 'fc_mass' in u else None,
            'frame_disp': [u.get(f'frame_disp_x{k}') for k in (1, 2, 3)],
            'grid mass': m['mass'], 'grid mass/(0)': m['mass'] / m0['mass'],
            'tot-E/(0) - 1': m['tot-E'] / m0['tot-E'] - 1.0,
            'scal-0/(0) - 1': m['scal-0'] / m0['scal-0'] - 1.0 if 'scal-0' in m else None,
            'CRECOV Mbound_all': crec},
    }
    json.dump(man, open(f'{out}/MANIFEST.json', 'w'), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
