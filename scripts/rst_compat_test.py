#!/usr/bin/env python3
"""R-140 item 4 (D6): the standing restart forward-compatibility test.

Continue one checkpoint with an old binary and with a new binary (the new features off) and
compare everything the two continuations write:
  - every history file, byte for byte;
  - every bin dump and every restart they write, byte for byte after the parameter copy
    (the first '<par_end>' marker): the copies legitimately differ (the new binary writes
    <job> restart_format_version and restart_state_layout, outputs/restart_state.hpp), the
    data and the fixed header must not.
PASS iff every file exists in both and matches. The checkpoint is normally a base8-era
restart (format version 0, no layout) and OLD the binary that wrote it.

    rst_compat_test.py --old OLD_BIN --new NEW_BIN --rst FILE.rst --nlim N --work DIR
                       [--launcher "mpiexec -n 12 ..."] [--keep] [-- extra athena args]
--nlim is the absolute cycle to stop at (AthenaK's nlim is absolute, not "N more").
Writes DIR/old, DIR/new (the runs) and DIR/rst_compat.csv; exit code 0 iff PASS.
"""
import argparse
import os
import shlex
import subprocess
import sys


def run(binary, rst, nlim, d, launcher, extra):
    os.makedirs(d, exist_ok=True)
    cmd = shlex.split(launcher) + [binary, '-r', os.path.abspath(rst), '-d', d,
                                   f'time/nlim={nlim}'] + extra
    with open(os.path.join(d, 'out.log'), 'w') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=d)
    return rc


def body(path, after_par):
    b = open(path, 'rb').read()
    if not after_par:
        return b
    i = b.find(b'<par_end>')
    return b[i:] if i >= 0 else b


def files(d):
    out = []
    for root, _, fs in os.walk(d):
        for f in fs:
            p = os.path.relpath(os.path.join(root, f), d)
            if p.endswith(('.hst', '.bin', '.rst')):
                out.append(p)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--rst', required=True)
    ap.add_argument('--nlim', type=int, required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--launcher', default='')
    ap.add_argument('extra', nargs='*')
    a = ap.parse_args()
    w = os.path.abspath(a.work)
    rc = {k: run(b, a.rst, a.nlim, os.path.join(w, k), a.launcher, a.extra)
          for k, b in (('old', a.old), ('new', a.new))}
    fo, fn = files(os.path.join(w, 'old')), files(os.path.join(w, 'new'))
    rows, ok = ['file,in_old,in_new,match'], rc['old'] == 0 and rc['new'] == 0
    for p in sorted(set(fo) | set(fn)):
        io, inn = p in fo, p in fn
        m = io and inn and body(os.path.join(w, 'old', p), not p.endswith('.hst')) == \
            body(os.path.join(w, 'new', p), not p.endswith('.hst'))
        ok = ok and m
        rows.append(f'{p},{int(io)},{int(inn)},{int(m)}')
    # the new binary must have read the checkpoint's state (no init-C2P fallback)
    nl = open(os.path.join(w, 'new', 'out.log'), errors='replace').read()
    fallback = 'WARNING restart_state' in nl
    ok = ok and not fallback and len(fo) > 0
    rows.append(f'# rc_old={rc["old"]} rc_new={rc["new"]} files={len(fo)}/{len(fn)} '
                f'state_fallback={int(fallback)} {"PASS" if ok else "FAIL"}')
    open(os.path.join(w, 'rst_compat.csv'), 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(r for r in rows if r.startswith('#') or not r.endswith(',1')))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
