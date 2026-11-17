#!/usr/bin/env python3
"""c267 (R-260): bitwise comparison of two arms' final restarts and histories. Per MeshBlock (the same tree is required) the
evolved data region of each record: the MHD conserved [5 + nscalars], the face fields and the 17 ADM arrays (tde_dmde.py's
layout); the restart_state block that follows differs by construction when one arm has no flag slot, so it is compared
only when both records have the same size. Histories (*.hst): row by row, as text.
    c267_cmp.py DIR_A DIR_B [--nscal 1]"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tde_dmde as T   # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('--nscal', type=int, default=1)
o = ap.parse_args()
ra = sorted(glob.glob(f'{o.a}/rst/*.rst'))[-1]; rb = sorted(glob.glob(f'{o.b}/rst/*.rst'))[-1]
ha, hb = T.header(ra), T.header(rb)
print(f'{ra}\n{rb}\n t {ha["time"]!r} {hb["time"]!r}; nmb {ha["nmb"]} {hb["nmb"]}; record {ha["dsize"]} {hb["dsize"]}')
same_tree = ha['nmb'] == hb['nmb'] and np.array_equal(ha['ll'], hb['ll'])
print(' tree identical:', same_tree)
if same_tree:
    no = 24; ncc = no**3; nm = 5 + o.nscal
    nev = (nm * ncc + (no + 1) * no * no * 3 + 17 * ncc) * 8
    ma = np.memmap(ra, np.uint8, 'r', offset=ha['data_off'], shape=(ha['nmb'], ha['dsize']))
    mb = np.memmap(rb, np.uint8, 'r', offset=hb['data_off'], shape=(hb['nmb'], hb['dsize']))
    bad = [b for b in range(ha['nmb']) if not np.array_equal(ma[b, :nev], mb[b, :nev])]
    print(f' evolved data (cons + B + ADM): {ha["nmb"] - len(bad)}/{ha["nmb"]} blocks bitwise' +
          (f'; first differing blocks {bad[:5]}' if bad else ''))
    if ha['dsize'] == hb['dsize']:
        bad2 = [b for b in range(ha['nmb']) if not np.array_equal(ma[b, nev:], mb[b, nev:])]
        print(f' restart_state: {ha["nmb"] - len(bad2)}/{ha["nmb"]} blocks bitwise')
for f in sorted(glob.glob(f'{o.a}/*.hst')):
    g = f'{o.b}/{os.path.basename(f)}'
    if not os.path.exists(g):
        continue
    la = [l for l in open(f) if not l.startswith('#')]; lb = [l for l in open(g) if not l.startswith('#')]
    n = min(len(la), len(lb)); d = [i for i in range(n) if la[i] != lb[i]]
    print(f' {os.path.basename(f)}: {n} rows, {len(d)} differ' + (f' (first {d[0]})' if d else '') +
          (f'; lengths {len(la)} vs {len(lb)}' if len(la) != len(lb) else ''))
