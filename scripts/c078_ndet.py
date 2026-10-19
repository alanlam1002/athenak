#!/usr/bin/env python3
"""R-078 item 7: compare redlog Allreduce logs of two identical runs (a, b).

  c078_ndet.py <run_a dir> <run_b dir> <athena binary> [rank ...]

For each logged rank: the first call whose local input or result differs; per call site
(symbolised), how many calls differ in input (the rank's own GPU-side reduction) vs only in the
result (another rank or the MPI reduction); and the relative difference |a-b|/|a| of each site's
first value over the call sequence, so the growth is visible. Also the MLEDGER M_end and dt per
cycle from out.log.
"""
import collections
import re
import struct
import subprocess
import sys


def parse(path):
    out = []
    for ln in open(path):
        head, i, o = ln.rstrip('\n').split(' | ')
        seq, addr, sz, cnt = head.split()
        out.append((int(seq), addr, int(sz), int(cnt), i[3:].split(), o[4:].split()))
    return out


def val(h, sz):
    if sz == 8:
        return struct.unpack('<d', struct.pack('<Q', int(h, 16)))[0]
    return struct.unpack('<i', struct.pack('<I', int(h, 16)))[0]


def sym(binary, addrs):
    if not addrs:
        return {}
    r = subprocess.run(['addr2line', '-f', '-C', '-e', binary] + list(addrs),
                       stdout=subprocess.PIPE, universal_newlines=True).stdout.split('\n')
    return {a: r[2 * k].split('(')[0][:60] for k, a in enumerate(addrs)}


def ledger(path):
    rows = {}
    for ln in open(path):
        m = re.match(r'MLEDGER cycle=(\d+) time=\S+ dt=(\S+) Mu1=\S+ Mend=(\S+)', ln)
        if m:
            rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return rows


def main():
    a, b, binary = sys.argv[1:4]
    ranks = sys.argv[4:] or ['0', '7']
    for r in ranks:
        A, B = parse(f'{a}/redlog.{r}.txt'), parse(f'{b}/redlog.{r}.txt')
        n = min(len(A), len(B))
        print(f'== rank {r}: {len(A)} / {len(B)} calls')
        first = None
        site = collections.OrderedDict()
        for k in range(n):
            ca, cb = A[k], B[k]
            if ca[1] != cb[1]:
                print(f'  call sequence diverges at {k}: {ca[1]} vs {cb[1]}')
                break
            din, dout = ca[4] != cb[4], ca[5] != cb[5]
            s = site.setdefault(ca[1], {'n': 0, 'in': 0, 'outonly': 0, 'first': None, 'rel': []})
            s['n'] += 1
            if din:
                s['in'] += 1
            elif dout:
                s['outonly'] += 1
            if (din or dout) and first is None:
                first = k
            if (din or dout) and s['first'] is None:
                s['first'] = k
            if ca[5]:
                va, vb = val(ca[5][0], ca[2]), val(cb[5][0], cb[2])
                s['rel'].append((k, abs(va - vb) / abs(va) if va != 0 else abs(vb)))
        names = sym(binary, site.keys())
        print(f'  first differing call: {first}')
        for ad, s in site.items():
            rel = [x for _, x in s['rel']]
            nz = [x for x in rel if x > 0]
            print(f"  {names.get(ad, ad):60s} calls {s['n']:6d}  input-differs {s['in']:6d}  "
                  f"result-only {s['outonly']:6d}  first {s['first']}  "
                  f"rel max {max(rel) if rel else 0:.2e}  "
                  f"rel first/last nonzero {nz[0] if nz else 0:.2e} / {nz[-1] if nz else 0:.2e}")
    la, lb = ledger(f'{a}/out.log'), ledger(f'{b}/out.log')
    print('== per cycle: rel. diff of dt and of M_end (MLEDGER, 10 digits)')
    for c in sorted(set(la) & set(lb)):
        (da, ma), (db, mb) = la[c], lb[c]
        print(f'  cycle {c}: dt {abs(da - db) / da:.2e}  Mend {abs(ma - mb) / ma:.2e}')


if __name__ == '__main__':
    main()
