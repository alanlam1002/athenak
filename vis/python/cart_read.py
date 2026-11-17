#!/usr/bin/env python3
"""Reader for AthenaK `file_type = cart` outputs (cart/<base>.<id>.<nnnnn>.bin), including the K-013 tilted grids.

File layout (src/outputs/cartgrid.cpp WriteOutputFile): the raw MetaData struct (outputs.hpp:403; 52 bytes on x86-64:
int cycle, float time, float center[3], float extent[3], int numpoints[3], bool is_cheb, 3 bytes padding, int noutvars),
int len + the space-separated variable labels, then per variable float32 data written k (slowest), j, i (fastest).
A tilted or nearest-cell grid also has cart/<base>.<id>.plane.txt with the full-precision geometry (e1, e2, e3).

    d = read_cart(path)  ->  dict(cycle, time, center, extent, numpoints, is_cheb, vars={label: array[nz, ny, nx]},
                                  e1, e2, e3, interp, u, v, w (1D grid coordinates along e1, e2, e3))
Points are center + u[i] e1 + v[j] e2 + w[k] e3 (untilted: e = the coordinate axes).
"""
import os
import re
import struct

import numpy as np

_MD = struct.Struct('<if3f3f3i?3xi')


def read_plane(path):
    """the sidecar of a cart file (None if the grid is axis-aligned with Lagrange interpolation)."""
    m = re.match(r'(.*)\.\d{5}\.bin$', path)
    f = f'{m.group(1)}.plane.txt' if m else None
    if not f or not os.path.exists(f):
        return None
    out = {}
    for ln in open(f):
        if ln.startswith('#') or not ln.strip():
            continue
        k, *v = ln.split()
        out[k] = v[0] if k == 'interp' else np.array([float(x) for x in v])
    return out


def read_cart(path):
    with open(path, 'rb') as fh:
        b = fh.read()
    cyc, t, cx, cy, cz, ex, ey, ez, nx, ny, nz, cheb, nv = _MD.unpack_from(b, 0)
    o = _MD.size
    (ln,) = struct.unpack_from('<i', b, o); o += 4
    labels = b[o:o + ln].decode().split(); o += ln
    n = nx * ny * nz
    data = np.frombuffer(b, '<f4', count=nv * n, offset=o).reshape(nv, nz, ny, nx)
    d = dict(cycle=cyc, time=t, center=np.array([cx, cy, cz]), extent=np.array([ex, ey, ez]),
             numpoints=(nx, ny, nz), is_cheb=bool(cheb), vars=dict(zip(labels, data)),
             e1=np.array([1.0, 0, 0]), e2=np.array([0, 1.0, 0]), e3=np.array([0, 0, 1.0]), interp='lagrange')
    p = read_plane(path)
    if p:
        d.update(center=p['center'], extent=p['extent'], e1=p['e1'], e2=p['e2'], e3=p['e3'], interp=p['interp'])
    ext = d['extent']

    def axis(e, m):
        if m == 1:
            return np.zeros(1)
        if d['is_cheb']:
            return e * np.cos(np.arange(m) * np.pi / (m - 1))
        return -e + np.arange(m) * (2 * e / (m - 1))
    d['u'], d['v'], d['w'] = axis(ext[0], nx), axis(ext[1], ny), axis(ext[2], nz)
    return d


if __name__ == '__main__':
    import sys
    for f in sys.argv[1:]:
        d = read_cart(f)
        print(f, 'cycle', d['cycle'], 't', d['time'], 'n', d['numpoints'], 'e1', d['e1'], 'e2', d['e2'], d['interp'],
              {k: (float(np.nanmin(v)), float(np.nanmax(v))) for k, v in d['vars'].items()})
