#!/usr/bin/env python3
"""R-078 item 6: compare two AthenaK restart files byte for byte, header excluded.

The header is the parameter dump up to and including '<par_end>\n'; it differs between two
restart blocks written at the same cycle (each stores its own file_number, last_time, tag).
Everything after it -- mesh header (nmb_total, sizes, time, dt, ncycle), the logical
locations, costs and all data arrays -- must be identical.

  c078_rstcmp.py A.rst B.rst
"""
import sys
import numpy as np

MARK = b"<par_end>\n"


def body(path):
    with open(path, "rb") as f:
        head = f.read(1 << 22)
        p = head.find(MARK)
        if p < 0:
            sys.exit(f"{path}: no {MARK!r} in the first 4 MiB")
        f.seek(p + len(MARK))
        return p + len(MARK), np.fromfile(f, dtype=np.uint8)


def main():
    a, b = sys.argv[1], sys.argv[2]
    ha, da = body(a)
    hb, db = body(b)
    print(f"A {a}: header {ha} B, body {da.size} B")
    print(f"B {b}: header {hb} B, body {db.size} B")
    if da.size != db.size:
        print("RESULT: DIFFER (body sizes differ)")
        return 1
    diff = np.flatnonzero(da != db)
    if diff.size == 0:
        print(f"RESULT: IDENTICAL ({da.size} body bytes)")
        return 0
    print(f"RESULT: DIFFER in {diff.size} bytes; first at body offset {diff[0]}, "
          f"last at {diff[-1]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
