"""Shared .tab reader for single-zone (reduced-dimension) M1 tests.

vis/python/athena_read.tab() assumes every column named in the header
comment (gid, i, x1v, j, x2v, k, x3v, <vars...>) is present in the data
rows. AthenaK's tab writer, however, drops the coordinate columns for
degenerate (size-1) mesh dimensions from the data rows while still listing
them in the header comment, so athena_read.tab() misparses any tab output
from a reduced-dimension run (e.g. a single-zone test, which is
degenerate in all three dimensions by construction). This reader sidesteps
that mismatch: the physical variable columns are always the trailing
columns of both the header and each data row, regardless of how many
leading coordinate columns got dropped.
"""
import glob
import os
import re

import numpy as np

_COORD_HEADER_COLS = ["gid", "i", "x1v", "j", "x2v", "k", "x3v"]


def read_tab(fname):
    with open(fname) as f:
        lines = f.readlines()
    time = float(re.search(r"time=(\S+)", lines[0]).group(1))
    header = lines[1].lstrip("#").split()
    var_names = header[len(_COORD_HEADER_COLS):]
    data = {"time": time}
    columns = [[] for _ in var_names]
    for line in lines[2:]:
        vals = [float(x) for x in line.split()]
        for column, val in zip(columns, vals[-len(var_names):]):
            column.append(val)
    for name, column in zip(var_names, columns):
        data[name] = np.array(column)
    return data


def load_series(tab_dir, basename, file_id, column):
    """Load a (time, spatially-averaged value) series from a sequence of .tab files."""
    pattern = os.path.join(tab_dir, "{}.{}.*.tab".format(basename, file_id))
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("No files matched {} -- run AthenaK first".format(pattern))
    times, values = [], []
    for fname in files:
        data = read_tab(fname)
        times.append(data["time"])
        values.append(np.mean(data[column]))
    order = np.argsort(times)
    return np.array(times)[order], np.array(values)[order]
