"""
bhstar Stage-0 regression test (numerics-005 U4), GPU (1 rank) version.
Runs the reduced Stage-0a problem (inputs/bhstar_stage0_small.athinput: free fall, radiation,
shell source, energy budget on) for 100 cycles and checks
  - the energy-budget closure (E(t) - E(0) - sum of closure terms) relative to the
    injected energy is below 1e-9 in every history row,
  - no skipped radiation-fluid coupling cells,
  - all history values are finite and the radiation energy has grown.
"""

# Modules
import os
import numpy as np
import pytest
import test_suite.testutils as testutils

CLOSURE = ['inj', 'rad_upd', 'gas_upd', 'gas_src', 'coupl', 'c2p', 'mask', 'regrid']
HST = "bhstar_s0.user.hst"


def arguments():
    """Assemble arguments for run command"""
    return ["job/basename=bhstar_s0"]


input_file = "inputs/bhstar_stage0_small.athinput"


def read_hst(path):
    labels = None
    for line in open(path):
        if line.startswith('#') and '[1]=' in line:
            labels = [t.split('=')[1] for t in line.split() if '=' in t]
    data = np.loadtxt(path, ndmin=2)
    return {lab: data[:, i] for i, lab in enumerate(labels)}


def test_run():
    """Run the test and check the history."""
    try:
        results = testutils.run(input_file, arguments())
        assert results, "bhstar Stage-0 test run failed."
        h = read_hst(HST)
        for k, v in h.items():
            if not np.all(np.isfinite(v)):
                pytest.fail(f"non-finite values in history column {k}")
        etot = h['E_gas'] + h['E_rad']
        res = (etot - etot[0]) - sum(h[k] for k in CLOSURE)
        scale = np.max(np.abs(h['inj']))
        rel = np.max(np.abs(res))/scale
        if rel > 1.0e-9:
            pytest.fail(f"energy budget does not close: max residual/injected = {rel:g}")
        if np.max(h['n_badcell']) > 0:
            pytest.fail(f"skipped coupling cells: {np.max(h['n_badcell']):g}")
        if not h['E_rad'][-1] > h['E_rad'][0]:
            pytest.fail("radiation energy did not grow with the shell source on")
    finally:
        if os.path.exists(HST):
            os.remove(HST)
        testutils.cleanup()
