"""
Multigrid manufactured-solution tests (GPU): fine/coarse interface accuracy.

pgen mg_manufactured solves the gravity Poisson problem for a known potential on a
periodic, statically refined mesh and prints the error per cell class and the second
difference of the error across block edges on coarser faces (MGMANU, MGKINK, MGOPEN).
Two resolutions (32^3 and 64^3, the same block tree) give the observed orders.

- nested boxes (inputs/mg_manufactured.athinput): centred prolongation slopes at every
  face end;
- L-shaped level 1 (inputs/mg_manufactured_L.athinput): adds face ends next to a
  same-level edge neighbour, where the prolongation slope is one-sided.

The coarser-face rule (src/cfc/DEVELOPMENT.md item 67) must give phi second order, the
gradient error converging, and no kink at block edges
(mg_utils.assert_mg_manufactured_orders).
"""

import pytest

from test_suite.multigrid.mg_utils import (
    run_mg_manufactured, assert_mg_manufactured_orders, cleanup,
)


@pytest.mark.parametrize("deck, open_ends", [
    ("inputs/mg_manufactured.athinput", False),
    ("inputs/mg_manufactured_L.athinput", True),
])
def test_mg_manufactured_gpu(deck, open_ends):
    """Observed orders 32^3 -> 64^3 and absence of a kink at coarser-face block edges."""
    try:
        lo = run_mg_manufactured(deck, 32)
        hi = run_mg_manufactured(deck, 64)
        assert_mg_manufactured_orders(lo, hi, label=f"mg_manufactured_gpu {deck}: ",
                                      open_ends=open_ends)
    finally:
        cleanup()
