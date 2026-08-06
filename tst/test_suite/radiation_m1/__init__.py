"""
Photon grey M1 radiation transport tests.

Bootstraps sys.path so test files in this package can import the
hand-validated check scripts and shared tab reader under inputs/tests/
(inputs/tests/m1_tab_utils.py, inputs/tests/check_rad_m1_photon_*.py)
directly, instead of duplicating their analytic checks here. Resolved
relative to this file's own location (not cwd) since pytest imports this
package after the harness has already chdir'd into tst/build/src.
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "inputs", "tests"))
