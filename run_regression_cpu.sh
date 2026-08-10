#!/bin/bash
#SBATCH -J athenak_regr_cpu
#SBATCH -o /sakura/ptmp/tlam/athenak_merge_largesim2/regression_cpu.out
#SBATCH -e /sakura/ptmp/tlam/athenak_merge_largesim2/regression_cpu.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 01:30:00
#SBATCH -p p.sakura

set -e

module purge
module load cmake/3.26
module load gcc/13
module load intel/2025.3
module load impi/2021.17
module load gsl/2.4

export OMP_NUM_THREADS=1

INTEL_ROOT=/mpcdf/soft/SLE_15/packages/x86_64/intel/2025.3.0
GCC_TOOLCHAIN=/mpcdf/soft/SLE_15/packages/x86_64/gcc/13.1.0

REPO=/sakura/ptmp/tlam/athenak_merge_largesim2
BUILD_DIR=${REPO}/tst/build
CUSTOM_TEST=test_suite/dyngrmhd/test_dyngrmhd_dynbbh_metric_cpu.py

echo "=== [1/2] default build (built_in_pgens) into ${BUILD_DIR} (bypassing"
echo "    run_test_suite.py's own testutils.cmake(), which picks up the wrong"
echo "    system compiler by default) ==="
rm -rf ${BUILD_DIR}
cmake -B ${BUILD_DIR} \
  -DAthena_ENABLE_MPI=OFF \
  -DCMAKE_CXX_COMPILER=${INTEL_ROOT}/bin/icpx \
  -DCMAKE_C_COMPILER=${INTEL_ROOT}/bin/icx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-fp-model=precise --gcc-toolchain=${GCC_TOOLCHAIN}" \
  -DKokkos_ARCH_NATIVE=ON \
  ${REPO}
cmake --build ${BUILD_DIR} -j 32
ln -sf ../../inputs ${BUILD_DIR}/src/inputs

echo "=== running full _cpu pytest suite, excluding ${CUSTOM_TEST} -- that one"
echo "    needs its own dedicated -D PROBLEM=dyn_grmhd/dynbbh build, done"
echo "    separately below. Matches tst/run_test_suite.py's own"
echo "    run_tests_with_custom_problem() pattern exactly (build default minus"
echo "    the custom test, then rebuild+rerun just the custom test) --"
echo "    previously this file skipped that split and ran the custom test"
echo "    against the default build, where it always failed (wrong pgen) ==="
set +e
cat > /tmp/run_regr_default_$$.py << PYEOF
import os
import sys

TST = "${REPO}/tst"
os.chdir(TST)
sys.path.insert(0, TST)
import test_suite.testutils as testutils  # noqa: E402,F401
import pytest  # noqa: E402

os.chdir(os.path.join(TST, "build", "src"))
sys.exit(pytest.main(["../../test_suite/", "-k", "_cpu", "-v",
                       "--ignore", os.path.join(TST, "${CUSTOM_TEST}")]))
PYEOF
python /tmp/run_regr_default_$$.py
rc1=$?
rm -f /tmp/run_regr_default_$$.py
echo "=== default _cpu suite (minus ${CUSTOM_TEST}) exited with $rc1 ==="
set -e

echo "=== [2/2] dedicated build for ${CUSTOM_TEST}: -D PROBLEM=dyn_grmhd/dynbbh"
echo "    (rm -rf + reconfigure the same ${BUILD_DIR} -- matches"
echo "    testutils.clean_make()'s own reuse of tst/build rather than a"
echo "    separate directory) ==="
rm -rf ${BUILD_DIR}
cmake -B ${BUILD_DIR} \
  -DAthena_ENABLE_MPI=OFF \
  -DCMAKE_CXX_COMPILER=${INTEL_ROOT}/bin/icpx \
  -DCMAKE_C_COMPILER=${INTEL_ROOT}/bin/icx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-fp-model=precise --gcc-toolchain=${GCC_TOOLCHAIN}" \
  -DKokkos_ARCH_NATIVE=ON \
  -DPROBLEM=dyn_grmhd/dynbbh \
  ${REPO}
cmake --build ${BUILD_DIR} -j 32
ln -sf ../../inputs ${BUILD_DIR}/src/inputs

echo "=== running ${CUSTOM_TEST} against the dyn_grmhd/dynbbh build ==="
set +e
cat > /tmp/run_regr_dynbbh_$$.py << PYEOF
import os
import sys

TST = "${REPO}/tst"
os.chdir(TST)
sys.path.insert(0, TST)
import test_suite.testutils as testutils  # noqa: E402,F401
import pytest  # noqa: E402

os.chdir(os.path.join(TST, "build", "src"))
sys.exit(pytest.main([os.path.join(TST, "${CUSTOM_TEST}"), "-v"]))
PYEOF
python /tmp/run_regr_dynbbh_$$.py
rc2=$?
rm -f /tmp/run_regr_dynbbh_$$.py
echo "=== ${CUSTOM_TEST} exited with $rc2 ==="
set -e

echo "=== SUMMARY: default suite rc=${rc1}, ${CUSTOM_TEST} rc=${rc2} ==="
if [ ${rc1} -ne 0 ] || [ ${rc2} -ne 0 ]; then
  exit 1
fi
exit 0
