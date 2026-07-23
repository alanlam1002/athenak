#!/bin/bash
#
# Build script for AthenaK's reference (discrete-ordinate) radiation
# equilibration test, -DPROBLEM=rad_relax, on Sakura. Modeled on
# build_diffref_sakura.sh. Separate build directory from build_m1 since
# PROBLEM=rad_relax is mutually exclusive with the M1 tests' built_in_pgens
# build. Used for Stage 7 (paper arXiv:2302.04283 Section 3.6 Equilibration
# cross-check).

set -e  # Exit on error

echo "Loading modules..."
module purge
module load cmake/3.26
module load gcc/13
module load intel/2025.3
module load impi/2021.17
module load gsl/2.4

echo "Currently loaded modules:"
module list

ATHENAK_ROOT=${HOME}/athenak_m1
BUILD_DIR=${ATHENAK_ROOT}/build_relax

INTEL_ROOT=/mpcdf/soft/SLE_15/packages/x86_64/intel/2025.3.0
IMPI_ROOT=/mpcdf/soft/SLE_15/packages/x86_64/intel_oneapi/2025.3/mpi/2021.17
GCC_TOOLCHAIN=/mpcdf/soft/SLE_15/packages/x86_64/gcc/13.1.0

echo "Setting up build directory: ${BUILD_DIR}"
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

echo "Cleaning previous CMake configuration..."
rm -rf CMakeCache.txt CMakeFiles

echo "Configuring with CMake..."
cmake \
  -DPROBLEM=rad_relax \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_ENABLE_OPENMP=ON \
  -DCMAKE_CXX_COMPILER=${IMPI_ROOT}/bin/mpiicpx \
  -DCMAKE_C_COMPILER=${INTEL_ROOT}/bin/icx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-fp-model=precise --gcc-toolchain=${GCC_TOOLCHAIN}" \
  -DKokkos_ARCH_NATIVE=ON \
  -DKokkos_ENABLE_DEBUG=ON \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
  -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  ${ATHENAK_ROOT}

echo "Building with make..."
make -j20

if [ -f "${BUILD_DIR}/src/athena" ]; then
    echo ""
    echo "=========================================="
    echo "Build successful!"
    echo "Executable: ${BUILD_DIR}/src/athena"
    ls -lh ${BUILD_DIR}/src/athena
    echo "=========================================="
else
    echo "Error: Executable not found!"
    exit 1
fi
