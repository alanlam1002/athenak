#!/bin/bash
#
# MPI+OpenMP production build script for the scalarized-neutron-star (RNS-ST) problem
# generator, -DPROBLEM=dyn_grmhd/dyngr_rns_st, on Sakura. Same module set/compilers as
# build_rns_st_sakura.sh (the serial smoke-test build, kept untouched), but:
#   - -DAthena_ENABLE_MPI=ON (that build was serial-only -- launching it under `srun`
#     with >1 task just runs N independent, uncoordinated full-domain copies side by
#     side, not a real domain decomposition)
#   - Kokkos debug/bounds-check turned OFF, matching the existing build_tov_mpi_p4
#     production-build convention, appropriate now that this is a real multi-node run
#     rather than a quick smoke test
# Keeps OpenMP ON (unlike build_tov_mpi_p4, which is pure-MPI/no-OpenMP) since the
# SLURM job script for this run uses a hybrid MPI+OpenMP layout
# (--cpus-per-task=20 + OMP_NUM_THREADS=20).

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

ATHENAK_ROOT=${HOME}/athenak_st
BUILD_DIR=${ATHENAK_ROOT}/build_rns_st_mpi

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
  -DPROBLEM=dyn_grmhd/dyngr_rns_st \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_ENABLE_OPENMP=ON \
  -DCMAKE_CXX_COMPILER=${IMPI_ROOT}/bin/mpiicpx \
  -DCMAKE_C_COMPILER=${INTEL_ROOT}/bin/icx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-fp-model=precise --gcc-toolchain=${GCC_TOOLCHAIN}" \
  -DKokkos_ARCH_NATIVE=ON \
  -DKokkos_ENABLE_DEBUG=OFF \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
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
