#!/bin/bash
#
# Build script for AthenaK with the CFC (conformally flat condition) solver and
# the Celephais BNS initial-data reader on Sakura, via the dyn_grmhd/celephais/
# celephais_bns problem generator (CFC variant -- no z4c involved).
#
# Renamed from cfc_kadath_bns_sakura.sh: Kadath_release (the AEI's Kadath fork
# used for BNS initial data) renamed itself to "Celephais" upstream -- new
# library name (libcelephais.a), new CMake cache variable (CELEPHAIS_EXTRA_LIBS).
# Also adopts the Fortran-compiler/Kadath-toolchain fixes verified working on
# the z4c-side celephais_bns_sakura.sh: mpiicx (not icx) for C, and
# CMAKE_Fortran_COMPILER=mpigfortran-13.1.0 instead of MPI_Fortran_COMPILER.

set -e  # Exit on error

# Load required modules
echo "Loading modules..."
module purge
module load cmake/3.26
module load gcc/13
module load intel/2025.3
module load mkl/2025.3
module load impi/2021.17
module load gsl/2.4
module load fftw-serial/3.3.10
module load boost/1.83

echo "Currently loaded modules:"
module list

# Set paths
ATHENAK_ROOT=${HOME}/athenak_cfc
BUILD_DIR=${ATHENAK_ROOT}/build_cfc_celephais_bns

INTEL_ROOT=/mpcdf/soft/SLE_15/packages/x86_64/intel/2025.3.0
IMPI_ROOT=/mpcdf/soft/SLE_15/packages/x86_64/intel_oneapi/2025.3/mpi/2021.17
GCC_TOOLCHAIN=/mpcdf/soft/SLE_15/packages/x86_64/gcc/13.1.0

# Create and navigate to build directory
echo "Setting up build directory: ${BUILD_DIR}"
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Clean previous configuration
echo "Cleaning previous CMake configuration..."
rm -rf CMakeCache.txt CMakeFiles

# Configure with CMake
echo "Configuring with CMake..."
cmake \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_ENABLE_OPENMP=ON \
  -DPROBLEM=dyn_grmhd/celephais/celephais_bns \
  -DCMAKE_CXX_COMPILER=${IMPI_ROOT}/bin/mpiicpx \
  -DCMAKE_C_COMPILER=${IMPI_ROOT}/bin/mpiicx \
  -DCMAKE_Fortran_COMPILER=mpigfortran-13.1.0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-fp-model=precise --gcc-toolchain=${GCC_TOOLCHAIN}" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$FFTW_HOME/lib" \
  -DBLA_VENDOR=Intel10_64lp_seq \
  -DLAPACK_ROOT=${MKLROOT} \
  -DBLAS_ROOT=${MKLROOT} \
  -DKokkos_ARCH_NATIVE=ON \
  -DKokkos_ENABLE_DEBUG=ON \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
  -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  ${ATHENAK_ROOT}

# Build
echo "Building with make..."
make -j20

# Check if executable was created
if [ -f "${BUILD_DIR}/src/athena" ]; then
    echo ""
    echo "=========================================="
    echo "Build successful!"
    echo "Executable: ${BUILD_DIR}/src/athena"
    ls -lh ${BUILD_DIR}/src/athena
    echo "=========================================="
    echo ""
    echo "To run the executable, load the required modules:"
    echo "  module purge"
    echo "  module load gcc/10 impi/2019.9 gsl/2.4 fftw-serial/3.3.10 mkl"
    echo "  export LD_LIBRARY_PATH=/mpcdf/soft/SLE_15/packages/skylake/fftw/gcc_10-10.3.0/3.3.10/lib:\$LD_LIBRARY_PATH"
    echo ""
else
    echo "Error: Executable not found!"
    exit 1
fi
