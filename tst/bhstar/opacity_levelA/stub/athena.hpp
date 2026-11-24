// Minimal stand-in for src/athena.hpp so that radiation_opacities.hpp compiles on the host
// without Kokkos (bhstar E7(A) test only).
#ifndef ATHENA_HPP_
#define ATHENA_HPP_
#include <cmath>
using Real = double;
#define KOKKOS_INLINE_FUNCTION inline
#endif
