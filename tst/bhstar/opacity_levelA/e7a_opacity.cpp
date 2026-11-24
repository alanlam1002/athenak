// bhstar WP2 branch 3, test E7(A): evaluate OpacityFunction (opacity_model = saha_kramers)
// at (rho, T) pairs read from stdin and print kappa_s, kappa_aR, kappa_P in cm^2/g.
// Build: g++ -O2 -I stub -I ../../../src e7a_opacity.cpp -o e7a_opacity
#include <cstdio>
#include "radiation/radiation_opacities.hpp"

int main() {
  double rho, t;
  while (std::scanf("%lf %lf", &rho, &t) == 2) {
    Real sa, ss, sp;
    // unit scales 1 and dens = rho in cgs: sigma = rho * kappa (per cm)
    OpacityFunction(rho, 1.0, t, 1.0, 1.0, 2.0/3.0, 0.593, false, true,
                    0.0, 0.0, 0.0, 0.0, 0.0, sa, ss, sp);
    std::printf("%.17e %.17e %.17e %.17e %.17e\n", rho, t, ss/rho, sa/rho, (sa + sp)/rho);
  }
  return 0;
}
