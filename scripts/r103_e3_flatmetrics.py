import sys, glob, numpy as np
sys.path.insert(0, '/home/tlam/athenak_tde/scripts')
import de_J_tests as J
D = sys.argv[1]
def tabs(d, kind='mhd_w_bcc'):
    out = []
    for fn in sorted(glob.glob(f'{d}/tab/*{kind}*.tab')):
        t = float(open(fn).readline().split('time=')[1].split()[0])
        a = np.loadtxt(fn, ndmin=2)
        out.append(dict(t=t, x=a[:, 2], rho=a[:, 3], u=a[:, 4], P=a[:, 7], a=a))
    return out
J.tabs = tabs
ref = {M: J.post_shock_K(f'{D}/col_M{M}_off') for M in J.MACH}
print('energy-only post-shock K/K0 (RH):', ', '.join(f'M={M} {ref[M]:.5f} ({J.k_rh(float(M)):.5f})' for M in J.MACH))
for v in ('coord', 'covariant'):
    cap = [(J.post_shock_K(f'{D}/col_M{M}_{v}') - 1) / (ref[M] - 1) for M in J.MACH]
    print(f'{v:9s} captured fraction at M = 1.2/1.5/2/3: ' + ' / '.join(f'{x:.3f}' for x in cap))
    for c in J.HOM:
        fin = tabs(f'{D}/hom_c{c}_{v}')[-1]
        dn = fin['rho'] > 0.3 * fin['rho'].max()
        K = fin['P'][dn] / fin['rho'][dn]**J.GAM / J.K0
        dg = tabs(f'{D}/hom_c{c}_{v}', 'mhd_de_diag')
        nfl = sum(int((g['a'][:, 5] > 0.5).sum()) for g in dg)
        print(f'   hom c={c:4s}: rho>0.3rho_max K/K0-1 max {K.max()-1:+.2e} min {K.min()-1:+.2e}; flagged cells summed over dumps {nfl}')
    s = tabs(f'{D}/sod_{v}')[-1]; r = tabs(f'{D}/sod_off')[-1]
    print(f'   sod: max |P - P_energyonly|/P {np.max(np.abs(s["P"]-r["P"])/r["P"]):.2e}, max |rho-rho_eo|/rho {np.max(np.abs(s["rho"]-r["rho"])/r["rho"]):.2e}')
