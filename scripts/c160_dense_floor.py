#!/usr/bin/env python3
"""c160 (R-194 B.1 readout, C-156 item 4): per arm and 3D dump, the dense-band statistics of c122_b1.py (mass-weighted
<K/K0 - 1> over rho >= 0.01 / 0.1 rho_c, warm > +5% mass fraction, rho_max/rho_c), each also EXCLUDING the cells at the
temperature floor (T <= 1.0001e-22), and the floored mass fraction.
    c160_dense_floor.py OUT.csv LABEL=RUNDIR [...] [--times 72,96,120]"""
import glob, re, sys, types
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402
GAM = 5.0/3.0
TFL = 1.0001e-22


def par(d, k):
    return float(re.search(rf'^\s*{k}\s*=\s*([-+.\deE]+)', open(f'{d}/parfile.par').read(), re.M).group(1))


def main():
    out = sys.argv[1]
    ts = [float(x) for x in sys.argv[sys.argv.index('--times') + 1].split(',')] if '--times' in sys.argv else [72, 96, 120]
    specs = [a for a in sys.argv[2:] if '=' in a]
    rows = ['arm,t,d01_mean,d01_mean_unfl,d01_warm5,d01_floor_frac,d1_mean,d1_mean_unfl,d1_floor_frac,rhomax']
    for s in specs:
        lab, d = s.split('=', 1)
        rhoc, k0 = par(d, 'rhoc'), par(d, 'kappa')
        fs = sorted(glob.glob(f'{d}/bin/*.prim_3d.0*.bin'))
        tt = [bc.read_binary(f)['time'] for f in fs]
        for t in ts:
            f = fs[int(np.argmin(np.abs(np.array(tt) - t)))]
            P = bc.read_binary(f); g = np.asarray(P['mb_geometry']); n = P['nx1_out_mb']; q = P['mb_data']
            cat = lambda k: np.concatenate([np.asarray(q[k][b]).ravel() for b in range(len(g))]).astype(float)
            rho, p, T = cat('dens'), cat('press'), cat('temperature')
            dv = np.concatenate([np.full(n**3, ((g[b, 1] - g[b, 0])/n)**3) for b in range(len(g))])
            x = p/rho**GAM/k0 - 1.0; fl = T <= TFL; w = rho*dv
            r = [lab, f'{P["time"]:.2f}']
            for thr in (0.01, 0.1):
                m = rho >= thr*rhoc; mu = m & ~fl
                mean = (w[m]*x[m]).sum()/w[m].sum(); mun = (w[mu]*x[mu]).sum()/w[mu].sum()
                ff = w[m & fl].sum()/w[m].sum()
                if thr == 0.01:
                    r += [f'{mean:.5f}', f'{mun:.5f}', f'{w[m][x[m] > 0.05].sum()/w[m].sum():.4f}', f'{ff:.4f}']
                else:
                    r += [f'{mean:.5f}', f'{mun:.5f}', f'{ff:.4f}']
            r.append(f'{rho.max()/rhoc:.4f}')
            rows.append(','.join(r)); print(rows[-1], flush=True)
    open(out, 'w').write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
