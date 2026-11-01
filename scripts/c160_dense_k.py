#!/usr/bin/env python3
"""c160_trd: dense-band (rho >= 0.01, 0.1 rho_c) rho dV-weighted K/K0 - 1 per prim_3d dump (ISO_A rho_c, kappa).  c160_dense_k.py RUNDIR..."""
import sys,types,glob,numpy as np
sys.modules['h5py']=types.ModuleType('h5py'); sys.path.insert(0,'/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc
rc=3.24e-4; k0=1.2346626297e-2; G=5/3
for d in sys.argv[1:]:
    out=[]
    for f in sorted(glob.glob(f'{d}/bin/*prim_3d*.bin')):
        P=bc.read_binary(f); g=np.asarray(P['mb_geometry'],float); n=P['nx1_out_mb']
        r=np.array([np.asarray(P['mb_data']['dens'][b],float) for b in range(len(g))]); p=np.array([np.asarray(P['mb_data']['press'][b],float) for b in range(len(g))])
        dv=(((g[:,1]-g[:,0])/n)**3)[:,None,None,None]; K=p/r**G/k0; w=r*dv
        s=[]
        for th in (0.01,0.1):
            m=r>=th*rc; s.append((w[m]*(K[m]-1)).sum()/w[m].sum())
        out.append(f"t={P['time']:.0f}:{100*s[0]:+.2f}/{100*s[1]:+.2f}")
    print(d, ' '.join(out))
