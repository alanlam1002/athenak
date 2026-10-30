#!/usr/bin/env python3
"""c150: per prim_3d dump, non-finite blocks and the peak |Wv| (block, rho there).
    c150_nanloc.py DUMP.bin [...]"""
import sys,os,glob,numpy as np
sys.path.insert(0,'/lus/flare/projects/CompactBinaryMerger/tlam/home_builds/athenak_tde/wt_tde_scripts/scripts')
from tde_ensemble_plots import bc
for f in sys.argv[1:]:
    P=bc.read_binary(f); g=np.asarray(P['mb_geometry']); q=P['mb_data']; n=P['nx1_out_mb']
    bad=[];vm=(0,None,0.0)
    for b in range(len(g)):
        x0,x1,y0,y1,z0,z1=g[b][:6]
        r=np.asarray(q['dens'][b]); ok=np.isfinite(r)&np.isfinite(np.asarray(q['velx'][b]))
        if (~ok).any(): bad.append((b,int((~ok).sum()),[round(v,2) for v in g[b][:6]]))
        u=np.sqrt(sum(np.asarray(q[k][b])**2 for k in ('velx','vely','velz')))
        u=np.where(np.isfinite(u),u,0); 
        if u.max()>vm[0]: i=np.unravel_index(u.argmax(),u.shape); vm=(u.max(),[round(v,2) for v in g[b][:6]],float(r[i]))
    print(os.path.basename(f),'t=%.2f'%P['time'],'nonfinite blocks',len(bad),bad[:3],'max |Wv| %.3g at block %s rho %.2e'%vm)
