import sys, types, glob
sys.modules['h5py']=types.ModuleType('h5py')
sys.path.insert(0,'/home/tlam/athenak_tde/vis/python')
import bin_convert as bc
import numpy as np
BASE='/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/tde_elliptic_tracker_nexteval_v2/output-0000/bin'
GAM=5.0/3.0; GF=GAM/(GAM-1.0)   # rho*h = rho + GF*p
EDGES=[0.0,2.0,5.0,10.0,20.0,40.0,1e9]

def one(n):
    dp=bc.read_binary('%s/cfc_tde_wd_imbh_elliptic_tracker.prim_xy.%05d.bin'%(BASE,n))
    da=bc.read_binary('%s/cfc_tde_wd_imbh_elliptic_tracker.adm_xy.%05d.bin'%(BASE,n))
    nx,ny=dp['nx1_out_mb'],dp['nx2_out_mb']
    gp={tuple(l):i for i,l in enumerate(np.asarray(dp['mb_logical']).tolist())}
    ga={tuple(l):i for i,l in enumerate(np.asarray(da['mb_logical']).tolist())}
    geo=np.asarray(dp['mb_geometry'])
    tau_b=np.zeros(len(EDGES)-1); D_b=np.zeros(len(EDGES)-1); Wmax=0.0; Wmax_r=0.0
    for key,ip in gp.items():
        if key not in ga: continue
        ia=ga[key]
        rho=np.asarray(dp['mb_data']['dens'])[ip,0]
        p  =np.asarray(dp['mb_data']['press'])[ip,0]
        ux =np.asarray(dp['mb_data']['velx'])[ip,0]
        uy =np.asarray(dp['mb_data']['vely'])[ip,0]
        uz =np.asarray(dp['mb_data']['velz'])[ip,0]
        psi4=np.asarray(da['mb_data']['adm_psi4'])[ia,0]
        x1min,x1max,x2min,x2max=geo[ip,0],geo[ip,1],geo[ip,2],geo[ip,3]
        dx=(x1max-x1min)/nx; dy=(x2max-x2min)/ny
        xs=x1min+(np.arange(nx)+0.5)*dx; ys=x2min+(np.arange(ny)+0.5)*dy
        X,Y=np.meshgrid(xs,ys); r=np.hypot(X,Y)
        psi4=np.maximum(psi4,1e-300)
        W2=1.0+psi4*(ux*ux+uy*uy+uz*uz); W2=np.maximum(W2,1.0); W=np.sqrt(W2)
        sg=psi4**1.5
        D=sg*rho*W
        tau=sg*((rho+GF*p)*W2-p)-D
        dA=dx*dy
        idx=np.digitize(r.ravel(),EDGES)-1
        np.add.at(tau_b,idx,(tau*dA).ravel())
        np.add.at(D_b,idx,(D*dA).ravel())
        # track max W among cells with non-trivial density
        msk=rho>1e-12
        if msk.any():
            k=np.argmax(np.where(msk,W,0)); k=np.unravel_index(k,W.shape)
            if W[k]>Wmax: Wmax=W[k]; Wmax_r=r[k]
    return dp['time'],tau_b,D_b,Wmax,Wmax_r

lbl=['r<2','2-5','5-10','10-20','20-40','r>40']
print('%7s | %s | %s | %9s %8s'%('t',' '.join('%8s'%l for l in lbl),' '.join('%7s'%('M:'+l) for l in ['r<2','2-5','5-10','>10']),'Wmax','@r'))
for n in [int(a) for a in sys.argv[1:]]:
    t,tb,Db,Wm,Wr=one(n)
    ft=tb/tb.sum(); fD=Db/Db.sum()
    print('%7.1f | %s | %s | %9.2f %8.2f'%(t,
        ' '.join('%8.4f'%v for v in ft),
        ' '.join('%7.4f'%v for v in [fD[0],fD[1],fD[2],fD[3:].sum()]),Wm,Wr))
