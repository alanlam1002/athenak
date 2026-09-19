import sys, types, glob, os
sys.modules['h5py']=types.ModuleType('h5py')
sys.path.insert(0,'/home/tlam/athenak_tde/vis/python')
import bin_convert as bc
import numpy as np

BASE='/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/tde_elliptic_tracker_nexteval_v2/output-0000/bin'
files=sorted(glob.glob(BASE+'/*.prim_xy.*.bin'))
lo,hi=float(sys.argv[1]),float(sys.argv[2])

rows=[]
for f in files:
    n=int(f.split('.')[-2])
    if not (lo<=n<=hi): continue
    d=bc.read_binary(f)
    t=d['time']
    dens=np.asarray(d['mb_data']['dens'])     # (nmb,1,ny,nx)
    geo=np.asarray(d['mb_geometry'])          # x1min,x1max,x2min,x2max,...
    nx=d['nx1_out_mb']; ny=d['nx2_out_mb']
    # global max over all blocks, with coords
    best=(-1e300,0,0); cells=[]
    for m in range(dens.shape[0]):
        blk=dens[m,0]
        x1min,x1max,x2min,x2max=geo[m,0],geo[m,1],geo[m,2],geo[m,3]
        dx=(x1max-x1min)/nx; dy=(x2max-x2min)/ny
        xs=x1min+(np.arange(nx)+0.5)*dx
        ys=x2min+(np.arange(ny)+0.5)*dy
        j,i=np.unravel_index(np.argmax(blk),blk.shape)
        if blk[j,i]>best[0]: best=(blk[j,i],xs[i],ys[j])
        cells.append((blk,xs,ys))
    rmax,xm,ym=best
    # competing clump: highest cell farther than 4.0 (2x box radius) from the max
    second=(-1e300,0,0)
    for blk,xs,ys in cells:
        X,Y=np.meshgrid(xs,ys)
        far=((X-xm)**2+(Y-ym)**2)>16.0
        if far.any():
            v=np.where(far,blk,-1e300); j,i=np.unravel_index(np.argmax(v),v.shape)
            if v[j,i]>second[0]: second=(v[j,i],X[j,i],Y[j,i])
    rows.append((t,xm,ym,rmax,second[0],second[1],second[2]))

rows.sort()
print('%8s %9s %9s %11s %8s %9s %11s'%('t','x_max','y_max','rho_max','jump','r_max','rho2/rho1'))
prev=None
for (t,x,y,r,r2,x2,y2) in rows:
    jump = np.hypot(x-prev[0],y-prev[1]) if prev else 0.0
    print('%8.2f %9.3f %9.3f %11.4e %8.3f %9.3f %11.4f'%(t,x,y,r,jump,np.hypot(x,y),r2/r if r>0 else 0))
    prev=(x,y)
np.save(sys.argv[3], np.array(rows))
