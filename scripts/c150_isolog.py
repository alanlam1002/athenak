#!/usr/bin/env python3
"""c150: per ISO arm, from out.log: cycles, mean/min/max dt, wall per cycle, MCLOSE R/M_grid, FOFC (CRECOV fofc_hydro/fofc_faces) totals.
    c150_isolog.py RUNDIR [...]"""
import re,sys
for a in sys.argv[1:]:
    dts=[];fh=[];mc=None
    for l in open(a+'/out.log'):
        if l.startswith('MLEDGER '):
            m=re.search(r'cycle=(\d+) time=(\S+) dt=(\S+)',l); dts.append((int(m.group(1)),float(m.group(2)),float(m.group(3))))
        elif l.startswith('CRECOV'):
            m=re.search(r'cycle=(\d+).*fofc_hydro=(\S+).*fofc_faces=(\S+)',l); fh.append((int(m.group(1)),float(m.group(2)),float(m.group(3))))
        elif l.startswith('MCLOSE '): mc=re.search(r'R/M_grid=(\S+) .* (PASS|FAIL)',l).groups()
        elif l.startswith('PHASE_TIMERS_FINAL'): w=float(re.search(r'wall=(\S+)',l).group(1)); nc=int(re.search(r' cycles=(\d+)',l).group(1))
    d=[x[2] for x in dts]
    first=next((c for c,h,f in fh if h>0),None)
    print(a,'ncyc',nc,'mean dt %.4f'%(120/nc),'min/max dt %.3g/%.3g'%(min(d),max(d)),'wall/cyc %.3f'%(w/nc),'MCLOSE',mc,'fofc_hydro tot %.3g ncyc>0 %d first %s last %.3g faces_last %.3g'%(sum(h for c,h,f in fh),sum(1 for c,h,f in fh if h>0),first,fh[-1][1],fh[-1][2]))
