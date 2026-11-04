import sys
def rows(f):
    return [l.split() for l in open(f) if not l.startswith('#')]
def hdr(f):
    return [l for l in open(f) if l.startswith('#')][-1].strip()
H='cfc_tde_wd_imbh_parabolic_tracker'
def cmp(a,b,ca,cb,lab):
    A,B=rows(a),rows(b)
    n=min(len(A),len(B)); bad=[i for i in range(n) if [A[i][c] for c in ca]!=[B[i][c] for c in cb]]
    print(f'{lab}: rows {len(A)} vs {len(B)}; compared {n}; mismatching rows {len(bad)}', ('first '+str(bad[0])) if bad else '')
    return A,B
# G1 vs G0
cmp(f'G1/{H}.mhd.hst',f'G0/{H}.mhd.hst',list(range(13))+[15],list(range(13))+[13],'G1 vs G0 mhd (cols1-13 + tracer)')
cmp(f'G1/{H}.user.hst',f'G0/{H}.user.hst',None or list(range(len(rows(f'G0/{H}.user.hst')[0]))),list(range(len(rows(f'G0/{H}.user.hst')[0]))),'G1 vs G0 user')
print('user hdr same:',hdr(f'G1/{H}.user.hst')==hdr(f'G0/{H}.user.hst'))
# tag mass columns
A=rows(f'G1/{H}.mhd.hst'); print('G1 tag masses first/last:',A[0][13],A[0][14],'|',A[-1][13],A[-1][14],'mass',A[0][2],A[-1][2])
# Gd vs c192 De
De='../c192_b1de/De'
A,B=rows(f'Gd/{H}.mhd.hst'),rows(f'{De}/{H}.mhd.hst')
n=len(A); bad=[i for i in range(n) if A[i]!=B[i]]
print(f'Gd vs De mhd: Gd rows {n}; mismatching {len(bad)}', ('first row %d t=%s'%(bad[0],A[bad[0]][0])) if bad else '')
A,B=rows(f'Gd/{H}.user.hst'),rows(f'{De}/{H}.user.hst')
n=len(A); bad=[i for i in range(n) if A[i]!=B[i]]
print(f'Gd vs De user: Gd rows {n}; mismatching {len(bad)}', ('first row %d t=%s'%(bad[0],A[bad[0]][0])) if bad else '')
# Gr vs G1: rows with time > restart time
for k in ('mhd','user'):
    A,B=rows(f'Gr/{H}.{k}.hst'),rows(f'G1/{H}.{k}.hst')
    t0=float(A[0][0]); Bs=[r for r in B if float(r[0])>=t0]
    n=min(len(A),len(Bs)); bad=[i for i in range(n) if A[i]!=Bs[i]]
    print(f'Gr vs G1 {k}: Gr rows {len(A)} from t={t0}; G1 rows in range {len(Bs)}; mismatching {len(bad)}')
