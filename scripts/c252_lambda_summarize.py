import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
TP={'P10':459.8,'P18':491.1,'PD07':508.7}; COL={'P10':'#2a78d6','P18':'#eb6834','PD07':'#1baf7a'}
D={k:pd.read_csv(f'lam_{k}.csv').sort_values('t').reset_index(drop=True) for k in TP}
print('Lambda at t - t_p:')
for k,d in D.items():
    x=d.t-TP[k]; out=[]
    for dt in (-100,-50,0,50,100,200,300,400,500,600,700):
        if x.min()-1<=dt<=x.max()+1:
            i=(x-dt).abs().idxmin(); out.append(f'{dt:+d}:{d.Lambda[i]:.3g}(M{d.M_trk[i]:.2f},rm{d.rhomax_loc[i]:.3g})')
    print(k,'  '.join(out))
def trig(d,L,tp,after=None):
    below=d.Lambda<L; t0=None
    for i in range(len(d)):
        if after is not None and d.t[i]-tp<after: t0=None; continue
        if below[i]:
            if t0 is None: t0=d.t[i]
            if d.t[i]-t0>=20: return t0, d.t[i]
        else: t0=None
    return None
for L in (0.3,0.5,1,2,3):
    print('Lambda_lost',L, {k:(None if (r:=trig(d,L,TP[k])) is None else tuple(round(v-TP[k],1) for v in r)) for k,d in D.items()},
          ' after t_p+50:', {k:(None if (r:=trig(d,L,TP[k],50)) is None else tuple(round(v-TP[k],1) for v in r)) for k,d in D.items()})
for k,d in D.items():
    for lo,hi in ((-1e9,0),(0,1e9),(50,1e9)):
        s=d[(d.t-TP[k]>=lo)&(d.t-TP[k]<hi)]; i=s.Lambda.idxmin()
        print(k,f'min Lambda in t-t_p [{lo},{hi}): {s.Lambda[i]:.3g} at t-t_p={s.t[i]-TP[k]:+.0f}')
    print(k,'max offset from origin',d.off.max().round(2),'last t-t_p',round(d.t.iloc[-1]-TP[k]),'no-target dumps',int((d.ok==0).sum()))
fig,ax=plt.subplots(figsize=(7.5,4.2))
for k,d in D.items(): ax.semilogy(d.t-TP[k],d.Lambda,color=COL[k],label=k,marker='o',ms=2.5)
ax.axhline(1,color='#52514e',ls='--',lw=1); ax.axvline(0,color='#52514e',ls=':',lw=0.8)
ax.set_xlabel('t − t_p  [M]'); ax.set_ylabel('Λ_trk = ρ̄_trk / ρ_tid'); ax.legend(frameon=False)
ax.set_title('c252: relative-tracking Λ_trk, dx 1/16 production (f_rel 0.3)',loc='left',fontsize=10)
fig.tight_layout(); fig.savefig('c252_lambda.png',dpi=130)
