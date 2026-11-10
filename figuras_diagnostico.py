"""Figuras de dependencia temporal y variabilidad entre réplicas, sin datos crudos."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T = pd.read_csv('resultados/tabla_por_trayectoria.csv')
G = json.load(open('resultados/preliminar.json'))['largas']
blue, orange, green = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,3,figsize=(7.2,2.6),layout='constrained')
for ns,color in [(5,blue),(50,orange)]:
    x=np.sort(T.loc[T.ns==ns,'beta_medio'].dropna())
    axs[0].plot(x,np.arange(1,len(x)+1)/len(x),color=color,label=f'{ns} ns (n={len(x):,})')
axs[0].axvline(.9,color=green,ls=':',lw=1)
axs[0].axvline(1,color='0.4',ls='--',lw=1)
axs[0].set(xlabel='Effective MSD exponent β',ylabel='Cumulative fraction',xlim=(0,1.3),ylim=(0,1))
axs[0].legend(loc='lower left',frameon=True,facecolor='white',edgecolor='none',framealpha=.9,fontsize=7)
t=G['retardos_ns']
axs[1].fill_between(t,G['razon_p10'],G['razon_p90'],color=blue,alpha=.18)
axs[1].plot(t,G['razon_mediana'],color=blue,marker='o',ms=3)
axs[1].axhline(1,color='0.4',ls='--',lw=1)
axs[1].set(ylabel=r'$D_{app}(\tau)/D_{app}(25\,ns)$')
axs[2].plot(t,G['spearman_vs_25ns'],color=orange,marker='o',ms=3)
axs[2].set(ylabel='Spearman ρ with 25 ns ranking',ylim=(.85,1.01))
for ax in axs[1:]:
    ax.set_xscale('log');ax.set_xticks([.5,1,5,25],['0.5','1','5','25']);ax.set_xlabel('Observation lag τ (ns)')
for j,ax in enumerate(axs):ax.set_title(f'({chr(97+j)})',loc='left',fontsize=10)
fig.savefig('paper/figures/fig1_time_dependence.pdf');fig.savefig('paper/figures/fig1_time_dependence.png',dpi=180);plt.close(fig)
C=T[(T.ns==5)&~T.literatura].sort_values('id')
R=C.groupby('sistema').filter(lambda g:len(g)==2)
x=R.groupby('sistema').D_extremos.first();y=R.groupby('sistema').D_extremos.last();d=np.log(x/y)
fig,axs=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
axs[0].scatter(x,y,s=16,color=blue,alpha=.8)
lim=[min(x.min(),y.min())*.8,max(x.max(),y.max())*1.2]
axs[0].plot(lim,lim,color='0.4',ls='--',lw=1)
axs[0].set(xscale='log',yscale='log',xlim=lim,ylim=lim,xlabel='Original label (Å²/ns)',ylabel='Replica label (Å²/ns)')
axs[1].hist(d,bins=15,color=blue,alpha=.8)
axs[1].axvline(0,color='0.4',ls='--',lw=1)
axs[1].set(xlabel='ln(original / replica)',ylabel='Systems')
axs[1].text(.98,.95,f'n = {len(x)}\n'+r'$\sigma_{\mathrm{run}}$'+f' = {d.std(ddof=1)/np.sqrt(2):.2f}',ha='right',va='top',transform=axs[1].transAxes)
fig.savefig('paper/figures/fig_noise_replicas.pdf');fig.savefig('paper/figures/fig_noise_replicas.png',dpi=180)
