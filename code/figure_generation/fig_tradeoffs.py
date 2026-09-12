"""Paired descriptive trade-offs from verified H/K records; no training."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shared import ROOT,SRC,save,capture

plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
 'mathtext.fontset':'dejavuserif','font.size':8,'axes.linewidth':.7,
 'axes.spines.top':False,'axes.spines.right':False})
def read(rel):return pd.read_csv(capture(ROOT/'06_evidence'/rel))
h=read('H_tables_v2/by_seed.csv');h=h[h.study=='L0_L6'].set_index(['role','seed'])
k=read('secondary_K_v1/K_by_seed.csv').set_index(['role','seed'])
c=read('common_contours_v1/by_run.csv').set_index(['case','run_id'])
inv=read('common_inventory_v2/by_seed.csv');inv=inv[(inv.grid==256)&(inv.mapping=='nearest')].set_index(['case','role','seed'])
expected_h=read('H_tables_v2/paired_effects.csv');expected_h=expected_h[expected_h.study=='L0_L6'].set_index(['role','metric'])
expected_k=read('secondary_K_v1/K_paired_effects.csv').set_index(['role','metric'])
expected_c=read('common_contours_v1/paired_effects.csv').set_index(['case','deletion'])
expected_i=read('common_inventory_v2/paired.csv');expected_i=expected_i[(expected_i.grid==256)&(expected_i.mapping=='nearest')].set_index(['case','seed'])
seeds=[0,1,2,42];markers=['o','s','^','v']
roles={'H':['PLAIN_TWONET','FRONT_PLUME','FV'],'K':['no_representation','no_front_supervision','no_local_fv']}
metrics=['contour','saturation','local','inventory']
def value(case,role,seed,metric):
    table=h if case=='H' else k
    r=table.loc[role,seed]
    if metric=='contour':return float(c.loc[(case,r.run_id),'conditional_chamfer_m'])
    if metric=='inventory':return float(inv.loc[(case,'complete' if role in ['M7','complete'] else 'no_local_fv',seed),'error'])
    key={'saturation':'S_rel_l2','local':'local_FV_CO2_RMSE' if case=='H' else 'local_FV_residual','band':'S_RMSE_front_band' if case=='H' else 'band_RMSE'}[metric]
    return float(r[key])
paired=[];summary=[]
for metric in metrics:
    for case in ['H','K']:
        for idx in (range(3) if metric in ['contour','saturation'] else [2]):
            role=roles[case][idx];base='M7' if case=='H' else 'complete';vals=[]
            for seed in seeds:
                a=value(case,base,seed,metric);b=value(case,role,seed,metric)
                assert a>0 and np.isfinite(b)
                pct=100*(b-a)/a;vals.append(pct)
                paired.append(dict(case=case,metric=metric,deletion=role,group=idx,seed=seed,complete=a,deleted=b,percent=pct))
                if metric=='inventory':np.testing.assert_allclose(pct,expected_i.loc[(case,seed),'deletion_relative_change_percent'],atol=1e-10)
            mean=np.mean(vals);sd=np.std(vals,ddof=1)
            if metric!='inventory':
                if metric=='contour':e=expected_c.loc[case,role]
                else:
                    key='S_rel_l2' if metric=='saturation' else ('local_FV_CO2_RMSE' if case=='H' else 'local_FV_residual')
                    e=(expected_h if case=='H' else expected_k).loc[role,key]
                np.testing.assert_allclose([mean,sd],[e.mean_percent,e.SD_percent],atol=1e-10)
            summary.append(dict(case=case,metric=metric,deletion=role,n=4,mean=mean,SD=sd,n_increased=int(np.sum(np.array(vals)>0))))
pd.DataFrame(paired).to_csv(SRC/'HK_component_paired.csv',index=False)
pd.DataFrame(summary).to_csv(SRC/'HK_component_summary.csv',index=False)

# Absolute H trade-off: arrows point from deletion to complete, not vice versa.
fig,axes=plt.subplots(1,2,figsize=(7.48,3.5));fig.subplots_adjust(left=.1,right=.98,bottom=.22,top=.79,wspace=.38)
absolute=[]
for ax,metric,title,letter in zip(axes,['band','inventory'],['Front-band reconstruction','Area-weighted inventory'],'ab'):
    for seed,marker in zip(seeds,markers):
        x=[value('H',role,seed,'local') for role in ['FV','M7']]
        y=[value('H',role,seed,metric) for role in ['FV','M7']]
        ax.annotate('',xy=(x[1],y[1]),xytext=(x[0],y[0]),arrowprops={'arrowstyle':'->','color':'#88939e','lw':.8},zorder=1)
        for j in range(2):ax.scatter(x[j],y[j],marker=marker,s=28,facecolor='white' if j==0 else '#b66b31',edgecolor='#34698c' if j==0 else '#b66b31',zorder=3)
        absolute.append(dict(seed=seed,metric=metric,no_FV_local=x[0],complete_local=x[1],no_FV_error=y[0],complete_error=y[1]))
    for role,color in [('FV','#34698c'),('M7','#b66b31')]:
        x=np.array([value('H',role,s,'local') for s in seeds]);y=np.array([value('H',role,s,metric) for s in seeds])
        ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),fmt='D',ms=5,color=color,capsize=3,lw=1.1,zorder=4)
    ax.set_xlabel('Local FV residual');ax.set_ylabel('Front-band RMSE' if metric=='band' else 'Common inventory error')
    ax.set_title(title,fontsize=8.5,pad=10);ax.text(-.16,1.06,letter,weight='bold',transform=ax.transAxes)
    ax.grid(alpha=.16);ax.margins(.15)
handles=[Line2D([],[],marker='o',ls='',mfc='white',mec='#34698c',label='No FV (M6)'),Line2D([],[],marker='o',ls='',color='#b66b31',label='Complete (M7)'),Line2D([],[],marker='D',color='#555555',label='Mean ± sample SD')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.99),ncol=3,frameon=False,fontsize=7.4)
fig.text(.5,.08,'Arrows: no FV → complete, paired within seed. Shapes: ○ 0, □ 1, △ 2, ▽ 42.',ha='center',fontsize=7)
fig.text(.5,.025,'Four seeds; descriptive endpoints, not a continuous frontier. Inventory is not flux-balance closure.',ha='center',fontsize=6.8)
save(fig,'H_FV_tradeoff');plt.close(fig)
pd.DataFrame(absolute).to_csv(SRC/'H_FV_tradeoff_source.csv',index=False)

# Within-case deletion percentages; H/K are not paired geological replicates.
df=pd.DataFrame(paired)
fig,axes=plt.subplots(2,2,figsize=(7.48,5.7));fig.subplots_adjust(left=.20,right=.98,bottom=.17,top=.89,wspace=.55,hspace=.55)
titles=['Conditional contour Chamfer','Global saturation relative $L_2$','Local FV residual: FV deletion','Common inventory: FV deletion']
for panel,(ax,metric,title) in enumerate(zip(axes.flat,metrics,titles)):
    groups=list(range(3)) if metric in ['contour','saturation'] else [2]
    extent=[0]
    for pos,group in enumerate(groups):
        for case,offset,color,mk in [('H',-.14,'#34698c','o'),('K',.14,'#b66b31','s')]:
            g=df[(df.case==case)&(df.metric==metric)&(df.group==group)].sort_values('seed');v=g.percent.to_numpy();assert len(v)==4
            for j,vv in enumerate(v):ax.scatter(vv,pos+offset+(j-1.5)*.036,marker=mk,s=17,facecolor='white',edgecolor=color,lw=.8,zorder=3)
            ax.errorbar(v.mean(),pos+offset,xerr=v.std(ddof=1),fmt='D',ms=3.5,color=color,capsize=2,lw=1,zorder=4)
            extent.extend([*v,v.mean()-v.std(ddof=1),v.mean()+v.std(ddof=1)])
            if len(groups)==1:ax.text(.98,.90 if case=='H' else .78,f'{case}: {(v>0).sum()}/4 increased',transform=ax.transAxes,ha='right',fontsize=6.8,color=color)
    lo,hi=min(extent),max(extent);pad=(hi-lo)*.1
    ax.set(xlim=(lo-pad,hi+pad),ylim=(len(groups)-.5,-.5),yticks=range(len(groups)),
           yticklabels=['Representation','Front supervision','Local FV'] if len(groups)==3 else ['Local FV'],xlabel='Change after deletion (%)')
    ax.axvline(0,ls='--',lw=.7,color='#777777');ax.grid(axis='y',alpha=.15)
    ax.set_title(title,fontsize=8,pad=12);ax.text(-.18,1.08,chr(97+panel),transform=ax.transAxes,weight='bold')
    ax.tick_params(labelsize=7)
handles=[Line2D([],[],marker=m,ls='',mfc='white',mec=c,label=case) for case,c,m in [('H','#34698c','o'),('K','#b66b31','s')]]
handles.append(Line2D([],[],marker='D',color='#555555',label='Mean ± sample SD'))
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.56,.99),ncol=3,frameon=False,fontsize=7.5)
fig.text(.5,.09,'Four seed pairs per case; positive = higher metric after deletion. Linear panel ranges differ.',ha='center',fontsize=7)
fig.text(.5,.045,'Contour: four post-initial times. Inventory: five times including t = 0, common quadrature/EOS.',ha='center',fontsize=6.8)
fig.text(.5,.012,'H and K are two fixed configurations; local residual is objective-related, not global conservation.',ha='center',fontsize=6.8)
save(fig,'HK_component_comparison');plt.close(fig)
(ROOT/'04_checks/BATCH24_NUMERIC_CHECK.json').write_text(json.dumps({'paired_points':len(paired),'summary_groups':len(summary),'summary':summary,'inventory_grid':256,'mapping':'nearest','n_seeds':4,'submission_ready':False},indent=2)+'\n')
print(pd.DataFrame(summary).to_string(index=False))
