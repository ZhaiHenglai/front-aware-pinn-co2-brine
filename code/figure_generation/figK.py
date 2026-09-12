"""K archived-field plots, using the established legacy field-map layout."""
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared import ROOT,OUT,SRC,H,save,capture

plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
 'mathtext.fontset':'dejavuserif','font.size':8,'axes.linewidth':.7})
base=H.parent/'PINN12_723_clean'
folder=base/'evaluation_supplement_20260905T173006Z/by_run/complete/complete_seed0/snapshots/complete'
metadata=json.loads(capture(folder/'snapshot_metadata.json').read_text())
assert metadata['run_id']=='M7_BASE_LOO-NONE_seed0'
probe=json.loads((ROOT/'04_checks/K_CHECKPOINT_PROBE.json').read_text())
records=[];data={}
for tag in ['early','middle','final']:
    path=folder/tag/'snapshot_grid.npz';local=capture(path)
    data[tag]={k:v.copy() for k,v in np.load(local).items()}
    d=data[tag];X,Y=d['X'],d['Y']
    mask=(X**2+Y**2>.25)&((X-5)**2+(Y-5)**2>.25)
    assert X.shape==Y.shape==(500,500)
    for key in ['S_true','S_pred','p_true','p_pred']:
        assert np.array_equal(np.isfinite(d[key]),mask)
    expected=next(v['time_seconds'] for v in metadata['times'] if v['tag']==tag)
    assert float(d['time_seconds'])==expected
    records.append({'tag':tag,'time_seconds':expected,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'preserved_array':local.name,'valid_points':int(mask.sum())})
final=next(s for s in probe['sources'] if s['snapshot']==str(folder/'final/snapshot_grid.npz'))
assert final['snapshot_sha256']==records[-1]['sha256']
assert hashlib.sha256((base/'runs/complete/complete_seed0/final_checkpoint.pt').read_bytes()).hexdigest()==final['checkpoint_sha256']
checks={'sources':records,'final_checkpoint_and_snapshot_match_prior_probe':True,'rows':[]}

def render(name,tags):
    rows=[('S',t) for t in tags]+[('p','final')];n=len(rows)
    fig=plt.figure(figsize=(7.48,2*n+.3 if n==2 else 2*n))
    gs=fig.add_gridspec(n,6,width_ratios=[1,1,.032,.135,1,.032],wspace=.075,hspace=.15,
                       left=.092,right=.955,top=.94 if n==2 else .97,bottom=.15 if n==2 else .065)
    for i,(kind,tag) in enumerate(rows):
        d=data[tag];factor=1 if kind=='S' else 1e6
        true=d[kind+'_true']/factor;pred=d[kind+'_pred']/factor;err=pred-true
        bounds=(0,.8) if kind=='S' else (10,11.4);limit=.25 if kind=='S' else 1.2
        assert np.nanmin(true)>=bounds[0] and np.nanmax(true)<=bounds[1]
        assert np.nanmin(pred)>=bounds[0] and np.nanmax(pred)<=bounds[1]
        assert np.nanmax(abs(err))<limit
        axs=[fig.add_subplot(gs[i,j]) for j in [0,1,4]];ims=[]
        for j,(ax,a) in enumerate(zip(axs,[true,pred,err])):
            cmap=plt.get_cmap('RdBu_r' if j==2 else 'viridis').copy();cmap.set_bad('white')
            lo,hi=(-limit,limit) if j==2 else bounds
            ims.append(ax.pcolormesh(d['X'],d['Y'],np.ma.masked_invalid(a),cmap=cmap,vmin=lo,vmax=hi,
                                     rasterized=True,shading='auto'))
            ax.set(aspect='equal',xlim=(0,5),ylim=(0,5),xticks=[0,2.5,5],yticks=[0,2.5,5])
            if i==n-1:ax.set_xlabel('$x$ (m)',fontsize=7.5)
            else:ax.set_xticklabels([])
            if j==0:ax.set_ylabel('$y$ (m)',fontsize=7.5)
            else:ax.set_yticklabels([])
            ax.tick_params(labelsize=6.8,length=2.2)
            if i==0:ax.set_title(['reference','complete prediction','signed error'][j],fontsize=7.8,pad=3)
        qty='$S_{\\mathrm{CO_2}}$' if kind=='S' else '$p$ (MPa)'
        axs[0].text(-.40,.5,f"$\\bf{{{chr(97+i)}}}$  {qty},  $t={float(d['time_seconds'])/86400:.3f}$ d",
                    transform=axs[0].transAxes,rotation=90,va='center',ha='center',fontsize=7.6)
        for j,im in [(2,ims[1]),(5,ims[2])]:
            cb=fig.colorbar(im,cax=fig.add_subplot(gs[i,j]))
            if j==5:cb.set_ticks(np.linspace(-limit,limit,5))
            cb.ax.tick_params(labelsize=6.2,length=2);cb.outline.set_linewidth(.5)
        checks['rows'].append({'figure':name,'quantity':kind,'tag':tag,'time_seconds':float(d['time_seconds']),
          'reference_min':float(np.nanmin(true)),'reference_max':float(np.nanmax(true)),
          'prediction_min':float(np.nanmin(pred)),'prediction_max':float(np.nanmax(pred)),
          'max_abs_error':float(np.nanmax(abs(err))),'error_limit':limit,'clipped_points':int(np.sum(abs(err)>limit))})
    fig.text(.5,.007,'K, seed 0; error = prediction − reference. Full displayed error ranges; scales differ from H.',ha='center',fontsize=6.3)
    save(fig,name);plt.close(fig)

render('K_fields_main',['final'])
render('K_fields_complete',['early','middle','final'])
pd.DataFrame(checks['rows']).to_csv(SRC/'K_field_ranges.csv',index=False)
(ROOT/'04_checks/BATCH23_K_FIELD_CHECK.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
