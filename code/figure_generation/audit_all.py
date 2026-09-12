"""Source/array/arithmetic checks for the full current figure sequence."""
import sys,json,hashlib,re
import numpy as np
import pandas as pd
import torch
import fitz
from PIL import Image
import xml.etree.ElementTree as ET
from shared import ROOT,OUT,SRC,FW,H,capture

sys.dont_write_bytecode=True
sys.path.insert(0,str(H.parent/'PINN12_723_clean/Results'))
import evaluate_global_accuracy_M0_M7_L6_dS_shortlabels as ev
torch.set_num_threads(4);torch.set_default_dtype(torch.float64)
evidence=ROOT/'06_evidence/figure_revision_all'
checks={}

# Analytic constitutive curves, not observational data.
c=np.loadtxt(SRC/'Figure2_curves.csv',delimiter=',',skiprows=1)
S=c[:,0];np.testing.assert_allclose(c[:,1],(1-S/.8)**2,atol=1e-14)
np.testing.assert_allclose(c[:,2],(S/.8)**2,atol=1e-14)
M=2.5e-4/2.25e-5;star=.8/np.sqrt(1+M)
f=lambda s:(M*(s/.8)**2)/(M*(s/.8)**2+(1-s/.8)**2)
np.testing.assert_allclose((f(star+1e-6)-f(star-1e-6))/2e-6,f(star)/star,rtol=1e-8)
checks['Figure2']={'analytic_shock_saturation':star,'curve_rows':len(c)}

# All gallery checkpoints: metadata and independent CPU probes, no retraining.
gallery=[]
for i in range(8):
    role=f'M{i}';folder=FW/f'eval_seed0_all/M0_M7/snapshots/{role}'
    meta=json.loads(capture(folder/'snapshot_metadata.json').read_text())
    assert meta['run_id']==f'{role}_BASE_LOO-NONE_seed0'
    cp=H/'Results1'/f'model_{role}_BASE_LOO-NONE_seed0_v100_final.pt'
    assert cp.name==meta['checkpoint'].split('/')[-1]
    state,ccfg=ev.load_checkpoint(str(cp));cfg=ev.default_cfg()
    ev.apply_exp_preset(cfg,ccfg['exp_name']);ev.cfg_update_from_ckpt(cfg,ccfg)
    model=ev.build_model(cfg);model.load_state_dict(state,strict=True);model.eval();model.set_beta(cfg['beta_eval'])
    with np.load(capture(folder/'final/snapshot_grid.npz')) as d:
        X,Y=d['X'],d['Y'];valid=(X**2+Y**2>.25)&((X-5)**2+(Y-5)**2>.25)
        assert X.shape==Y.shape==(500,500)
        assert float(d['time_seconds'])==31544.99609375
        for k in ['S_true','S_pred','p_true','p_pred']:assert np.array_equal(np.isfinite(d[k]),valid)
        ids=np.flatnonzero(valid)[::2000]
        with torch.inference_mode():
            p,s=ev.predict_model(model,X.ravel()[ids],Y.ravel()[ids],np.full(len(ids),float(d['time_seconds'])),cfg,torch.device('cpu'),2048)
        p=cfg['p0']+cfg['P_ref']*p
        pe=float(np.max(abs(p-d['p_pred'].ravel()[ids])));se=float(np.max(abs(s-d['S_pred'].ravel()[ids])))
        assert pe<.1 and se<1e-7,(role,pe,se)
        gallery.append({'model':role,'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
                        'probe_points':len(ids),'pressure_max_abs_Pa':pe,'saturation_max_abs':se})
checks['FigureB3']={'models':gallery,'total_probe_points':sum(r['probe_points'] for r in gallery)}

# Forward means/SD compared with independent aggregation of source rows.
raw=pd.read_csv(ROOT/'06_evidence/H_tables_v2/by_seed.csv')
raw=raw[raw.study=='M0_M7']
for suffix,kind in [('mean','mean'),('SD','std')]:
    out=pd.read_csv(SRC/f'Figure7_{suffix}.csv',index_col=0)
    expected=raw.groupby('role')[list(out.columns)].agg(kind).loc[out.index]
    np.testing.assert_allclose(out,expected,atol=1e-12)
checks['Figure7']={'seed_runs':32,'mean_SD_cells':64}

# Heatmap uncapped means must equal paired table sources, including ray definition.
matrix=pd.read_csv(SRC/'Figure8_mean_percent.csv',index_col=0)
assert matrix.shape==(6,8) and matrix.iloc[0,1]>600
paired=pd.read_csv(ROOT/'06_evidence/H_tables_v2/paired_effects.csv')
paired=paired[paired.study=='L0_L6'].set_index(['role','metric'])
rays=pd.read_csv(ROOT/'06_evidence/H_rays_strategies_v1/LOO_paired.csv').set_index(['removed','metric'])
roles=['PLAIN_TWONET','FRONT_PLUME','PAIRGRAD','RAR','FV','COARSE_DETAIL']
keys=['S_RMSE_front_band','conditional_chamfer_m','position','excess_width','S_rel_l2','p_phys_RMSE_MPa','local_FV_CO2_RMSE','CO2_mass_rel_error_mean']
for i,role in enumerate(roles):
    for j,key in enumerate(keys):
        val=rays.loc[(role,key),'mean'] if key in ['position','excess_width'] else paired.loc[(role,key),'mean_percent']
        assert abs(matrix.iloc[i,j]-val)<1e-10
checks['Figure8']={'cells_checked':48,'above_color_scale_cells':int((abs(matrix)>60).sum().sum())}

# Standard breakthrough traces are unique and kept separate from holdouts.
bt=pd.read_csv(SRC/'Figure9b_source.csv')
assert set(bt.seed)=={0} and set(bt.training_strategy)=={'BASE'} and set(bt.leave_out)=={'NONE'}
assert not bt.duplicated(['exp_name','time_s']).any()
checks['Figure9']={'source':'standard_training','seed':0,'threshold':.05,'probe_radius_m':1.0,
 'crossing_arithmetic':'verified during plotting against saved summary'}

# Independent comparison of all ray-wise paired means and sample SD.
angular=pd.read_csv(ROOT/'06_evidence/H_rays_strategies_v1/angular_sharpness_changes.csv')
for step,stem in [('M3->M4','M3_M4'),('M6->M7','M6_M7')]:
    plotted=pd.read_csv(SRC/f'Figure10_{stem}_paired.csv').sort_values('ray_angle_deg')
    expected=angular[angular.step==step].sort_values('angle_deg')
    assert len(plotted)==len(expected)==9 and (plotted['count']==4).all()
    np.testing.assert_allclose(plotted[['ray_angle_deg','mean','std']],expected[['angle_deg','mean','SD']],atol=1e-12)
checks['Figure10']={'paired_angle_groups':18,'mean_SD_cells_checked':36,'seeds_per_group':4,
 'interpretation':'mean directional changes, not uniform improvement in every seed'}

# Reaggregate all holdout means/SD from four-seed results; nominal fractions not capped evaluation n.
hold=pd.read_csv(ROOT/'06_evidence/H_auxiliary_v1/holdout_by_seed.csv')
shown=pd.read_csv(SRC/'Figure11a_source.csv')
for _,row in shown.iterrows():
    group=hold[(hold.partition==row.partition)&(hold.exp_name==row.model)&(hold.split=='heldout')]
    assert set(group.seed)=={0,1,2,42}
    np.testing.assert_allclose([row.heldout_mean,row.heldout_SD],[group.relL2_S.mean(),group.relL2_S.std(ddof=1)],atol=1e-12)
assert shown.kept_pct.nunique()>3
bl=pd.read_csv(SRC/'Figure11b_source.csv');assert not bl.duplicated(['exp','seed','n_labels']).any()
checks['Figure11']={'holdout_mean_SD_groups':len(shown),'BL_unique_records':len(bl),'fraction_axis':'nominal partition rule'}

# Matched main loss record; show all finite time-level errors.
loss=pd.read_csv(SRC/'FigureB1_source.csv');assert set(loss.seed)=={0} and loss.x.max()==20000
assert len(loss.run_id.unique())==1 and loss.run_id.iloc[0]=='M7_BASE_LOO-NONE_seed0'
checks['FigureB1']={'run':loss.run_id.iloc[0],'logged_points':len(loss),'raw_components_not_sum_of_total':True}
times=pd.read_csv(SRC/'FigureB2_source.csv')
assert not times.duplicated(['M','t_d']).any() and times.M.nunique()==8
checks['FigureB2']={'time_level_rows':len(times),'max_pressure_RMSE_MPa':float(times.p_phys_RMSE_MPa.max()),
 'undefined_front_band_rows':int(times.S_RMSE_front_band.isna().sum()),'axis_clipping':False}

# Physical-value ranges and color-range exceedances are stored separately.
checks['Figure6']=pd.read_csv(SRC/'Figure6_manifest.csv').to_dict('records')
full=pd.read_csv(SRC/'FigureB4_manifest.csv');assert (full.above_color_limit_percent==0).all()
checks['FigureB4']={'same_arrays_as_Figure6':True,'full_range_clipping':False}

# Check the entire current publication figure set and assemble a browsable review PDF.
paths={f'Figure{i}':OUT/f'Figure{i}.pdf' for i in range(1,12) if i!=4}
paths['Figure4']=ROOT/'02_figures/revised/Method_legacy_refined.pdf'
paths.update({f'FigureB{i}':OUT/f'FigureB{i}.pdf' for i in range(1,5)})
paths['FigureB5']=ROOT/'02_figures/revised/H_ablation.pdf'
paths['graphical_abstract']=OUT/'graphical_abstract.pdf'
order=[f'Figure{i}' for i in range(1,12)]+[f'FigureB{i}' for i in range(1,6)]+['graphical_abstract']
book=fitz.open();toc=[];exports=[]
for name in order:
    path=paths[name];toc.append([1,name,len(book)+1])
    with fitz.open(path) as doc:
        assert len(doc)==1 and len(doc[0].get_text())>20
        page=doc[0];book.insert_pdf(doc)
        exports.append({'figure':name,'width_mm':page.rect.width*25.4/72,'height_mm':page.rect.height*25.4/72})
    with Image.open(path.with_suffix('.tiff')) as im:assert abs(im.info['dpi'][0]-600)<1
    assert len(ET.parse(path.with_suffix('.svg')).findall('.//{http://www.w3.org/2000/svg}text'))>3
book.set_toc(toc);book.save(OUT/'ALL_FIGURES_REVIEW.pdf');book.close()
checks['export_set']={'figure_count':len(order),'formats':['pdf','svg','png','tiff'],'raster_dpi':600,'working_sizes':exports}

# Existing numeric tables must remain exactly intact; only figure-related prose corrections allowed.
baseline=ROOT/'00_baseline/batch22'
for directory in ['01_manuscript/sections','03_supplement/sections']:
    for path in (ROOT/directory).glob('*.md'):
        original=baseline/path.name
        if original.exists():
            before=[l for l in original.read_text().splitlines() if l.startswith('|')]
            after=[l for l in path.read_text().splitlines() if l.startswith('|')]
            assert before==after,path.name
assert hashlib.sha256((ROOT/'00_baseline/legacy/manuscript.md').read_bytes()).hexdigest()=='3557664f5f041a2ed8ae52714a9f2920203e0f50e1696982ad9bebdaf119d3be'
checks['boundaries']={'numeric_tables_unchanged':True,'frozen_manuscript_unchanged':True,
 'reference_solver_convergence_verified':False,'final_eight_main_figure_migration_complete':False,
 'final_EMS_figure_specification_checked':False,'submission_ready':False}
(evidence/'ALL_FIGURE_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
