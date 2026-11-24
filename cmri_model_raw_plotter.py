#!/usr/bin/env python3
"""Replot cmri_model H200 and duplication figures from source CSV/JSON.

This script NEVER reads model weights, reruns inference, or changes source files.
A panel is OMITTED if the measurement cannot be resolved unambiguously; review
provenance_manifest.json and missing_inputs.txt before using plots in slides.
"""
import argparse, csv, json, math, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from docx import Document
from docx.shared import Pt, Inches

C = ['#f3d98d','#e9ad5d','#db875c','#c76b66','#a4627e','#755271']
INK='#292625'; LIGHT='#f9f7f3'; GRAY='#716d67'
plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.labelcolor':INK,'text.color':INK})

ALIASES={
 'updates':['num_updates','update','updates','step','global_step','optimizer_step','n_updates'],
 'epoch':['epoch','epochs'],
 'loss':['loss','total_loss','train_loss','valid_loss','validation_loss','total_validation_loss'],
 'masked_acc':['masked_acc','masked_accuracy','masked_token_accuracy','token_accuracy','accuracy'],
 'element':['element','atom','atom_type','token','token_name','symbol','class','class_label'],
 'support':['n_masked','count','support','n','total','num_masked'],
 'noisy':['noisy_coord_rmse','input_coord_rmse','coord_rmse_noisy','noisy_rmse','rmse_noisy','corrupted_coord_rmse','before_rmse','input_rmse'],
 'denoised':['reconstructed_coord_rmse','predicted_coord_rmse','coord_rmse_pred','denoised_rmse','rmse_denoised','output_rmse','after_rmse','model_coord_rmse'],
 'dataset':['dataset','task','endpoint','property'],
 'method':['representation','feature','features','model','method','input_type','encoder'],
 'rmse':['rmse','test_rmse','rmse_test','scaffold_rmse','rmse_mean'],
 'split':['seed','random_seed','split','fold'],
 'multiplicity':['multiplicity','copies','n_records','record_count','frequency','num_records','count_per_smiles'],
 'sample_group':['group','source','sampling','dataset','sample_type','condition'],
}

def key(s): return re.sub('[^a-z0-9]','',str(s).lower())
def column(df,names):
  lookup={key(x):x for x in df.columns}
  for x in names:
    if key(x) in lookup:return lookup[key(x)]
  return None

def alias(df,name): return column(df,ALIASES[name])
def read_csv(path):
  if not path or not Path(path).is_file(): return None
  try:
    d=pd.read_csv(path,low_memory=False)
    return d if not d.empty else None
  except Exception:return None

def numeric(d,col): return pd.to_numeric(d[col],errors='coerce')
def save(fig,path):
  fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',facecolor='white')
  fig.savefig(path.with_suffix('.png'),dpi=230,bbox_inches='tight',facecolor='white')
  plt.close(fig)

def configure(ax,title,label):
  ax.set_title(title,loc='left',fontsize=11,fontweight='bold',pad=12)
  ax.text(-.06,1.12,label,transform=ax.transAxes,fontsize=12,fontweight='bold')
  ax.grid(axis='y',alpha=.18,zorder=0)
  ax.set_axisbelow(True)

def flat_json(v,p=''):
  if isinstance(v,dict):
    for k,w in v.items(): yield from flat_json(w,f'{p}.{k}' if p else k)
  elif isinstance(v,(int,float)) and not isinstance(v,bool):yield p,float(v)

def metrics(path):
  if not path or not Path(path).is_file():return {}
  try:return dict(flat_json(json.loads(Path(path).read_text())))
  except Exception:return {}

def choose_metric(measures,include,exclude=()):
  matching=[(k,v) for k,v in measures.items() if all(t in key(k) for t in include) and not any(t in key(k) for t in exclude) and np.isfinite(v)]
  return matching[0] if len(matching)==1 else None

def get_training(d):
  if d is None:return None
  x=alias(d,'updates') or alias(d,'epoch'); y=alias(d,'loss')
  if not (x and y):return None
  data=pd.DataFrame({'x':numeric(d,x),'y':numeric(d,y)}).dropna().sort_values('x')
  return (data,x,y) if len(data)>=2 else None

def plot_training(ax,train,valid,manifest):
  vals=[]
  for label,path,color in [('Training',train,C[3]),('Validation',valid,C[5])]:
    df=read_csv(path); series=get_training(df)
    if series:
      data,xc,yc=series
      ax.plot(data.x,data.y,label=label,color=color,lw=1.8)
      vals.append((label,xc,yc,len(data)))
      manifest['panels']['training_'+label.lower()]={'source':str(path),'x_col':xc,'y_col':yc,'n_rows':len(data)}
  if not vals: raise ValueError('H200 training history: unrecognized columns')
  ax.set_xlabel('Optimizer updates (or epoch, as recorded)')
  ax.set_ylabel('Logged total loss (source-defined)')
  ax.legend(frameon=False,fontsize=8)
  configure(ax,'H200 training / validation history','a')
  if len(vals)==2 and vals[0][1]!=vals[1][1]:
    ax.text(.02,.02,'Caution: x-axis column names differ; check alignment',transform=ax.transAxes,fontsize=7,color='#b34c3d')

def plot_atom(ax,path,manifest):
  d=read_csv(path)
  if d is None:raise ValueError('atom class CSV missing')
  e=alias(d,'element'); a=alias(d,'masked_acc'); n=alias(d,'support')
  if not(e and a):raise ValueError(f'atom class columns unrecognized: {list(d.columns)}')
  z=pd.DataFrame({'el':d[e].astype(str),'acc':numeric(d,a)})
  if n:z['n']=numeric(d,n)
  z=z.dropna(subset=['acc'])
  if z.acc.max()<=1.001:z.acc*=100
  z=z[(z.acc>=0)&(z.acc<=100)]
  z=z[~z.el.str.match(r'^\[|^<',case=False)]
  if len(z)<2:raise ValueError('atom accuracy has too few valid elements')
  order=['C','N','O','S','F','Cl','Br','I','P']
  z['ord']=z.el.map(lambda s:order.index(s) if s in order else 99)
  z=z.sort_values(['ord','el'])
  bars=ax.bar(z.el,z.acc,color=[C[min(i,len(C)-1)] for i in range(len(z))],edgecolor='white',zorder=2)
  if 'n' in z:
    for b,v in zip(bars,z.n):
      if pd.notna(v):ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{int(v):,}',ha='center',va='bottom',fontsize=7)
  ax.set_ylim(0,110);ax.set_ylabel('Masked atom accuracy (%)')
  configure(ax,'Element-wise masked atom recovery (validation 10K)','b')
  manifest['panels']['per_element']={'source':str(path),'element_col':e,'accuracy_col':a,'count_col':n,'n_elements':len(z)}

def plot_coord(ax,path,summary,manifest):
  d=read_csv(path); pairs=[]
  if d is not None:
    a=alias(d,'noisy'); b=alias(d,'denoised')
    if a and b:
      q=pd.DataFrame({'noisy':numeric(d,a),'out':numeric(d,b)}).dropna()
      q=q[(q.noisy>=0)&(q.out>=0)]
      if not q.empty:
        # RMS aggregate of per-record RMSE, explicitly NOT atom-weighted global RMSE.
        pairs=[float(np.sqrt(np.mean(q.noisy**2))),float(np.sqrt(np.mean(q.out**2)))]
        manifest['panels']['geometry']={'source':str(path),'columns':[a,b],'n_pairs':len(q),'aggregation':'RMS of per-record RMSE, not atom-weighted global RMSE'}
  if not pairs:
    m=metrics(summary)
    a=choose_metric(m,['noisy','rmse']) or choose_metric(m,['input','coord','rmse'])
    b=choose_metric(m,['denois','rmse']) or choose_metric(m,['pred','coord','rmse']) or choose_metric(m,['recover','rmse'])
    if a and b:
      pairs=[a[1],b[1]]
      manifest['panels']['geometry']={'source':str(summary),'keys':[a[0],b[0]],'aggregation':'summary provided directly'}
  if not pairs:raise ValueError('No unambiguous paired noisy/recovered coordinate RMSE columns/summary keys')
  ax.bar(['Noisy coordinates','Recovered coordinates'],pairs,color=[C[2],C[4]],width=.58)
  for i,v in enumerate(pairs):ax.text(i,v+max(pairs)*.03,f'{v:.3f} Å',ha='center',fontweight='bold')
  if pairs[0]>0:
    pct=100*(1-pairs[1]/pairs[0]);ax.text(.5,.94,f'{pct:.1f}% lower coordinate RMSE',transform=ax.transAxes,ha='center',fontsize=10,fontweight='bold')
  ax.set_ylim(0,max(pairs)*1.26);ax.set_ylabel('Coordinate RMSE (Å)')
  configure(ax,'3D coordinate denoising','c')

def plot_global_acc(ax,train_sum,valid_sum,manifest):
  pairs=[]
  for label,p in [('Train',train_sum),('Validation',valid_sum)]:
    m=metrics(p)
    choices=[(k,v) for k,v in m.items() if ('maskedacc' in key(k) or 'maskedtokenaccuracy' in key(k)) and 'per' not in key(k)]
    if len(choices)!=1:continue
    k,v=choices[0];val=v*100 if v<=1 else v
    if val<=100:
      pairs.append((label,val))
      manifest['panels']['global_acc_'+label.lower()]={'source':str(p),'key':k,'value_percent':val}
  if not pairs:raise ValueError('No unique global masked accuracy key in summary JSON (per-element plot still available)')
  ax.bar([p[0] for p in pairs],[p[1] for p in pairs],color=[C[2],C[4]][:len(pairs)],width=.5)
  for i,(_,v) in enumerate(pairs):ax.text(i,v+1,f'{v:.2f}%',ha='center',fontweight='bold')
  ax.set_ylim(0,110);ax.set_ylabel('Masked atom accuracy (%)')
  configure(ax,'Overall masked atom reconstruction','d')

def ds_name(s):
  s=str(s).lower()
  if 'freesolv' in s or 'sampl' in s:return 'FreeSolv'
  if 'lipophilicity' in s or 'lipo' in s:return 'Lipophilicity'
  if 'esol' in s or 'delaney' in s:return 'ESOL'
  return None

def plot_bench(path,out,manifest):
  d=read_csv(path)
  if d is None:raise ValueError('benchmark CSV missing')
  dc=alias(d,'dataset');mc=alias(d,'method');rc=alias(d,'rmse')
  if not (dc and mc and rc):raise ValueError(f'benchmark columns unrecognized: {list(d.columns)}')
  d=d.copy();d['dataset_name']=d[dc].map(ds_name);d['metric']=numeric(d,rc);d=d.dropna(subset=['dataset_name','metric'])
  if d.empty:raise ValueError('no ESOL/FreeSolv/Lipophilicity benchmark rows')
  fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
  result=[]
  for i,(name,ax) in enumerate(zip(['ESOL','FreeSolv','Lipophilicity'],axes)):
    q=d[d.dataset_name==name]; ag=q.groupby(mc)['metric'].agg(['mean','std','count']).sort_values('mean')
    if ag.empty:ax.text(.5,.5,'No source rows',ha='center');ax.axis('off');continue
    names=list(ag.index); means=ag['mean'].values
    ax.barh(names,means,color=[C[j%len(C)] for j in range(len(names))],height=.6)
    if (ag['count']>1).any():ax.errorbar(means,range(len(names)),xerr=ag['std'].fillna(0).values,fmt='none',ecolor='#35302b',capsize=3)
    ax.invert_yaxis();ax.set_xlabel('Scaffold-held-out RMSE (source metric)');ax.set_title(f'{"abc"[i]}   {name}',loc='left',weight='bold')
    for j,v in enumerate(means):ax.text(v+max(means)*.025,j,f'{v:.3f}',va='center',fontsize=8)
    ax.set_xlim(0,max(means)*1.28);ax.grid(axis='x',alpha=.15)
    result.extend([{'dataset':name,'method':nm,'mean_rmse':float(ag.loc[nm,'mean']),'sd_rmse':float(ag.loc[nm,'std']) if np.isfinite(ag.loc[nm,'std']) else None,'n_rows':int(ag.loc[nm,'count'])} for nm in names])
  p=out/'cmri_model_downstream_benchmarks';save(fig,p)
  pd.DataFrame(result).to_csv(out/'downstream_aggregated.csv',index=False)
  manifest['panels']['downstream']={'source':str(path),'dataset_col':dc,'method_col':mc,'rmse_col':rc,'n_rows':len(d),'note':'SD computed over rows; check that rows are independent scaffold splits and no duplicates'}
  return p

def plot_h200(cfg,out,manifest):
  fig,axes=plt.subplots(2,2,figsize=(13.8,9.5),layout='constrained')
  steps=[('history',plot_training, (cfg['h200_train_history'],cfg['h200_validation_history'])),
         ('per_element',plot_atom,(cfg['atom_class_csv'],)),
         ('geometry',plot_coord,(cfg['molecule_diagnostics_csv'],cfg['valid_summary_json'])),
         ('global_accuracy',plot_global_acc,(cfg['train_summary_json'],cfg['valid_summary_json']))]
  valid=0
  for ax,(name,fn,vals) in zip(axes.ravel(),steps):
    try:fn(ax,*vals,manifest);valid+=1
    except Exception as e:
      ax.axis('off');ax.text(.5,.5,f'{name}: source not plottable\n{str(e)[:95]}',ha='center',va='center',wrap=True)
      manifest['missing'].append(f'{name}: {e}')
  if valid==0:plt.close(fig);return None
  fig.suptitle('cmri_model | H200 general pretraining: source-derived results',fontsize=16,fontweight='bold')
  p=out/'cmri_model_h200_main_results';save(fig,p)
  return p

def plot_duplicate(cfg,out,manifest):
  # Only source-derived data. Each panel is independently optional.
  fig,axs=plt.subplots(1,3,figsize=(16.8,5.3),layout='constrained')
  valid=0
  # Panel a: per-SMILES counts can be raw frequencies, one per molecular identity
  d=read_csv(cfg.get('duplicate_multiplicity_csv'))
  try:
    if d is None:raise ValueError('multiplicity source CSV missing')
    col=alias(d,'multiplicity')
    if col is None:raise ValueError(f'no multiplicity column: {list(d.columns)}')
    q=numeric(d,col).dropna();q=q[q>=1].astype(int)
    if q.empty:raise ValueError('no positive molecule frequencies')
    bins=[1,2,3,4,6,11,21]; labels=['1','2','3','4–5','6–10','11–20','>20']
    id_freq=[(q==1).sum(),(q==2).sum(),(q==3).sum(),((q>=4)&(q<=5)).sum(),((q>=6)&(q<=10)).sum(),((q>=11)&(q<=20)).sum(),(q>20).sum()]
    raw_freq=[int((q[(q==1)]).sum()),int(q[(q==2)].sum()),int(q[(q==3)].sum()),int(q[(q>=4)&(q<=5)].sum()),int(q[(q>=6)&(q<=10)].sum()),int(q[(q>=11)&(q<=20)].sum()),int(q[q>20].sum())]
    x=np.arange(len(labels));w=.38
    axs[0].bar(x-w/2,100*np.array(raw_freq)/sum(raw_freq),w,color=C[3],label='Raw record exposure')
    axs[0].bar(x+w/2,100*np.array(id_freq)/sum(id_freq),w,color=C[0],label='Uniform unique molecules')
    axs[0].set_xticks(x,labels);axs[0].set_ylabel('Sampled observations (%)');axs[0].legend(fontsize=7,frameon=False)
    configure(axs[0],'How duplication changes exposure','a');valid+=1
    manifest['panels']['duplicate_exposure']={'source':str(cfg['duplicate_multiplicity_csv']),'count_column':col,'identities':len(q),'records':int(q.sum())}
  except Exception as e:axs[0].axis('off');axs[0].text(.5,.5,'Source needed:\nper-identity record frequencies',ha='center');manifest['missing'].append('duplicate exposure: '+str(e))
  # Panel b: descriptor distributions must have a group (raw vs dedup) and descriptor columns
  d=read_csv(cfg.get('duplicate_descriptors_csv'))
  try:
    if d is None:raise ValueError('descriptor source CSV missing')
    group=alias(d,'sample_group')
    if group is None:raise ValueError('no raw/dedup group column')
    descriptor=column(d,['logp','clogp','heavy_atoms','tpsa','molecular_weight'])
    if descriptor is None:raise ValueError('no known numeric descriptor')
    grp=d[group].astype(str).str.lower();raw=d[grp.str.contains('raw')];unique=d[grp.str.contains('dedup|unique')]
    if raw.empty or unique.empty:raise ValueError('no raw and unique group values')
    r=numeric(raw,descriptor).dropna();u=numeric(unique,descriptor).dropna()
    lo=min(r.min(),u.min());hi=max(r.max(),u.max());bins=np.linspace(lo,hi,28)
    axs[1].hist(r,bins=bins,density=True,color=C[3],alpha=.55,label='Raw records')
    axs[1].hist(u,bins=bins,density=True,color=C[0],alpha=.65,label='Unique identities')
    axs[1].set_xlabel(descriptor);axs[1].set_ylabel('Density');axs[1].legend(fontsize=8,frameon=False)
    configure(axs[1],'Physicochemical distribution','b');valid+=1
    manifest['panels']['duplicate_descriptors']={'source':str(cfg['duplicate_descriptors_csv']),'descriptor':descriptor,'group_col':group,'n_raw':len(r),'n_unique':len(u)}
  except Exception as e:axs[1].axis('off');axs[1].text(.5,.5,'Source needed:\nraw vs dedup descriptor records',ha='center');manifest['missing'].append('duplicate descriptors: '+str(e))
  # Panel c: CI table with labels, estimate, lower, upper (no synthetic uncertainty)
  d=read_csv(cfg.get('duplicate_smd_ci_csv'))
  try:
    if d is None:raise ValueError('SMD CI source CSV missing')
    c_label=column(d,['descriptor','property','metric','feature']);c_est=column(d,['smd','standardized_mean_difference','difference','estimate'])
    c_low=column(d,['ci_lower','ci_low','lower','low','lo']);c_up=column(d,['ci_upper','ci_high','upper','high','hi'])
    if not all((c_label,c_est,c_low,c_up)):raise ValueError(f'SMD CI columns unrecognized: {list(d.columns)}')
    v=d[[c_label,c_est,c_low,c_up]].copy();
    for c in [c_est,c_low,c_up]:v[c]=pd.to_numeric(v[c],errors='coerce')
    v=v.dropna();v=v[(v[c_low]<=v[c_est])&(v[c_est]<=v[c_up])]
    if v.empty:raise ValueError('no valid CI rows')
    y=np.arange(len(v));axs[2].errorbar(v[c_est],y,xerr=[v[c_est]-v[c_low],v[c_up]-v[c_est]],fmt='o',color=C[4],capsize=3)
    axs[2].axvline(0,ls='--',color=GRAY);axs[2].set_yticks(y,list(v[c_label]));axs[2].invert_yaxis();axs[2].set_xlabel('Standardized difference (unique − raw)')
    configure(axs[2],'Descriptor shifts (bootstrap 95% CI)','c');valid+=1
    manifest['panels']['duplicate_smd']={'source':str(cfg['duplicate_smd_ci_csv']),'columns':[c_label,c_est,c_low,c_up],'n_features':len(v)}
  except Exception as e:axs[2].axis('off');axs[2].text(.5,.5,'Source needed:\nfeature-level SMD + 95% CI',ha='center');manifest['missing'].append('duplicate SMD: '+str(e))
  if valid==0:plt.close(fig);return None
  fig.suptitle('cmri_model | General dataset duplicate-effect audit',fontsize=16,fontweight='bold')
  p=out/'cmri_model_duplicate_audit';save(fig,p);return p

def make_doc(out,manifest):
  doc=Document();sec=doc.sections[0];sec.top_margin=Inches(.8);sec.bottom_margin=Inches(.7)
  doc.add_heading('cmri_model — H200 & Dataset Audit',0)
  doc.add_paragraph('Source-derived figure explanations and methodological limitations',style='Subtitle')
  doc.add_heading('H200 general pretraining',1)
  doc.add_paragraph('The general model was scaled from the initial T4 batch-size-4 setup to the H200 effective-batch-448 setup. The following results describe the H200 checkpoint evaluation, not an independently measured difference attributable solely to batch size.')
  doc.add_paragraph('Training history: assess whether training and validation loss follow consistent trends, noting that their sampling/aggregation schedules may differ. A declining curve does not prove transferability.')
  doc.add_paragraph('Masked atom accuracy: overall reconstruction can look strong because C is common. Element-wise results check N, O and other classes separately; uncertainty for rare Br/I counts should be acknowledged.')
  doc.add_paragraph('Coordinate denoising: report the RMSE and relative reduction only when paired noisy vs recovered measurements are available from the same evaluation. Reduced coordinate error demonstrates reconstruction, not prediction of RNA-delivery activity.')
  doc.add_heading('General-data duplication audit',1)
  doc.add_paragraph('The earlier project presentation reports 18.84 million training records, 13.27 million distinct raw SMILES, and 29.55% duplicate excess. These values are historical slide-reported metrics, not recomputed by this script unless the underlying audit table is found.')
  doc.add_paragraph('Exposure: repeated raw identities receive more sampling weight. Distribution: similar descriptor histograms can coexist with altered frequencies for individual scaffolds. Confidence intervals: observed small standardized changes are exploratory, not formal chemical equivalence.')
  doc.add_paragraph('We retained the completed checkpoint and prioritized lipid-specific continual pretraining. This is a resource/prioritization decision; it does not establish that deduplication has no effect on generalization.')
  doc.add_heading('Downstream property transfer (separate figure)',1)
  doc.add_paragraph('ESOL, FreeSolv, and Lipophilicity test whether frozen molecular representations transfer to experimental physicochemical property prediction with a simple supervised readout. They do not directly measure mRNA delivery, and comparisons to RDKit/Morgan must use matched splits and alpha tuning.')
  doc.add_heading('Source provenance and missing panels',1)
  for k,v in manifest.get('panels',{}).items():
    p=doc.add_paragraph(style='List Bullet');p.add_run(k+': ').bold=True;p.add_run(str(v.get('source','Unknown source')))
  if manifest['missing']:
    doc.add_heading('Missing or ambiguous source fields',2)
    for m in manifest['missing']:doc.add_paragraph(m,style='List Bullet')
  doc.add_paragraph('The script never fabricates missing metrics. Review provenance_manifest.json before presenting figures as H200 results.')
  target=out/'cmri_model_figure_justification.docx';doc.save(target)
  return target

def audit_paths(cfg,out):
  rows=[]
  for k,v in cfg.items():
    if not k.endswith(('_csv','_json','_history')):continue
    p=Path(v) if v else None
    exists=bool(p and p.is_file())
    cols=''
    if exists and p.suffix.lower()=='.csv':
      try:cols=' | '.join(pd.read_csv(p,nrows=0).columns)
      except Exception as e:cols='READ ERROR: '+str(e)
    if exists and p.suffix.lower()=='.json':
      try:cols=' | '.join(list(metrics(p))[:35])
      except Exception:pass
    rows.append({'input':k,'path':str(p) if p else '','exists':exists,'columns_or_keys':cols})
  pd.DataFrame(rows).to_csv(out/'source_schema_audit.csv',index=False)
  return rows

def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--config',type=Path,help='JSON override of expected source paths')
  parser.add_argument('--out',type=Path,default=Path('/home/ypan/heart_liver/model_train/heart-molecular-pretraining/summary_figures'))
  parser.add_argument('--inspect-only',action='store_true',help='Only inventory sources / columns')
  args=parser.parse_args()
  base=Path('/project/yli4/ypan/heart_evaluation/general_audit_10k_v2')
  ck=base/'checkpoint_evaluation'; chem=base/'chemistry_slide_evidence'
  cfg={
    'h200_train_history':str(base/'training_history/H200_training_history.csv'),
    'h200_validation_history':str(base/'training_history/H200_validation_history.csv'),
    'atom_class_csv':str(ck/'full_10k_batch8/atom_token_class_accuracy.csv'),
    'molecule_diagnostics_csv':str(ck/'full_10k_batch8/molecule_diagnostics.csv'),
    'train_summary_json':str(ck/'train_10k_batch8/evaluation_summary.json'),
    'valid_summary_json':str(ck/'full_10k_batch8/evaluation_summary.json'),
    'benchmark_csv':str(chem/'independent_property_benchmarks/benchmark_all_runs.csv'),
    'duplicate_multiplicity_csv':'', 'duplicate_descriptors_csv':'','duplicate_smd_ci_csv':'',
  }
  if args.config:cfg.update(json.loads(args.config.read_text()))
  out=args.out.expanduser();out.mkdir(parents=True,exist_ok=True)
  audited=audit_paths(cfg,out)
  if args.inspect_only:
    print('Saved',out/'source_schema_audit.csv')
    for r in audited:print(r['input'],r['exists'],r['columns_or_keys'][:160])
    return
  manifest={'model_name':'cmri_model','panels':{},'missing':[], 'schema_audit':str(out/'source_schema_audit.csv'),'source_note':'Input data are read-only; no model execution performed.'}
  h=plot_h200(cfg,out,manifest);dup=plot_duplicate(cfg,out,manifest)
  try:ds=plot_bench(cfg['benchmark_csv'],out,manifest)
  except Exception as e:ds=None;manifest['missing'].append('Downstream: '+str(e))
  doc=make_doc(out,manifest)
  manifest['outputs']={'h200_svg':str(h.with_suffix('.svg')) if h else None,'duplicate_svg':str(dup.with_suffix('.svg')) if dup else None,'downstream_svg':str(ds.with_suffix('.svg')) if ds else None,'notes_docx':str(doc)}
  (out/'provenance_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
  (out/'missing_inputs.txt').write_text('\n'.join(manifest['missing'])+'\n')
  print(json.dumps(manifest['outputs'],indent=2))
  print('Missing:',len(manifest['missing']),'; inspect',out/'missing_inputs.txt')
if __name__=='__main__':main()
