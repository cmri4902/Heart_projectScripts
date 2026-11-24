#!/usr/bin/env python3
"""cmri_model: read-only H200/duplicate/downstream raw-data figure builder.

Usage:
  python cmri_model_raw_plotter.py --inspect-only
  python cmri_model_raw_plotter.py
  python cmri_model_raw_plotter.py --config /path/to/duplicate_config.json

Output: project/summary_figures/*.svg, *.png, figure_notes.md, and optional .docx.
Never loads checkpoints, performs inference, or modifies source data.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path('/home/ypan/heart_liver/model_train/heart-molecular-pretraining')
BASE = Path('/project/yli4/ypan/heart_evaluation/general_audit_10k_v2')
COLORS = ['#F3D98D', '#E9AD5D', '#DB875C', '#C76B66', '#A4627E', '#755271']
plt.rcParams.update({'svg.fonttype':'none', 'font.family':'DejaVu Sans',
                     'font.size':9, 'axes.spines.top':False,
                     'axes.spines.right':False})


def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())


def col(df, choices):
    lookup = {norm(x): x for x in df.columns}
    for c in choices:
        if norm(c) in lookup: return lookup[norm(c)]
    return None


def as_num(df, name): return pd.to_numeric(df[name], errors='coerce')


def csv_read(path):
    if not path or not Path(path).is_file(): return None
    try:
        d = pd.read_csv(path, low_memory=False)
        return d if len(d) else None
    except Exception: return None


def json_read(path):
    if not path or not Path(path).is_file(): return {}
    try: return json.loads(Path(path).read_text())
    except Exception: return {}


def flatten(v, prefix=''):
    if isinstance(v, dict):
        for k, value in v.items():
            yield from flatten(value, (prefix+'.' if prefix else '')+k)
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
        yield prefix, float(v)


def discover_history(base, split):
    folder = base/'training_history'
    if not folder.exists(): return ''
    candidates = [p for p in folder.rglob('*.csv')
                  if 'h200' in p.name.lower()
                  and (('train' in p.name.lower() and 'valid' not in p.name.lower())
                       if split=='train' else 'valid' in p.name.lower())
                  and ('gap' not in p.name.lower())]
    # prefer the explicit training and validation history files
    key = 'training_history' if split=='train' else 'validation_history'
    candidates.sort(key=lambda p: (key not in p.name.lower(), len(p.name)))
    return str(candidates[0]) if candidates else ''


def cfg_defaults():
    ck = BASE/'checkpoint_evaluation'
    chem = BASE/'chemistry_slide_evidence'
    return {
      'h200_train_history': discover_history(BASE, 'train'),
      'h200_validation_history': discover_history(BASE, 'valid'),
      'atom_class_csv': str(ck/'full_10k_batch8/atom_token_class_accuracy.csv'),
      'molecule_diagnostics_csv': str(ck/'full_10k_batch8/molecule_diagnostics.csv'),
      'train_summary_json': str(ck/'train_10k_batch8/evaluation_summary.json'),
      'valid_summary_json': str(ck/'full_10k_batch8/evaluation_summary.json'),
      'benchmark_csv': str(chem/'independent_property_benchmarks/benchmark_all_runs.csv'),
      'duplicate_multiplicity_csv': '',
      'duplicate_descriptors_csv': '',
      'duplicate_smd_ci_csv': ''
    }


def save(fig, root):
    fig.savefig(root.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    fig.savefig(root.with_suffix('.png'), bbox_inches='tight', facecolor='white', dpi=230)
    plt.close(fig)


def set_panel(ax, label, title):
    ax.set_title(title, loc='left', weight='bold', fontsize=11, pad=12)
    ax.text(-.04, 1.11, label, transform=ax.transAxes, weight='bold', fontsize=13)
    ax.grid(alpha=.16, axis='y'); ax.set_axisbelow(True)


def training_plot(ax, config, sources):
    plotted = 0
    for split, color in [('train', COLORS[2]), ('validation', COLORS[5])]:
        path = config['h200_train_history' if split=='train' else 'h200_validation_history']
        d = csv_read(path)
        if d is None: continue
        x = col(d, ['num_updates','update','updates','step','global_step','optimizer_step','epoch'])
        y = col(d, ['loss','total_loss','train_loss','valid_loss','validation_loss'])
        if not x or not y: continue
        z = pd.DataFrame({'x':as_num(d,x),'y':as_num(d,y)}).dropna().sort_values('x')
        if len(z)<2: continue
        ax.plot(z.x,z.y, label=split.capitalize(), color=color, lw=1.7)
        sources[split+'_history']={'file':path,'x':x,'y':y,'n':len(z)}
        plotted += 1
    if not plotted: raise ValueError('H200 history CSV(s) missing or unsupported columns')
    ax.legend(frameon=False)
    ax.set_xlabel('Updates / epochs (source column)'); ax.set_ylabel('Logged loss')
    set_panel(ax,'a','H200 general pretraining history')


def element_plot(ax, config, sources):
    path = config['atom_class_csv']; d = csv_read(path)
    if d is None: raise ValueError('Atom class accuracy CSV missing')
    atom = col(d,['token_symbol','element','atom','atom_type','token_name','symbol'])
    acc = col(d,['accuracy','masked_atom_accuracy','masked_accuracy','masked_acc'])
    count = col(d,['masked_targets','masked_target_count','n_masked','count','support'])
    if not atom or not acc: raise ValueError('Cannot identify element or accuracy columns: '+str(list(d.columns)))
    z = pd.DataFrame({'atom':d[atom].astype(str),'acc':as_num(d,acc)})
    if count: z['n'] = as_num(d,count)
    z = z.dropna(subset=['acc']); z = z[z.atom.isin(['C','N','O','S','F','Cl','Br','I','P'])].copy()
    if z.empty: raise ValueError('No common atom symbols found')
    if z.acc.max()<=1.01: z['acc'] *= 100
    z = z[(z.acc>=0)&(z.acc<=100)]
    order = ['C','N','O','S','F','Cl','Br','I','P']
    z['sort'] = z.atom.map({a:i for i,a in enumerate(order)})
    z = z.sort_values('sort')
    bars = ax.bar(z.atom,z.acc,color=[COLORS[i%len(COLORS)] for i in range(len(z))],width=.72)
    if 'n' in z:
        for b,n in zip(bars,z.n):
            if pd.notna(n): ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.0,f'{int(n):,}',ha='center',fontsize=7)
    ax.set_ylim(0,112); ax.set_ylabel('Accuracy (%)')
    set_panel(ax,'b','Per-element masked atom recovery · validation 10K')
    sources['element_accuracy']={'file':path,'atom':atom,'accuracy':acc,'support':count,'n_classes':len(z)}


def geometry_plot(ax, config, sources):
    path=config['molecule_diagnostics_csv']; d=csv_read(path)
    if d is None: raise ValueError('Molecular diagnostics CSV missing')
    noisy = col(d,['coord_noisy_rmse','noisy_coord_rmse','input_coord_rmse','noisy_rmse'])
    pred = col(d,['coord_pred_rmse','predicted_coord_rmse','reconstructed_coord_rmse','denoised_rmse'])
    if not noisy or not pred: raise ValueError('Paired coordinate RMSE columns missing: '+str(list(d.columns)))
    z = pd.DataFrame({'noisy':as_num(d,noisy),'pred':as_num(d,pred)}).dropna()
    z = z[(z.noisy>=0)&(z.pred>=0)]
    if len(z)==0: raise ValueError('No valid noisy/recovered coordinate pairs')
    # Mean per-molecule RMSE, not atom-weighted global RMSE
    vals = [z.noisy.mean(),z.pred.mean()]
    ax.bar(['Noisy input','Model recovery'],vals,color=[COLORS[1],COLORS[4]],width=.56)
    for i,v in enumerate(vals): ax.text(i,v+max(vals)*.025,f'{v:.4f} Å',ha='center',weight='bold')
    if vals[0]>0:
        ax.text(.5,.94,f'{100*(1-vals[1]/vals[0]):.1f}% lower mean molecule RMSE',
                transform=ax.transAxes,ha='center',weight='bold',fontsize=9)
    ax.set_ylim(0,max(vals)*1.25); ax.set_ylabel('Mean per-molecule RMSE (Å)')
    set_panel(ax,'c','Coordinate reconstruction from noisy input')
    sources['coordinate']={'file':path,'columns':[noisy,pred],'n_molecules':len(z),'aggregation':'mean of per-molecule RMSE'}


def summary_accuracy(summary_path):
    d=json_read(summary_path); choices=[]
    for k,v in flatten(d):
        nk=norm(k)
        if ('maskedatomaccuracy' in nk or 'maskedtokenaccuracy' in nk or 'maskedacc' in nk) and 'majority' not in nk:
            if np.isfinite(v) and 0<=v<=100: choices.append((k,v))
    if not choices:return None
    # deterministic priority; multiple identical terms can still be ambiguous
    choices.sort(key=lambda x:(len(x[0].split('.')),len(x[0])))
    k,v=choices[0]
    return k, v*100 if v<=1 else v


def overall_plot(ax, config, sources):
    vals=[]
    for label,key in [('Train','train_summary_json'),('Validation','valid_summary_json')]:
        r=summary_accuracy(config[key])
        if r:
            metric,v=r;vals.append((label,v))
            sources[label.lower()+'_overall_accuracy']={'file':config[key],'key':metric,'accuracy_percent':v}
    if not vals:raise ValueError('No overall masked accuracy metric in summary JSON')
    ax.bar([x[0] for x in vals],[x[1] for x in vals],color=[COLORS[2],COLORS[4]][:len(vals)],width=.55)
    for i,(_,v) in enumerate(vals):ax.text(i,v+1,f'{v:.2f}%',ha='center',weight='bold')
    ax.set_ylim(0,110);ax.set_ylabel('Masked atom accuracy (%)')
    set_panel(ax,'d','Overall masked atom reconstruction')


def figure_h200(config,out,manifest):
    fig,axes=plt.subplots(2,2,figsize=(13.5,9.1),layout='constrained')
    tasks=[('training',training_plot),('element',element_plot),('geometry',geometry_plot),('overall',overall_plot)]
    successes=0
    for ax,(name,fn) in zip(axes.flat,tasks):
        try:fn(ax,config,manifest['panels']);successes+=1
        except Exception as e:
            ax.axis('off'); ax.text(.5,.5,'Unavailable: '+name+'\n'+str(e)[:120],ha='center',va='center',wrap=True)
            manifest['missing'].append(name+': '+str(e))
    fig.suptitle('cmri_model | H200 general molecular pretraining',fontsize=16,weight='bold')
    if successes:save(fig,out/'cmri_model_h200_main_results')
    else:plt.close(fig)


def figure_duplicate(config,out,manifest):
    fig,axes=plt.subplots(1,3,figsize=(16.2,5.0),layout='constrained')
    done=0
    d=csv_read(config['duplicate_multiplicity_csv'])
    try:
        if d is None:raise ValueError('Multiplicity CSV not configured')
        c=col(d,['n_records','multiplicity','copies','record_count','count_per_smiles'])
        if not c:raise ValueError('Need one multiplicity count per unique SMILES')
        x=as_num(d,c).dropna().astype(int);x=x[x>0]
        if x.empty:raise ValueError('No valid multiplicities')
        masks=[x==1,x==2,x==3,(x>=4)&(x<=5),(x>=6)&(x<=10),(x>=11)&(x<=20),x>20]
        unique=np.array([int(m.sum()) for m in masks]);raw=np.array([int(x[m].sum()) for m in masks])
        labels=['1','2','3','4–5','6–10','11–20','>20']; ix=np.arange(7)
        axes[0].bar(ix-.19,100*raw/raw.sum(),width=.38,color=COLORS[3],label='Raw records')
        axes[0].bar(ix+.19,100*unique/unique.sum(),width=.38,color=COLORS[0],label='Unique identities')
        axes[0].set_xticks(ix,labels);axes[0].set_ylabel('Sample fraction (%)');axes[0].legend(frameon=False,fontsize=8)
        set_panel(axes[0],'a','Molecular exposure due to duplicates');done+=1
        manifest['panels']['multiplicity']={'file':config['duplicate_multiplicity_csv'],'column':c}
    except Exception as e:axes[0].axis('off');axes[0].text(.5,.5,'Multiplicity source unavailable',ha='center');manifest['missing'].append(str(e))
    d=csv_read(config['duplicate_descriptors_csv'])
    try:
        if d is None:raise ValueError('Descriptor CSV not configured')
        g=col(d,['group','source','sampling','condition']);p=col(d,['logP','cLogP','heavy_atoms','TPSA'])
        if not g or not p:raise ValueError('Need group and numeric descriptor columns')
        group=d[g].astype(str).str.lower(); raw=as_num(d[group.str.contains('raw')],p).dropna()
        uniq=as_num(d[group.str.contains('dedup|unique')],p).dropna()
        if raw.empty or uniq.empty:raise ValueError('Need raw and dedup observations')
        axes[1].hist(raw,bins=25,density=True,alpha=.6,color=COLORS[3],label='Raw records')
        axes[1].hist(uniq,bins=25,density=True,alpha=.6,color=COLORS[0],label='Deduplicated')
        axes[1].set_xlabel(p);axes[1].set_ylabel('Density');axes[1].legend(frameon=False,fontsize=8)
        set_panel(axes[1],'b','Descriptor distribution comparison');done+=1
        manifest['panels']['descriptors']={'file':config['duplicate_descriptors_csv'],'descriptor':p,'group':g}
    except Exception as e:axes[1].axis('off');axes[1].text(.5,.5,'Descriptor source unavailable',ha='center');manifest['missing'].append(str(e))
    d=csv_read(config['duplicate_smd_ci_csv'])
    try:
        if d is None:raise ValueError('SMD CI CSV not configured')
        f=col(d,['descriptor','property','feature']);e=col(d,['smd','estimate','standardized_mean_difference'])
        l=col(d,['ci_lower','ci_low','lower']);h=col(d,['ci_upper','ci_high','upper'])
        if not all([f,e,l,h]):raise ValueError('Need descriptor, estimate, CI lower/upper columns')
        q=d[[f,e,l,h]].copy()
        for k in [e,l,h]:q[k]=pd.to_numeric(q[k],errors='coerce')
        q=q.dropna();q=q[(q[l]<=q[e])&(q[e]<=q[h])]
        if q.empty:raise ValueError('No valid SMD rows')
        y=np.arange(len(q));axes[2].errorbar(q[e],y,xerr=[q[e]-q[l],q[h]-q[e]],fmt='o',color=COLORS[4],capsize=3)
        axes[2].axvline(0,color='#7a7772',ls='--');axes[2].set_yticks(y,list(q[f]));axes[2].invert_yaxis()
        axes[2].set_xlabel('Unique − raw / raw SD');set_panel(axes[2],'c','Descriptor shifts: 95% CI');done+=1
        manifest['panels']['smd']={'file':config['duplicate_smd_ci_csv'],'columns':[f,e,l,h]}
    except Exception as e:axes[2].axis('off');axes[2].text(.5,.5,'SMD + CI source unavailable',ha='center');manifest['missing'].append(str(e))
    fig.suptitle('cmri_model | General dataset duplication audit',fontsize=15,weight='bold')
    if done:save(fig,out/'cmri_model_duplicate_audit')
    else:plt.close(fig)


def dataset_name(s):
    x=str(s).lower()
    if 'esol' in x or 'delaney' in x:return 'ESOL'
    if 'freesolv' in x or 'sampl' in x:return 'FreeSolv'
    if 'lipophilicity' in x or 'lipo' in x:return 'Lipophilicity'
    return None


def figure_benchmark(config,out,manifest):
    d=csv_read(config['benchmark_csv'])
    if d is None:manifest['missing'].append('Benchmark CSV unavailable');return
    ds=col(d,['benchmark','dataset','task']);m=col(d,['model','method','representation','features']);r=col(d,['rmse','test_rmse'])
    frac=col(d,['train_fraction','training_fraction','fraction'])
    if not all([ds,m,r]):manifest['missing'].append('Benchmark CSV columns not recognized: '+str(list(d.columns)));return
    q=d.copy();q['dsname']=d[ds].map(dataset_name);q['value']=as_num(d,r)
    if frac:
        ff=as_num(d,frac)
        threshold=99.9 if ff.max(skipna=True)>1.01 else .999
        q=q[ff>=threshold]  # main full-label comparisons only
    q=q.dropna(subset=['dsname','value'])
    if q.empty:manifest['missing'].append('No benchmark rows');return
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for i,(name,ax) in enumerate(zip(['ESOL','FreeSolv','Lipophilicity'],axes)):
        group=q[q.dsname==name].groupby(m)['value'].agg(['mean','std','count']).sort_values('mean')
        if group.empty:ax.axis('off');continue
        names=list(group.index);vals=group['mean'].to_numpy();sd=group['std'].fillna(0).to_numpy()
        ax.barh(names,vals,color=[COLORS[j%len(COLORS)] for j in range(len(vals))])
        if (group['count']>1).any():ax.errorbar(vals,np.arange(len(vals)),xerr=sd,fmt='none',ecolor='#3b3533',capsize=3)
        ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.3);ax.set_xlabel('RMSE (lower is better)');ax.grid(axis='x',alpha=.15)
        ax.set_title('abc'[i]+'  '+name,loc='left',fontweight='bold')
        for j,v in enumerate(vals):ax.text(v+max(vals)*.025,j,f'{v:.3f}',va='center',fontsize=8)
    fig.suptitle('cmri_model | Frozen molecular features + Ridge benchmarks',fontsize=15,weight='bold')
    save(fig,out/'cmri_model_downstream_benchmarks')
    manifest['panels']['benchmark']={'file':config['benchmark_csv'],'dataset_col':ds,'method_col':m,'metric_col':r,'fraction_col':frac}


def write_notes(out,manifest):
    lines=[
     '# cmri_model — Figure interpretation and provenance',
     '',
     '## H200 main results',
     'The General checkpoint was trained on H200 with an effective batch size of 448. '
     'Masked-atom recovery evaluates reconstruction of hidden atomic identities. '
     'Per-element recovery is essential because common elements (especially carbon) dominate the aggregate accuracy. '
     'Low Br/I support makes their accuracy estimates uncertain.',
     '',
     'Coordinate denoising compares noisy input coordinates and model-predicted coordinates. '
     'This code plots the mean of per-molecule RMSEs, not the atom-weighted global RMSE. '
     'Do not substitute this value for a previously reported 75.6% reduction without confirming the original aggregation.',
     '',
     'Training curves and checkpoint-validation metrics assess optimization and reconstruction, '
     'not biological activity or downstream mRNA delivery.',
     '',
     '## Duplication audit',
     'Previously reported: 18.84M records; 13.27M distinct raw SMILES; 29.55% duplicate excess. '
     'These historical summary values are not re-created without their audit source tables. '
     'Deduplication changes exposure frequency. Similar descriptor and Murcko distributions do not prove '
     'that deduplication cannot affect generalization. This remains a training ablation question.',
     '',
     '## ESOL, FreeSolv and Lipophilicity',
     'Frozen embeddings + Ridge test transfer of learned molecular representations to measured '
     'physicochemical properties. Comparisons with RDKit and Morgan are informative only if matched '
     'splits, identical target measurements and consistent hyperparameter tuning are verified. '
     'These tasks do not directly establish biological activity or mRNA delivery.',
     '',
     '## Data sources used',
    ]
    for name,details in manifest['panels'].items():lines.append('- '+name+': '+str(details))
    if manifest['missing']:
        lines+=['','## Missing inputs']+['- '+v for v in manifest['missing']]
    (out/'cmri_model_figure_justification.md').write_text('\n'.join(lines),encoding='utf-8')
    try:
        from docx import Document
        doc=Document();doc.add_heading('cmri_model — Figure Justification',0)
        blocks='\n'.join(lines).split('\n')
        for line in blocks:
            if line.startswith('# '):continue
            if line.startswith('## '):doc.add_heading(line[3:],1)
            elif line.startswith('- '):doc.add_paragraph(line[2:],style='List Bullet')
            elif line.strip():doc.add_paragraph(line)
        doc.save(out/'cmri_model_figure_justification.docx')
    except ImportError:
        manifest['missing'].append('Optional python-docx not installed; Markdown notes created instead')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inspect-only',action='store_true')
    ap.add_argument('--config',type=Path)
    ap.add_argument('--out',type=Path,default=PROJECT/'summary_figures')
    args=ap.parse_args();cfg=cfg_defaults()
    if args.config:cfg.update(json.loads(args.config.read_text()))
    out=args.out.expanduser();out.mkdir(parents=True,exist_ok=True)
    audit=[]
    for key,path in cfg.items():
        p=Path(path) if path else None
        present=bool(p and p.is_file());schema=''
        if present and p.suffix.lower()=='.csv':
            try:schema=' | '.join(pd.read_csv(p,nrows=0).columns)
            except Exception as e:schema='ERROR: '+str(e)
        elif present and p.suffix.lower()=='.json':
            schema=' | '.join(k for k,_ in list(flatten(json_read(p)))[:30])
        audit.append({'source':key,'found':present,'path':str(p or ''),'columns_or_keys':schema})
    pd.DataFrame(audit).to_csv(out/'source_schema_audit.csv',index=False)
    for a in audit:print(a['source'],a['found'],a['columns_or_keys'][:150])
    if args.inspect_only:return
    manifest={'model':'cmri_model','panels':{},'missing':[],'sources':audit}
    figure_h200(cfg,out,manifest)
    figure_duplicate(cfg,out,manifest)
    figure_benchmark(cfg,out,manifest)
    write_notes(out,manifest)
    (out/'provenance_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    (out/'missing_inputs.txt').write_text('\n'.join(manifest['missing'])+'\n')
    print('Output directory:',out)
    print('Review missing_inputs.txt and provenance_manifest.json before using the figures.')

if __name__=='__main__':main()
