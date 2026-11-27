#!/usr/bin/env python3
"""Recalculate exact 10-common-support F1 from archived prediction CSVs.
No model fitting, no CSV dataset access. Requires locally saved experimental outputs.
Usage: python cbid_31617_recompute_common10.py [--root ~/Downloads/CBID_RT2022_Controlled_Window_v2]
"""
from __future__ import annotations
import argparse, json, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

GAT_MODES = ('GATv2_uppertriangular', 'GATv2_selfloops')
BASE_MODES = ('RF_vote10', 'XGB_vote10', 'RF_window_mean', 'XGB_window_mean')
SEEDS_EXPECTED = (11, 22, 33, 42, 44)

def read_pair(csvfile):
    df=pd.read_csv(csvfile, keep_default_na=False)
    if not {'true','pred'}.issubset(df):
        raise ValueError(f'Missing true/pred columns in {csvfile}')
    if df[['true','pred']].isna().any().any():
        raise ValueError(f'NA in {csvfile}')
    return df[['true','pred']].astype(str).reset_index(drop=True)

def first_existing_support(files):
    for f in files:
        if f.exists():
            d=pd.read_csv(f)
            col='has_window_support_in_all_three_partitions'
            if col not in d: raise ValueError('Support file lacks column '+col)
            common=d[d[col].astype(str).str.lower().isin(['true','1'])]['class'].tolist()
            if len(common)!=10: raise ValueError(f'Expected 10 common-support classes, got {len(common)} in {f}: {common}')
            return common
    raise FileNotFoundError('No per_class_split_and_window_support.csv found')

def run_seed(directory, mode):
    selection=directory/f'{mode}_selection.json'
    if selection.exists():
        d=json.loads(selection.read_text())
        if 'seed' in d:return int(d['seed'])
        for k in ('arguments','params'):
            if k in d and isinstance(d[k],dict) and 'seed' in d[k]:return int(d[k]['seed'])
    metadata=directory/'full_run_metadata.json'
    if metadata.exists():
        d=json.loads(metadata.read_text()); args=d.get('command_arguments',{})
        if 'seed' in args:return int(args['seed'])
    return None

def inspect(root:Path,out:Path):
    root=root.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    gatdirs=sorted(root.glob('gat_*'))
    basedirs=sorted(root.glob('baselines_*'))+sorted(root.glob('all_*'))
    common=first_existing_support([d/'per_class_split_and_window_support.csv' for d in gatdirs+basedirs])
    all_classes=sorted(set(common+['NMAP_FIN_SCAN','Metasploit_Brute_Force_SSH']))
    print('PRIMARY METRIC SET: 10 classes with positive window support in TRAIN, VALIDATION, TEST.')
    print('Post-hoc descriptive definition, NOT preregistered. Classes:', common)
    rows=[]; all_pairs={}; by_model={}
    for d in gatdirs+basedirs:
        is_gat=d.name.startswith('gat_')
        modes=GAT_MODES if is_gat else BASE_MODES
        for model in modes:
            predfile=d/f'{model}_test_pred.csv'
            if not predfile.exists():continue
            pair=read_pair(predfile)
            seed=run_seed(d,model)
            if seed is None:
                print('WARNING: seed unavailable for',predfile,'; skipping rather than guessing')
                continue
            y=pair['true'].to_numpy(); yp=pair['pred'].to_numpy()
            if len(y)!=1769: raise ValueError(f'Unexpected test window length {len(y)} in {predfile}')
            macro10=f1_score(y,yp,labels=common,average='macro',zero_division=0)
            macro12=f1_score(y,yp,labels=all_classes,average='macro',zero_division=0)
            result={'model':model,'seed':seed,'n_test_windows':len(pair),
                    'accuracy':accuracy_score(y,yp),
                    'macroF1_10_direct':macro10,'macroF1_12_direct':macro12,
                    'n_wrong':int((y!=yp).sum()),
                    'source_dir':str(d)}
            rows.append(result); key=(model,seed)
            if key in all_pairs:
                print('WARNING: duplicate model/seed',key,'; keeping first result from', all_pairs[key][1])
                continue
            all_pairs[key]=(pair,d)
            cls=classification_report(y,yp,labels=all_classes,zero_division=0,output_dict=True)
            (out/f'{model}_seed{seed}_perclass.json').write_text(json.dumps(cls,indent=2),encoding='utf-8')
            by_model.setdefault(model,[]).append(result)
    if not rows:raise RuntimeError('No valid *_test_pred.csv results found')
    # Remove duplicate model/seed rows; early attempts are excluded if duplicates exist.
    results=pd.DataFrame(rows).drop_duplicates(['model','seed'],keep='first')
    results.sort_values(['model','seed']).to_csv(out/'exact_common10_by_model_seed.csv',index=False)
    summaries=[]
    for model,d in results.groupby('model',sort=True):
        n=len(d)
        summaries.append({'model':model,'n_seeds':n,
                          'seeds':','.join(map(str,sorted(d.seed))),
                          'macroF1_10_mean':float(d.macroF1_10_direct.mean()),
                          'macroF1_10_sample_sd':float(d.macroF1_10_direct.std(ddof=1)) if n>1 else None,
                          'macroF1_12_mean':float(d.macroF1_12_direct.mean()),
                          'accuracy_mean':float(d.accuracy.mean()),
                          'accuracy_sample_sd':float(d.accuracy.std(ddof=1)) if n>1 else None})
    summ=pd.DataFrame(summaries); summ.to_csv(out/'exact_common10_summary.csv',index=False)
    # Paired GAT comparison.
    pairs=[]
    for seed in SEEDS_EXPECTED:
        a=results[(results.model=='GATv2_uppertriangular')&(results.seed==seed)]
        b=results[(results.model=='GATv2_selfloops')&(results.seed==seed)]
        if len(a)==len(b)==1:
            pairs.append({'seed':seed,'delta_F1_10_real_minus_self':float(a.iloc[0].macroF1_10_direct-b.iloc[0].macroF1_10_direct),
                          'delta_accuracy':float(a.iloc[0].accuracy-b.iloc[0].accuracy)})
    pd.DataFrame(pairs).to_csv(out/'paired_GAT_delta_common10.csv',index=False)
    # Mixed-label control: cross-check alignment on matching full run manifest.
    mixed=[]
    for key,(preds,d) in all_pairs.items():
        mf=d/'test_window_manifest.csv'
        if not mf.exists():continue
        m=pd.read_csv(mf)
        if len(m)!=len(preds):raise ValueError(f'Row count mismatch manifest/predictions in {d}')
        truth=(m['majority_label'].astype(str).to_numpy()==preds.true.to_numpy())
        if not truth.all():raise AssertionError(f'True target mismatch between manifest and predictions in {d}')
        is_mixed=m['n_distinct_classes'].to_numpy()>1
        wrong=(preds.true.to_numpy()!=preds.pred.to_numpy())
        mixed.append({'model':key[0],'seed':key[1],'total_mixed':int(is_mixed.sum()),
                      'total_errors':int(wrong.sum()),'errors_on_mixed':int((wrong&is_mixed).sum()),
                      'errors_on_pure':int((wrong&~is_mixed).sum()),
                      'all_errors_confined_to_mixed':bool((wrong<=is_mixed).all())})
    pd.DataFrame(mixed).to_csv(out/'error_vs_mixed_windows.csv',index=False)
    # Independent prediction unit / no new splits.
    readme='CBID-IPS numerical audit, generated from saved TEST prediction files.\n\n'
    readme+='Ten classes were selected RETROSPECTIVELY as the intersection of positive window support across train/validation/test; this is descriptive, NOT a pre-registered primary endpoint.\n'
    readme+='The classes NMAP_FIN_SCAN (zero test windows) and Metasploit_Brute_Force_SSH (zero validation windows) are NOT deleted from raw model predictions, only omitted from descriptive common-support F1.\n'
    readme+='NO new test, no new split and no graph training is performed. Repeated use of the same test was exploratory and cannot establish generalization.\n'
    readme+='A better F1 for self-loop GATv2 compared with trees does NOT establish an attention benefit: with only one self-loop per node, attention weights normalize to 1. The model acts as a nonlinear per-record encoder plus pooling.\n'
    readme+='Tree results from one seed should NOT be described as a five-seed mean; use optional separate RF/XGB refits to quantify their training stochasticity, not as five independent test sets.\n'
    (out/'README_READ_BEFORE_PUBLICATION.txt').write_text(readme,encoding='utf-8')
    with zipfile.ZipFile(str(out)+'.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.glob('*')):
            if f.is_file():z.write(f,arcname=f.name)
    print('\nVERIFIED FROM SAVED PREDICTIONS:')
    print(summ.to_string(index=False,float_format=lambda v:f'{v:.6f}'))
    if mixed:
        print('\nMIXED-LABEL DIAGNOSTICS:')
        print(pd.DataFrame(mixed).to_string(index=False))
    print('\nOUTPUT:',out,'\nUPLOAD:',str(out)+'.zip')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.home()/'Downloads'/'CBID_RT2022_Controlled_Window_v2')
    ap.add_argument('--out',type=Path,default=Path.home()/'Downloads'/'CBID_31617_Exact_Common10_Audit')
    args=ap.parse_args();inspect(args.root,args.out)
