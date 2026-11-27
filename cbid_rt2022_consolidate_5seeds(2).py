#!/usr/bin/env python3
"""Audit and consolidate the five pre-existing RT-IoT2022 GATv2 runs.

No model is trained and no data CSV is opened. This script only reads already
saved selected-checkpoint metrics and prediction files, recomputes metrics from
predictions, and packages evidence for the IJ-AI #31617 revision.

Usage:
  python cbid_rt2022_consolidate_5seeds.py
  python cbid_rt2022_consolidate_5seeds.py --root ~/Downloads/CBID_RT2022_Controlled_Window_v2
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

VARIANTS = ('GATv2_uppertriangular', 'GATv2_selfloops')
SEEDS = (11, 22, 33, 42, 44)

def write_json(path, val):
    path.write_text(json.dumps(val, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def collect(root):
    found={}
    for run in sorted(root.glob('gat_*')):
        metadata=run/'full_run_metadata.json'
        if not metadata.is_file(): continue
        data=json.loads(metadata.read_text(encoding='utf-8'))
        seed=int(data['command_arguments']['seed'])
        if seed not in SEEDS: continue
        required=[run/f'{var}_{part}.json' for var in VARIANTS for part in ('test','validation')]
        required += [run/f'{var}_{part}_pred.csv' for var in VARIANTS for part in ('test','validation')]
        required += [run/f'{var}_selection.json' for var in VARIANTS]
        missing=[str(p.name) for p in required if not p.is_file()]
        if missing:
            print('Skipping incomplete run:',run,'missing:',missing)
            continue
        # Use latest *complete* run for each seed; report all alternatives separately.
        found.setdefault(seed,[]).append(run)
    return found

def audit(root,out):
    found=collect(root)
    missing=sorted(set(SEEDS)-set(found))
    if missing: raise SystemExit(f'Missing complete results for seeds {missing} under {root}.')
    chosen={seed:runs[-1] for seed,runs in found.items()}
    all_rows=[]; class_rows=[]; problems=[]; evidences=[]; versions=[]
    reference_truth={}
    for seed in SEEDS:
        run=chosen[seed]
        metadata=json.loads((run/'full_run_metadata.json').read_text(encoding='utf-8'))
        versions.append({'seed':seed,'path':str(run),'candidate_runs':[str(p) for p in found[seed]],
                        'args':metadata.get('command_arguments')})
        eligible=metadata['protocol']['eligible_window_labels_selected_using_train_validation_ONLY']
        for variant in VARIANTS:
            selection=json.loads((run/f'{variant}_selection.json').read_text(encoding='utf-8'))
            for partition in ('validation','test'):
                mpath=run/f'{variant}_{partition}.json'
                pred_path=run/f'{variant}_{partition}_pred.csv'
                obj=json.loads(mpath.read_text(encoding='utf-8'))
                pred=pd.read_csv(pred_path)
                if not {'true','pred'}.issubset(pred.columns):
                    problems.append(f'{pred_path}: missing true/pred columns');continue
                known_classes=[k for k in obj['per_class'] if k not in ('accuracy','macro avg','weighted avg','micro avg','samples avg')]
                if len(known_classes) != 12: problems.append(f'{mpath}: expected 12 classes, got {len(known_classes)}')
                y=pred['true'].astype(str).to_numpy()
                y_hat=pred['pred'].astype(str).to_numpy()
                accuracy=float(accuracy_score(y,y_hat))
                f_all=float(f1_score(y,y_hat,labels=known_classes,average='macro',zero_division=0))
                f_el=float(f1_score(y,y_hat,labels=eligible,average='macro',zero_division=0))
                for field,recomputed in [('accuracy',accuracy),('macro_F1_all_12_classes',f_all),('macro_F1_eligible_classes',f_el)]:
                    logged=float(obj[field])
                    if abs(logged-recomputed)>1e-8:
                        problems.append(f'{seed}/{variant}/{partition}: {field} logged={logged:.9f} recalculated={recomputed:.9f}')
                if partition in reference_truth:
                    if not np.array_equal(reference_truth[partition],y):
                        problems.append(f'{seed}/{variant}/{partition}: ground-truth order differs from other runs')
                else: reference_truth[partition]=y
                if accuracy < .95: problems.append(f'IMPORTANT LOW ACCURACY: seed={seed}, variant={variant}, partition={partition}, accuracy={accuracy:.5f}')
                row={'seed':seed,'variant':variant,'partition':partition,'epoch':int(selection['best_epoch']),
                     'accuracy':accuracy,'macro_F1_all12':f_all,'macro_F1_eligible11':f_el,
                     'n_windows':len(pred),'run_dir':str(run)}
                all_rows.append(row)
                for name in known_classes:
                    entry=obj['per_class'][name]
                    class_rows.append({'seed':seed,'variant':variant,'partition':partition,'class':name,
                                       'support':int(entry['support']), 'precision':float(entry['precision']),
                                       'recall':float(entry['recall']), 'f1':float(entry['f1-score'])})
                if seed==44 and variant==VARIANTS[0] and partition=='test':
                    confusion=pd.crosstab(pd.Series(y,name='true'),pd.Series(y_hat,name='pred'))
                    confusion.to_csv(out/'seed44_uppertriangular_confusion_matrix.csv')
                for fname in [mpath,pred_path]:
                    evidences.append({'seed':seed,'variant':variant,'partition':partition,
                                      'name':fname.name,'path':str(fname),'sha256':hash_file(fname)})
    tab=pd.DataFrame(all_rows); tab.to_csv(out/'metrics_all_seeds.csv',index=False)
    pc=pd.DataFrame(class_rows); pc.to_csv(out/'per_class_all_seeds.csv',index=False)
    test=tab[tab.partition=='test']
    agg=test.groupby('variant').agg(n_seeds=('seed','size'),
        accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
        macroF1_all12_mean=('macro_F1_all12','mean'),macroF1_all12_sd=('macro_F1_all12','std'),
        macroF1_eligible11_mean=('macro_F1_eligible11','mean'),
        macroF1_eligible11_sd=('macro_F1_eligible11','std')).reset_index()
    agg.to_csv(out/'test_summary_mean_sd.csv',index=False)
    paired=[]
    for seed in SEEDS:
        a=test[(test.seed==seed)&(test.variant==VARIANTS[0])].iloc[0]
        b=test[(test.seed==seed)&(test.variant==VARIANTS[1])].iloc[0]
        paired.append({'seed':seed,'accuracy_edges_minus_selfloops':a.accuracy-b.accuracy,
            'macroF1_all12_edges_minus_selfloops':a.macro_F1_all12-b.macro_F1_all12,
            'macroF1_eligible11_edges_minus_selfloops':a.macro_F1_eligible11-b.macro_F1_eligible11})
    pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    write_json(out/'audit_warnings.json',{'warnings':problems,'selected_runs':versions,
        'methodological_caveat':'Five seeds on the same class-sorted split, not independent networks. Test examined previously.'})
    write_json(out/'source_file_sha256.json',evidences)
    # Pack compact evidence, including full per-class test JSON and specific seed 44 predictions for debugging.
    evidence=out/'source_evidence';evidence.mkdir(exist_ok=True)
    for seed in SEEDS:
        run=chosen[seed]; d=evidence/f'seed_{seed}';d.mkdir(exist_ok=True)
        for var in VARIANTS:
            for pat in [f'{var}_selection.json',f'{var}_test.json',f'{var}_validation.json']:
                shutil.copy2(run/pat,d/pat)
            if seed==44:
                for pat in [f'{var}_test_pred.csv',f'{var}_validation_pred.csv']:
                    shutil.copy2(run/pat,d/pat)
    shutil.make_archive(str(out),'zip',root_dir=out)
    print('\n=== VERIFIED TEST RESULTS, FIVE SEEDS ===')
    print(agg.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\n=== PAIRWISE DIFFERENCES (EDGES - SELFLOOPS) ===')
    print(pd.DataFrame(paired).to_string(index=False,float_format=lambda x:f'{x:+.5f}'))
    print('\n=== WARNINGS ===')
    if problems:
        for warning in problems:print(' *',warning)
    else:print('Metrics match prediction files; no low-accuracy alerts')
    print('\nRESULTS:',out)
    print('UPLOAD:',str(out)+'.zip')
    print('No model was trained. No original RT-IoT2022 CSV read.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.home()/'Downloads'/'CBID_RT2022_Controlled_Window_v2')
    p.add_argument('--out',type=Path,default=Path.home()/'Downloads'/'CBID_31617_RT2022_FiveSeed_Audit')
    args=p.parse_args()
    if not args.root.is_dir():p.error(f'Directory not found: {args.root}')
    args.out.mkdir(parents=True,exist_ok=True)
    audit(args.root,args.out)
