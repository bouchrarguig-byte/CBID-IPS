#!/usr/bin/env python3
"""Compare previously archived RF/XGB window predictions on the SAME labels as V3.
No training. Selection labels are read from V3's TRAIN+VALIDATION rule.
Do not use this script to select hyperparameters based on prior test scores.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v3-run',type=Path,required=True,help='V3 preflight/gat results folder')
    ap.add_argument('--v2-baselines',type=Path,default=Path.home()/'Downloads'/'CBID_RT2022_Controlled_Window_v2',
                    help='Root containing baselines_* folders, or exact baselines run folder')
    ap.add_argument('--eval-test',action='store_true',help='Also display already explored test metrics')
    args=ap.parse_args()
    selection=args.v3_run/'validation_selection_labels.json'
    if not selection.is_file(): ap.error(f'Missing {selection}')
    selected=json.loads(selection.read_text(encoding='utf-8'))['selected_labels']
    support=pd.read_csv(args.v3_run/'per_class_split_and_window_support.csv')
    classes=support['class'].tolist()
    b=args.v2_baselines.expanduser()
    runs=[b] if b.name.startswith('baselines_') else sorted(b.glob('baselines_*'))
    runs=[r for r in runs if (r/'RF_vote10_validation_pred.csv').exists()]
    if not runs:ap.error(f'No existing baselines found under {b}')
    baseline=runs[-1]
    print('Baseline run:',baseline)
    print('Selection labels TRAIN+VAL:',selected)
    rows=[]
    for model in ('RF_vote10','XGB_vote10','RF_window_mean','XGB_window_mean'):
        for partition in (('validation','test') if args.eval_test else ('validation',)):
            f=baseline/f'{model}_{partition}_pred.csv'
            if not f.is_file():
                print('Skipping unavailable:',f);continue
            t=pd.read_csv(f)
            y=t['true'].astype(str).to_numpy(); pred=t['pred'].astype(str).to_numpy()
            if not set(y).issubset(set(classes)) or not set(pred).issubset(set(classes)):
                raise RuntimeError(f'Unexpected label names in {f}')
            score={ 'model':model,'partition':partition,'n_windows':len(t),
                    'accuracy':float(accuracy_score(y,pred)),
                    'macroF1_all12':float(f1_score(y,pred,labels=classes,average='macro',zero_division=0)),
                    'macroF1_selection':float(f1_score(y,pred,labels=selected,average='macro',zero_division=0)),
                    'macroF1_present_descriptive':float(f1_score(y,pred,labels=[c for c in classes if (y==c).sum()>0],average='macro',zero_division=0)),
                    'MCC':float(matthews_corrcoef(y,pred))}
            rows.append(score)
    if not rows:raise SystemExit('No model predictions read')
    table=pd.DataFrame(rows)
    out=args.v3_run/'matched_archived_baselines.csv'
    table.to_csv(out,index=False)
    print(table.to_string(index=False,float_format=lambda a:f'{a:.5f}'))
    print('Output:',out)
    print('ARCHIVED BASELINES ONLY; NO MODEL TRAINING.')

if __name__=='__main__':main()
