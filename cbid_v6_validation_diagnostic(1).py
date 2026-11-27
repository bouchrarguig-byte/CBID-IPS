#!/usr/bin/env python3
"""CBID-IPS IJ-AI #31617: post-hoc analysis of V6 validation predictions ONLY.

Reads the V6 run directory, never opens NF-ToN raw data, and never touches the
24-Apr holdout. Threshold sweeps on the SAME validation set are EXPLORATORY;
selected thresholds and scores cannot be reported as unbiased model performance.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, confusion_matrix, matthews_corrcoef)

DEFAULT_ROOT = Path.home() / 'Downloads' / 'CBID_NFToN_Validation_v6'

def locate(root: Path) -> Path:
    if (root / 'validation_metrics.csv').is_file():
        return root
    opts = sorted((p for p in root.glob('validation_v6_*')
                  if (p / 'validation_metrics.csv').exists()), reverse=True)
    if not opts:
        raise FileNotFoundError(f'No V6 run found under {root}; provide --run-dir /path/to/validation_v6_TIMESTAMP')
    return opts[0]

def compute(y, p, t):
    yh=(p >= t).astype(int)
    tn, fp, fn, tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
    return dict(threshold=float(t), macro_f1=float(f1_score(y,yh,average='macro',zero_division=0)),
                scanning_precision=float(precision_score(y,yh,zero_division=0)),
                scanning_recall=float(recall_score(y,yh,zero_division=0)),
                mcc=float(matthews_corrcoef(y,yh)),
                predicted_positive=int(yh.sum()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,default=DEFAULT_ROOT,
                    help='V6 output root OR specific validation_v6_TIMESTAMP directory')
    args=ap.parse_args()
    run=locate(args.run_dir)
    metrics=pd.read_csv(run/'validation_metrics.csv')
    files=sorted(run.glob('*_validation_predictions.csv'))
    if not files:
        raise RuntimeError('No *_validation_predictions.csv files in V6 run folder')
    rows=[]
    grids=np.round(np.arange(0.05,0.951,0.025),3)
    for file in files:
        df=pd.read_csv(file)
        if not {'y_true','prob_scan'}.issubset(df):
            print('Skipping unexpected file',file.name,file=sys.stderr);continue
        y=df['y_true'].to_numpy(dtype=int); p=df['prob_scan'].to_numpy(dtype=float)
        if len(y)==0 or set(np.unique(y))!={0,1} or not np.isfinite(p).all():
            raise ValueError(f'Invalid validation predictions in {file}')
        name=file.name.removesuffix('_validation_predictions.csv')
        fixed=compute(y,p,0.5)
        candidate=[compute(y,p,t) for t in grids]
        # In a post-hoc diagnostic, ALWAYS label this best threshold as optimistic.
        best=max(candidate,key=lambda r:(r['macro_f1'],-abs(r['threshold']-.5)))
        rows.append({'model_seed':name, 'rows':len(y), 'positive':int(y.sum()),
                     'positive_prevalence':float(y.mean()),
                     'average_precision':float(average_precision_score(y,p)),
                     'f1_fixed_05':fixed['macro_f1'],
                     'precision_fixed_05':fixed['scanning_precision'],
                     'recall_fixed_05':fixed['scanning_recall'],
                     'predpos_fixed_05':fixed['predicted_positive'],
                     'false_pos_fixed_05':fixed['fp'],
                     'false_neg_fixed_05':fixed['fn'],
                     'best_threshold_EXPLORATORY':best['threshold'],
                     'best_val_f1_OPTIMISTIC':best['macro_f1'],
                     'best_val_precision_OPTIMISTIC':best['scanning_precision'],
                     'best_val_recall_OPTIMISTIC':best['scanning_recall'],
                     'best_val_fp_OPTIMISTIC':best['fp']})
    if not rows:raise RuntimeError('No valid prediction files')
    out=run/'posthoc_validation_diagnostic'
    out.mkdir(exist_ok=True)
    result=pd.DataFrame(rows).sort_values('model_seed')
    result.to_csv(out/'validation_threshold_diagnostic.csv',index=False)
    note={
        'scope':'23-Apr-2019 validation only; this does NOT evaluate 24-Apr test',
        'warning':'THRESHOLDS SELECTED ON THESE SAME VALIDATION LABELS ARE OPTIMISTIC AND CANNOT BE REPORTED AS UNBIASED TEST SCORES.',
        'guidance':'If threshold tuning seems important, split the existing pre-holdout data into training, threshold-calibration, and independent validation in advance; then report locked results only on a genuinely unseen holdout.',
        'comparison_warning':'Repeated seeds on one split measure optimization variability, not independent confidence intervals.',
        'raw_validation_metrics_rows':len(metrics),
        'diagnostic_rows':len(rows)
    }
    (out/'README_VALIDATION_ONLY.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
    print('V6 validation post-hoc diagnostic (NO MODEL TRAINED, NO 24-Apr TEST READ)')
    print('Input:',run)
    print('Fixed threshold=0.5 results:')
    print(result[['model_seed','positive_prevalence','average_precision','f1_fixed_05','recall_fixed_05','precision_fixed_05','predpos_fixed_05','false_pos_fixed_05']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nEXPLORATORY ONLY — thresholds fitted to the SAME validation set (optimistically biased):')
    print(result[['model_seed','best_threshold_EXPLORATORY','best_val_f1_OPTIMISTIC']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nSaved',out)
    print('This is a diagnosis, not a new independent evaluation. Do not optimize on the already-seen 24-Apr test.')

if __name__=='__main__':main()
