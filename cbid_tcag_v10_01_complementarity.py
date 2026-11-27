#!/usr/bin/env python3
"""Exploratory paired test-score audit. No fitting, threshold selection or confirmation data.
Inputs are the original V9.05 and V9.06 test predictions and metrics.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

KEY=['fold','seed','w','src','dst']
META=['y','attack_flows','flows']
XMODELS=['M0_local','M1_temporal','M2_context','M3_full','M4_context_permuted']
GMODELS=['G0_no_message','G1_real_peer','G2_shuffled_peer','G3_context_mlp']

def load(path,models):
    d=pd.read_csv(path/'predictions.csv',dtype={'src':str,'dst':str})
    if set(d.model)!=set(models): raise ValueError('Unexpected models: '+str(path))
    if d[KEY+META+['score','alert']].isna().any().any() or d.duplicated(KEY+['model']).any(): raise ValueError('Missing values or duplicate keys')
    if not d.y.isin([0,1]).all() or not (d.y==(d.attack_flows>0)).all() or not ((d.attack_flows>=0)&(d.flows>=d.attack_flows)).all(): raise ValueError('Invalid labels or counts')
    if not np.isfinite(d.score).all() or not d.score.between(0,1).all(): raise ValueError('Invalid scores')
    if not d.alert.isin([True,False,0,1]).all(): raise ValueError('Invalid alerts')
    m=pd.read_csv(path/'metrics.csv').query("split=='test'")
    expected={(f,s,k) for f in (12,18) for s in (42,43,44) for k in models}
    if set(map(tuple,m[['fold','seed','model']].to_numpy()))!=expected or m.duplicated(['fold','seed','model']).any(): raise ValueError('Unexpected metrics coverage')
    d=d.merge(m[['fold','seed','model','threshold']],on=['fold','seed','model'],validate='many_to_one')
    if not (d.alert.astype(bool)==(d.score>=d.threshold)).all(): raise ValueError('Saved decisions disagree with thresholds')
    return d

def paired(x,g):
    a=x.merge(g,on=KEY,how='outer',suffixes=('_x','_g'),validate='one_to_one',indicator=True)
    if not (a._merge=='both').all(): raise ValueError('Prediction keys do not align')
    for c in META:
        if not (a[c+'_x']==a[c+'_g']).all(): raise ValueError('Metadata mismatch: '+c)
    return a

def evaluate(a):
    y=a.y_x.to_numpy()==1; x=a.alert_x.to_numpy(dtype=bool); g=a.alert_g.to_numpy(dtype=bool)
    def count(mask):return int(mask.sum())
    p=count(y); n=count(~y)
    if not p or not n:raise ValueError('Missing class')
    s=(a.score_x+a.score_g)/2
    return dict(n=len(a),positives=p,negatives=n,
        AP_x=average_precision_score(y,a.score_x),AP_g=average_precision_score(y,a.score_g),
        AP_fixed_half=average_precision_score(y,s),AUROC_fixed_half=roc_auc_score(y,s),
        TP_both=count(y&x&g),TP_x_only=count(y&x&~g),TP_g_only=count(y&~x&g),FN_both=count(y&~x&~g),
        FP_both=count(~y&x&g),FP_x_only=count(~y&x&~g),FP_g_only=count(~y&~x&g),TN_both=count(~y&~x&~g),
        recall_union=count(y&(x|g))/p,FPR_union=count(~y&(x|g))/n,
        recall_intersection=count(y&x&g)/p,FPR_intersection=count(~y&x&g)/n)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--xgb',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    x=load(a.xgb,XMODELS);g=load(a.graph,GMODELS); rows=[]
    for f in (12,18):
        for seed in (42,43,44):
            xx=x[(x.fold==f)&(x.seed==seed)];gg=g[(g.fold==f)&(g.seed==seed)]
            for xm in XMODELS:
                for gm in GMODELS:
                    z=paired(xx[xx.model==xm],gg[gg.model==gm]);expected={12:37276,18:38788}[f]
                    if len(z)!=expected:raise ValueError('Unexpected test size')
                    rows.append(dict(fold=f,seed=seed,xgb=xm,graph=gm,**evaluate(z)))
    d=pd.DataFrame(rows);a.out.mkdir(parents=True,exist_ok=True);d.to_csv(a.out/'paired_metrics.csv',index=False)
    summary=d.groupby(['fold','xgb','graph']).mean(numeric_only=True).drop(columns='seed');summary.to_csv(a.out/'summary.csv')
    cols=['AP_x','AP_g','AP_fixed_half','TP_g_only','FP_g_only','recall_union','FPR_union','recall_intersection','FPR_intersection']
    print('All 120 pairings aligned by fold, seed, window, source, destination.');print(summary.xs('G1_real_peer',level='graph')[cols].to_string());print('Fixed 50/50 score fusion: ranking diagnostic only; no calibrated fusion threshold. Union/intersection use original saved alerts. All results exploratory.');print('Saved:',a.out)
    (a.out/'protocol.json').write_text(json.dumps(dict(scope='B exploratory test only',fusion='fixed 0.5 ranking diagnostic',fitted_parameters=False,confirmation_access=False),indent=2))
if __name__=='__main__':main()
