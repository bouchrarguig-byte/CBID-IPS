#!/usr/bin/env python3
"""Exploratory feature gate, not a graph propagation or independent confirmation test.
A chronological 60/20/20 clock split with 120 s embargo; B external attack-family shift.
Only manifest-listed numeric features enter models. No April 26 input.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

def threshold(y,p,alpha=.01):
    neg=np.sort(p[y==0])[::-1]
    if not len(neg): raise ValueError('No negatives in calibration')
    k=int(np.floor(alpha*len(neg)))
    return float(np.nextafter(neg[min(k,len(neg)-1)], np.inf))

def metrics(y,p,t):
    tn,fp,fn,tp=confusion_matrix(y,p>=t,labels=[0,1]).ravel()
    return dict(n=len(y),positives=int(y.sum()),AP=float(average_precision_score(y,p)) if y.sum() else None,
                AUROC=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                threshold=t,TP=int(tp),FP=int(fp),TN=int(tn),FN=int(fn),
                recall=float(tp/(tp+fn)) if tp+fn else None,FPR=float(fp/(fp+tn)) if fp+tn else None)

def perturb(d,cols,seed):
    # Same-clock permutation of whole context vectors across pair observations.
    # Not a degree-preserving edge shuffle: a feature-level negative control only.
    rng=np.random.default_rng(seed); result=d.copy(); a=d[cols].to_numpy(copy=True)
    for ix in d.groupby('w',sort=False).indices.values():
        a[ix]=a[rng.permutation(ix)]
    result[cols]=a
    return result,float(np.mean(np.any(a!=d[cols].to_numpy(),axis=1)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=8);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    m=json.loads((a.input/'feature_manifest.json').read_text()); g=m['feature_groups']; L,T,R=(g[k] for k in ('local','temporal','relational'))
    cols=L+T+R
    if len(cols)!=len(set(cols)) or set(cols)&{'y','Label','Attack','attack_flows','src','dst','w'}: raise ValueError('Invalid feature manifest')
    def load(day):
        d=pd.read_csv(a.input/f'pair_windows_{day}.csv',usecols=['src','dst','w','y','attack_flows']+cols)
        start=int(pd.Timestamp(day,tz='UTC').timestamp())//10
        if not d.w.between(start,start+8639).all() or d.duplicated(['src','dst','w']).any(): raise ValueError('Invalid day or duplicate keys')
        if not d.y.isin([0,1]).all() or not (d.y==(d.attack_flows>0)).all(): raise ValueError('Invalid target')
        if not np.isfinite(d[cols].to_numpy()).all(): raise ValueError('Nonfinite features')
        return d.sort_values(['w','src','dst']).reset_index(drop=True),start
    A,start=load('2019-04-23'); B,_=load('2019-04-25')
    # Fixed clock boundaries, not selected using labels. Purge both sides of each split.
    c1=start+5184; c2=start+6912; embargo=12
    sets={'train':A[A.w<c1-embargo], 'cal':A[(A.w>=c1+embargo)&(A.w<c2-embargo)], 'test_A':A[A.w>=c2+embargo], 'shift_B':B}
    sets={k:v.reset_index(drop=True) for k,v in sets.items()}
    audit={k:dict(n=len(v),positives=int(v.y.sum()),negatives=int((v.y==0).sum()),first_w=int(v.w.min()) if len(v) else None,last_w=int(v.w.max()) if len(v) else None) for k,v in sets.items()}
    print(json.dumps(audit,indent=2),flush=True);(a.out/'split_audit.json').write_text(json.dumps(audit,indent=2))
    for k in ('train','cal','test_A'):
        if sets[k].y.nunique()!=2: raise SystemExit('NO-GO: fixed chronological split lacks both classes: '+k+'; do not change boundaries post hoc.')
    shuffled={}; rates={}
    for i,(k,v) in enumerate(sets.items()): shuffled[k],rates[k]=perturb(v,R,31617+i)
    models={'M0_local':L,'M1_temporal':L+T,'M2_context':L+R,'M3_full':cols,'M4_context_permuted':cols}
    rows=[]; predictions=[]
    for name,features in models.items():
        ds=shuffled if name.startswith('M4') else sets
        model=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=.05,subsample=.8,colsample_bytree=.8,random_state=31617,n_jobs=a.threads,tree_method='hist',device=a.device,eval_metric='logloss')
        model.fit(ds['train'][features],ds['train'].y)
        t=threshold(ds['cal'].y.to_numpy(),model.predict_proba(ds['cal'][features])[:,1].astype(float))
        for k in ('cal','test_A','shift_B'):
            d=ds[k]; prob=model.predict_proba(d[features])[:,1].astype(float); y=d.y.to_numpy()
            r=dict(model=name,split=k,**metrics(y,prob,t)); rows.append(r); print(json.dumps(r),flush=True)
            z=sets[k][['src','dst','w','y','attack_flows','flows']].copy(); z['model']=name;z['split']=k;z['score']=prob;z['alert']=prob>=t;predictions.append(z)
    pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False);pd.concat(predictions,ignore_index=True).to_csv(a.out/'predictions.csv',index=False)
    (a.out/'protocol.json').write_text(json.dumps(dict(feature_groups=g,permutation_changed_fraction=rates,calibration_FPR_target=.01,embargo_seconds=120,seed=31617,scope='exploratory only; no day C; no graph message passing'),indent=2))
    print('Permutation changed fraction:',rates,flush=True); print('Saved:',a.out)
if __name__=='__main__': main()
