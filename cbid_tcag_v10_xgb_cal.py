#!/usr/bin/env python3
"""Exploratory chronological B-day evaluation, no confirmation day access.
Requires v9_03_relation_gate.py in the same directory. Fixed clock folds;
no post-hoc boundary selection. Context permutation is a feature control.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
from cbid_tcag_v9_03_relation_gate import threshold,metrics,perturb

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=8);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True)
 m=json.loads((a.input/'feature_manifest.json').read_text());g=m['feature_groups'];L,T,R=[g[k] for k in ('local','temporal','relational')];cols=L+T+R
 allowed={'flows','IN_BYTES','OUT_BYTES','IN_PKTS','OUT_PKTS','FLOW_DURATION_MILLISECONDS','RETRANSMITTED_IN_PKTS','RETRANSMITTED_OUT_PKTS','distinct_dst_ports','total_bytes','total_pkts','mean_duration','src_flows','src_bytes','src_destinations','pair_flows_prev60','pair_total_bytes_prev60','pair_total_pkts_prev60','pair_active_windows_prev60','source_src_flows_prev60','source_src_bytes_prev60','source_src_destinations_prev60','source_active_windows_prev60','peer_count','peer_mean_flows','peer_mean_total_bytes','peer_mean_total_pkts','peer_mean_distinct_dst_ports','peer_mean_pair_flows_prev60','peer_mean_pair_active_windows_prev60'}
 if set(cols)!=allowed or len(cols)!=len(allowed):raise ValueError('Unexpected manifest')
 day='2019-04-25';start=int(pd.Timestamp(day,tz='UTC').timestamp())//10
 d=pd.read_csv(a.input/f'pair_windows_{day}.csv',usecols=['src','dst','w','y','attack_flows']+cols).sort_values(['w','src','dst']).reset_index(drop=True)
 if not d.w.between(start,start+8639).all() or d.duplicated(['src','dst','w']).any() or not d.y.isin([0,1]).all() or not (d.y==(d.attack_flows>0)).all() or not np.isfinite(d[cols].to_numpy()).all():raise ValueError('Invalid data')
 models={'M0_local':L,'M1_temporal':L+T,'M2_context':L+R,'M3_full':cols,'M4_context_permuted':cols};rows=[];audit=[];pred=[]
 for h in (6,12,18):
  cut=start+h*360; cal=cut-2*360; end=cut+6*360; embargo=12
  ds={'train':d[d.w<cal-embargo], 'cal':d[(d.w>=cal+embargo)&(d.w<cut-embargo)], 'test':d[(d.w>=cut+embargo)&(d.w<end)]};ds={k:v.reset_index(drop=True) for k,v in ds.items()}
  counts={k:{'n':len(v),'positive':int(v.y.sum()),'negative':int((v.y==0).sum())} for k,v in ds.items()}
  eligible=all(min(v['positive'],v['negative'])>=100 for v in counts.values()); audit.append({'fold':h,'counts':counts,'eligible':eligible});print(json.dumps(audit[-1]),flush=True)
  if not eligible:continue
  for seed in (42,43,44):
   sh={k:perturb(v,R,seed+100*i)[0] for i,(k,v) in enumerate(ds.items())}
   for name,features in models.items():
    z=sh if name.startswith('M4') else ds;y=z['train'].y.to_numpy();weights=np.where(y==1,len(y)/(2*y.sum()),len(y)/(2*(len(y)-y.sum())))
    model=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=.05,subsample=.8,colsample_bytree=.8,random_state=seed,n_jobs=a.threads,tree_method='hist',device=a.device,eval_metric='logloss')
    model.fit(z['train'][features],y,sample_weight=weights)
    pc=model.predict_proba(z['cal'][features])[:,1].astype(float);t=threshold(z['cal'].y.to_numpy(),pc)
    for k in ('cal','test'):
     v=z[k];p1=pc if k=='cal' else model.predict_proba(v[features])[:,1].astype(float);yy=v.y.to_numpy();fpr,tpr,_=roc_curve(yy,p1);r=dict(fold=h,seed=seed,model=name,split=k,**metrics(yy,p1,t),oracle_recall_at_fpr01=float(tpr[fpr<=.01].max()));rows.append(r);print(json.dumps(r),flush=True)
     if k in ('cal','test'):
      q=ds[k][['src','dst','w','y','attack_flows','flows']].copy();q['split']=k;q['fold']=h;q['seed']=seed;q['model']=name;q['score']=p1;q['alert']=p1>=t;pred.append(q)
 (a.out/'split_audit.json').write_text(json.dumps(audit,indent=2));pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False)
 if not rows:print('NO-GO: no eligible fixed fold');return
 pd.concat(pred,ignore_index=True).to_csv(a.out/'predictions.csv',index=False)
 result=pd.DataFrame(rows).query("split=='test'");s=result.groupby(['fold','model'])[['AP','AUROC','recall','FPR','oracle_recall_at_fpr01']].mean();s.to_csv(a.out/'summary.csv');print(s.to_string());print('Saved:',a.out)
if __name__=='__main__':main()
