#!/usr/bin/env python3
"""Exploratory B-only fusion. Re-run original base models with calibration predictions.
Keep original scripts intact. Fit fusion on earlier calibration segment, set threshold
on later segment, separated by 120s on both sides. No test-driven selection.
"""
import argparse, json, subprocess, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from cbid_tcag_v9_03_relation_gate import threshold, metrics

KEY=['fold','seed','split','w','src','dst']

def regenerate(source, target, input_dir, out, device, threads, graph=False):
    s=source.read_text()
    if s.count("if k=='test':")!=1: raise ValueError('Unexpected original script: '+str(source))
    s=s.replace("if k=='test':", "if k in ('cal','test'):")
    token="q['fold']=h;"
    if s.count(token)!=1: raise ValueError('Unexpected prediction writer')
    s=s.replace(token,"q['split']=k;"+token)
    target.write_text(s)
    cmd=[sys.executable,str(target),'--input',str(input_dir),'--out',str(out),'--device',device,'--threads',str(threads)]
    if graph:cmd+=['--epochs','30']
    subprocess.run(cmd,check=True)
    return hashlib.sha256(s.encode()).hexdigest()

def load(path, model):
    d=pd.read_csv(path, usecols=KEY+['model','y','score'])
    d=d[d.model==model].drop(columns='model')
    if d.empty or d.duplicated(KEY).any() or not d.y.isin([0,1]).all() or not d.score.between(0,1).all():raise ValueError('Invalid predictions '+model)
    return d

def join(x,g):
    d=x.merge(g,on=KEY,how='outer',suffixes=('_x','_g'),indicator=True,validate='one_to_one')
    if not (d._merge=='both').all() or not (d.y_x==d.y_g).all():raise ValueError('Unaligned observations')
    return d.rename(columns={'y_x':'y'}).drop(columns=['y_g','_merge'])

def split_cal(d,fold):
    start=int(pd.Timestamp('2019-04-25',tz='UTC').timestamp())//10
    mid=start+fold*360-360
    return d[(d.split=='cal')&(d.w<mid-12)],d[(d.split=='cal')&(d.w>=mid+12)],d[d.split=='test']

def combine(d,alpha):
    return (1-alpha)*d.score_x.to_numpy()+alpha*d.score_g.to_numpy()

def select_alpha(d):
    # Same fixed grid for real and negative-control graph; AP selection on fit only.
    grid=np.linspace(0,1,11)
    ap=[average_precision_score(d.y,combine(d,a)) for a in grid]
    return float(grid[int(np.argmax(ap))])

def run_fusion(xpath,gpath,out):
    rows=[]; audits=[]
    # Both strong XGB variants retained; no retrospective winner selection.
    for xm in ['M0_local','M2_context','M3_full']:
      x=load(xpath,xm)
      for gm in ['G1_real_peer','G0_no_message','G2_shuffled_peer']:
        d=join(x,load(gpath,gm))
        for (fold,seed),v in d.groupby(['fold','seed']):
          fit,cal,test=split_cal(v,fold)
          counts={k:{'positive':int(q.y.sum()),'negative':int((q.y==0).sum())} for k,q in [('fit',fit),('threshold',cal),('test',test)]}
          ok=all(min(c.values())>=100 for c in counts.values())
          audits.append(dict(fold=int(fold),seed=int(seed),xgb=xm,graph=gm,eligible=ok,counts=counts))
          if not ok:continue
          alpha=select_alpha(fit)
          for kind,a in [('xgb',0.),('graph',1.),('fixed',.5),('learned_scalar',alpha)]:
            t=threshold(cal.y.to_numpy(),combine(cal,a),.01)
            r=dict(fold=int(fold),seed=int(seed),xgb=xm,graph=gm,variant=kind,alpha=a,**metrics(test.y.to_numpy(),combine(test,a),t))
            rows.append(r)
    (out/'fusion_split_audit.json').write_text(json.dumps(audits,indent=2))
    if not rows: print('NO-GO: no eligible calibration subdivisions; do not move boundaries post hoc');return
    df=pd.DataFrame(rows);df.to_csv(out/'fusion_metrics.csv',index=False)
    s=df.groupby(['fold','xgb','graph','variant'])[['alpha','AP','AUROC','recall','FPR']].mean();s.to_csv(out/'fusion_summary.csv');print(s.to_string())
    print('Eligible configurations:',sum(a['eligible'] for a in audits),'/',len(audits));print('Saved:',out)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--threads',type=int,default=8);p.add_argument('--reuse',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);base=Path(__file__).resolve().parent
    xp=a.out/'xgb';gp=a.out/'graph'
    if not a.reuse:
      hashes={}
      for source,target,out,graph in [('cbid_tcag_v9_05_walkforward.py','cbid_tcag_v10_xgb_cal.py',xp,False),('cbid_tcag_v9_06_graph_prototype.py','cbid_tcag_v10_graph_cal.py',gp,True)]:
        # Generated copies must live beside the dependency module.
        hashes[target]=regenerate(base/source,base/target,a.input,out,a.device,a.threads,graph)
      (a.out/'base_script_hashes.json').write_text(json.dumps(hashes,indent=2))
    run_fusion(xp/'predictions.csv',gp/'predictions.csv',a.out)
if __name__=='__main__':main()
