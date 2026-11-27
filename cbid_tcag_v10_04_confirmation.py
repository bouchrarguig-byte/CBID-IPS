#!/usr/bin/env python3
"""Two-stage confirmation: freeze on B, then evaluate C without fitting."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from xgboost import XGBClassifier
import cbid_tcag_v9_02_temporal_features as extract
from cbid_tcag_v9_03_relation_gate import threshold, metrics, perturb
from cbid_tcag_v9_06_graph_prototype import Model, pack, scores

SEEDS=(42,43,44)
B='2019-04-25'; C='2019-04-26'
KEY=['w','src','dst']

def digest(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def dump(p,obj): Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False))

def daystart(day): return int(pd.Timestamp(day,tz='UTC').timestamp())//10

def read_day(p,day,cols):
    d=pd.read_csv(p/f'pair_windows_{day}.csv').sort_values(KEY).reset_index(drop=True)
    if d.empty or d.duplicated(KEY).any() or not d.w.between(daystart(day),daystart(day)+8639).all() or not d.y.isin([0,1]).all() or not (d.y==(d.attack_flows>0)).all() or not np.isfinite(d[cols]).all().all() or (d[cols]<0).any().any(): raise ValueError('Invalid data '+day)
    return d

def specs(g):
    L,T,R=[g[k] for k in ('local','temporal','relational')]
    return {'M0_local':L,'M1_temporal':L+T,'M2_context':L+R,'M3_full':L+T+R}, {'G0_no_message':('none',L+T),'G1_real_peer':('real',L+T),'G2_shuffled_peer':('shuffle',L+T),'G3_context_mlp':('none',L+T+R)}

def hybrids(p):
    for i,name in enumerate(('G0_no_message','G1_real_peer','G2_shuffled_peer','G3_context_mlp')): p[f'H{i}_fixed']=.5*(p['M3_full']+p[name])
    return p

def sources():
    here=Path(__file__).resolve().parent
    return {n:digest(here/n) for n in (Path(__file__).name,'cbid_tcag_v9_02_temporal_features.py','cbid_tcag_v9_03_relation_gate.py','cbid_tcag_v9_06_graph_prototype.py')}

def freeze(a):
    if a.out.exists(): raise SystemExit('Use a new output directory; no overwrite')
    m=json.loads((a.input/'feature_manifest.json').read_text()); g=m['feature_groups']; cols=sum([g[k] for k in ('local','temporal','relational')],[])
    if m['sha256']!=extract.EXPECTED or len(cols)!=30 or len(set(cols))!=30 or set(cols)&{'y','Label','Attack','attack_flows','src','dst','w'}: raise ValueError('Manifest invalid')
    d=read_day(a.input,B,cols); s=daystart(B)
    tr=d[d.w<s+16*360-12].reset_index(drop=True); ca=d[(d.w>=s+16*360+12)&(d.w<s+18*360-12)].reset_index(drop=True)
    counts={k:[int(v.y.sum()),int((v.y==0).sum())] for k,v in [('train',tr),('cal',ca)]}
    if counts!={'train':[29493,51878],'cal':[4293,5314]}: raise ValueError('B split differs from prior experiment')
    a.out.mkdir(parents=True); xs,gs=specs(g); records=[]
    for seed in SEEDS:
        pred={}; y=tr.y.to_numpy(); weights=np.where(y==1,len(y)/(2*y.sum()),len(y)/(2*(len(y)-y.sum())))
        for name,features in xs.items():
            model=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=.05,subsample=.8,colsample_bytree=.8,random_state=seed,n_jobs=a.threads,tree_method='hist',device=a.device,eval_metric='logloss')
            model.fit(tr[features],y,sample_weight=weights); model.save_model(a.out/f'{seed}_{name}.json'); pred[name]=model.predict_proba(ca[features])[:,1].astype(float)
        for name,(mode,features) in gs.items():
            torch.manual_seed(seed); np.random.seed(seed)
            if a.device=='cuda': torch.cuda.manual_seed_all(seed)
            z=np.log1p(tr[features].to_numpy(dtype=float)); mean=z.mean(0); std=z.std(0); std[std<1e-8]=1
            _,batches=pack(tr,features,mean,std,mode,seed); frame,cb=pack(ca,features,mean,std,mode,seed+100)
            if not frame[KEY].equals(ca[KEY]): raise ValueError('Alignment')
            model=Model(len(features)).to(a.device); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001); pos,neg=counts['train']; lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,device=a.device)); rng=np.random.default_rng(seed)
            for epoch in range(30):
                model.train()
                for i in rng.permutation(len(batches)):
                    x,gp,n,yb=batches[i]; opt.zero_grad(); loss=lossfn(model(x.to(a.device),gp.to(a.device),n,mode),yb.to(a.device)); loss.backward(); opt.step()
            torch.save(model.state_dict(),a.out/f'{seed}_{name}.pt'); np.savez(a.out/f'{seed}_{name}.npz',mean=mean,std=std); pred[name]=scores(model,cb,a.device,mode)
        for name,p in hybrids(pred).items():
            t=threshold(ca.y.to_numpy(),p); records.append(dict(seed=seed,model=name,**metrics(ca.y.to_numpy(),p,t)))
        print('Frozen seed',seed,flush=True)
    pd.DataFrame(records).to_csv(a.out/'calibration.csv',index=False)
    manifest=dict(day_train=B,day_test=C,counts=counts,feature_groups=g,seeds=list(SEEDS),alpha=.5,epochs=30,calibration_fpr=.01,archive_sha256=extract.EXPECTED,development_sha256=digest(a.input/f'pair_windows_{B}.csv'),sources=sources(),files={p.name:digest(p) for p in a.out.iterdir() if p.is_file()})
    dump(a.out/'freeze.json',manifest); print('FREEZE SHA256:',digest(a.out/'freeze.json'))

def confirm(a):
    f=a.frozen/'freeze.json'
    if digest(f)!=a.freeze_sha256: raise ValueError('Freeze hash mismatch')
    m=json.loads(f.read_text());
    if m['sources']!=sources() or any(digest(a.frozen/n)!=h for n,h in m['files'].items()): raise ValueError('Frozen files/code changed')
    if a.out.exists(): raise SystemExit('Confirmation output already exists; no rerun')
    a.out.mkdir(parents=True); dump(a.out/'access_started.json',{'freeze_sha256':a.freeze_sha256,'day':C})
    extract.DAYS=(C,); old=sys.argv; sys.argv=['extract','--archive',str(a.archive),'--out',str(a.out/'features'),'--chunksize','250000']
    try: extract.main()
    finally: sys.argv=old
    g=m['feature_groups']; cols=sum([g[k] for k in ('local','temporal','relational')],[]); cm=json.loads((a.out/'features'/'feature_manifest.json').read_text())
    if cm['feature_groups']!=g: raise ValueError('Feature order changed')
    d=read_day(a.out/'features',C,cols); xs,gs=specs(g); cal=pd.read_csv(a.frozen/'calibration.csv'); rows=[]; predictions=[]
    for seed in SEEDS:
        pred={}
        for name,features in xs.items():
            model=XGBClassifier(); model.load_model(a.frozen/f'{seed}_{name}.json'); model.set_params(device=a.device,n_jobs=a.threads); pred[name]=model.predict_proba(d[features])[:,1].astype(float)
        for name,(mode,features) in gs.items():
            norm=np.load(a.frozen/f'{seed}_{name}.npz'); frame,bs=pack(d,features,norm['mean'],norm['std'],mode,seed+200)
            if not frame[KEY].equals(d[KEY]): raise ValueError('Alignment')
            model=Model(len(features)).to(a.device); model.load_state_dict(torch.load(a.frozen/f'{seed}_{name}.pt',map_location=a.device,weights_only=True)); pred[name]=scores(model,bs,a.device,mode)
        for name,p in hybrids(pred).items():
            t=float(cal.loc[(cal.seed==seed)&(cal.model==name),'threshold'].iloc[0]); r=dict(seed=seed,model=name,**metrics(d.y.to_numpy(),p,t)); rows.append(r)
            q=d[KEY+['y','flows','attack_flows']].copy(); q['seed']=seed;q['model']=name;q['score']=p;q['alert']=p>=t;predictions.append(q)
    df=pd.DataFrame(rows);df.to_csv(a.out/'metrics.csv',index=False);pd.concat(predictions,ignore_index=True).to_csv(a.out/'predictions.csv',index=False)
    summary=df.groupby('model')[['AP','AUROC','recall','FPR']].agg(['mean','std']);summary.to_csv(a.out/'summary.csv');print(summary.to_string());print(df[['seed','model','recall','FPR','TP','FP']].to_string(index=False));print('Saved:',a.out)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','confirm']);p.add_argument('--input',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--frozen',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--freeze-sha256');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--threads',type=int,default=8);a=p.parse_args();torch.set_num_threads(a.threads)
    if a.device=='cuda' and not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
    if a.stage=='freeze':
        if not a.input: p.error('--input required')
        freeze(a)
    else:
        if not (a.frozen and a.archive and a.freeze_sha256): p.error('--frozen, --archive, --freeze-sha256 required')
        confirm(a)
if __name__=='__main__':main()
