#!/usr/bin/env python3
"""Exploratory B-only TCAG prototype: gated learned peer messages on pair-window nodes.
No April 26 access. Same fixed folds, seeds and training settings for all variants.
Peer graph is an implicit directed clique per (destination, window), excluding self.
Shuffle preserves group sizes per clock window, not each node's original degree.
"""
import argparse,json,copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_curve
from cbid_tcag_v9_03_relation_gate import threshold,metrics

class Model(nn.Module):
    def __init__(self,n,hidden=64):
        super().__init__()
        self.local=nn.Sequential(nn.Linear(n,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.message=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU())
        self.gate=nn.Sequential(nn.Linear(2*hidden,hidden),nn.Sigmoid())
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,x,group,ng,mode):
        h=self.local(x); v=self.message(h)
        if mode=='none':
            m=torch.zeros_like(v)
        else:
            sums=v.new_zeros((ng,v.shape[1])).index_add_(0,group,v)
            count=v.new_zeros((ng,1)).index_add_(0,group,v.new_ones((len(v),1)))
            deg=count[group]-1
            m=(sums[group]-v)/deg.clamp_min(1)
            m=m*(deg>0)
        return self.head(h+self.gate(torch.cat([h,m],1))*m).squeeze(-1)

def pack(d,cols,mean,std,mode,seed,maxrows=4096):
    # Entire clock windows in a batch; no graph crosses a batch boundary.
    d=d.sort_values(['w','src','dst']).reset_index(drop=True)
    x=((np.log1p(d[cols].to_numpy(dtype=np.float64))-mean)/std).clip(-12,12).astype('float32')
    groups=pd.factorize(pd.MultiIndex.from_frame(d[['w','dst']]))[0].astype('int64')
    rng=np.random.default_rng(seed)
    windows=list(d.groupby('w',sort=False).indices.values())
    if mode=='shuffle':
        for ix in windows: groups[ix]=groups[rng.permutation(ix)]
    chunks=[]; current=[]; size=0
    for ix in windows:
        if current and size+len(ix)>maxrows:
            chunks.append(np.concatenate(current));current=[];size=0
        current.append(ix);size+=len(ix)
    if current:chunks.append(np.concatenate(current))
    batches=[]
    for ix in chunks:
        _,g=np.unique(groups[ix],return_inverse=True)
        batches.append((torch.from_numpy(x[ix]),torch.from_numpy(g),int(g.max()+1),torch.tensor(d.y.to_numpy()[ix],dtype=torch.float32)))
    return d,batches

def scores(model,batches,device,mode):
    model.eval();out=[]
    with torch.no_grad():
        for x,g,n,y in batches:out.append(torch.sigmoid(model(x.to(device),g.to(device),n,mode)).cpu().numpy())
    return np.concatenate(out).astype(float)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--epochs',type=int,default=30);p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA unavailable')
    torch.set_num_threads(a.threads);a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.input/'feature_manifest.json').read_text());groups=manifest['feature_groups'];cols=groups['local']+groups['temporal'];full=cols+groups['relational']
    if len(full)!=30 or len(set(full))!=30 or set(full)&{'src','dst','w','y','attack_flows','Label','Attack'}:raise ValueError('Bad feature manifest')
    day='2019-04-25';start=int(pd.Timestamp(day,tz='UTC').timestamp())//10
    d=pd.read_csv(a.input/f'pair_windows_{day}.csv',usecols=['src','dst','w','y','attack_flows']+full)
    if not d.w.between(start,start+8639).all() or d.duplicated(['src','dst','w']).any() or not d.y.isin([0,1]).all() or not (d.y==(d.attack_flows>0)).all() or not np.isfinite(d[full]).all().all() or (d[full]<0).any().any():raise ValueError('Invalid data')
    rows=[];pred=[];audit=[]
    for h in (6,12,18):
        cut=start+h*360;cal=cut-720;end=cut+2160
        ds={'train':d[d.w<cal-12],'cal':d[(d.w>=cal+12)&(d.w<cut-12)],'test':d[(d.w>=cut+12)&(d.w<end)]}
        counts={k:[int(v.y.sum()),int((v.y==0).sum())] for k,v in ds.items()};eligible=all(min(v)>=100 for v in counts.values());audit.append(dict(fold=h,counts=counts,eligible=eligible));print(audit[-1],flush=True)
        if not eligible:continue
        for seed in (42,43,44):
            for name,mode,features in [('G0_no_message','none',cols),('G1_real_peer','real',cols),('G2_shuffled_peer','shuffle',cols),('G3_context_mlp','none',full)]:
                torch.manual_seed(seed);np.random.seed(seed)
                if a.device=='cuda':torch.cuda.manual_seed_all(seed)
                z=np.log1p(ds['train'][features].to_numpy(dtype=float));mean=z.mean(0);std=z.std(0);std[std<1e-8]=1
                data={k:pack(v,features,mean,std,mode,seed+100*i) for i,(k,v) in enumerate(ds.items())}
                model=Model(len(features)).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                pos,neg=counts['train'];lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,device=a.device));rng=np.random.default_rng(seed)
                batches=data['train'][1]
                for epoch in range(a.epochs):
                    model.train()
                    for i in rng.permutation(len(batches)):
                        x,g,n,y=batches[i];opt.zero_grad();loss=lossfn(model(x.to(a.device),g.to(a.device),n,mode),y.to(a.device));loss.backward();opt.step()
                pc=scores(model,data['cal'][1],a.device,mode);t=threshold(data['cal'][0].y.to_numpy(),pc)
                for k in ('cal','test'):
                    frame,bs=data[k];y=frame.y.to_numpy();pr=pc if k=='cal' else scores(model,bs,a.device,mode);fpr,tpr,_=roc_curve(y,pr);r=dict(fold=h,seed=seed,model=name,split=k,**metrics(y,pr,t),oracle_recall_at_fpr01=float(tpr[fpr<=.01].max()));rows.append(r);print(json.dumps(r),flush=True)
                    if k=='test':
                        q=frame[['src','dst','w','y','attack_flows','flows']].copy();q['fold']=h;q['seed']=seed;q['model']=name;q['score']=pr;q['alert']=pr>=t;pred.append(q)
    (a.out/'split_audit.json').write_text(json.dumps(audit,indent=2));(a.out/'protocol.json').write_text(json.dumps(dict(epochs=a.epochs,hidden=64,seeds=[42,43,44],scope='exploratory B only',graph='same destination/window peer clique; no self',shuffle='group assignment permutation within window',calibration_fpr=.01),indent=2))
    if not rows:raise SystemExit('NO-GO: no eligible folds')
    pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False);pd.concat(pred,ignore_index=True).to_csv(a.out/'predictions.csv',index=False)
    s=pd.DataFrame(rows).query("split=='test'").groupby(['fold','model'])[['AP','AUROC','recall','FPR','oracle_recall_at_fpr01']].mean();s.to_csv(a.out/'summary.csv');print(s.to_string());print('Saved:',a.out)
if __name__=='__main__':main()
