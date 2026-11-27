#!/usr/bin/env python3
"""Development only: fixed 50/50 fusion using entire original calibration split.
No learned parameters or test-dependent threshold. Reads V10.02 saved scores only.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from cbid_tcag_v10_02_fusion import load, join, combine
from cbid_tcag_v9_03_relation_gate import threshold, metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    xp=a.input/'xgb'/'predictions.csv'; gp=a.input/'graph'/'predictions.csv'
    rows=[]; audits=[]
    for xm in ('M0_local','M2_context','M3_full'):
        x=load(xp,xm)
        for gm in ('G0_no_message','G1_real_peer','G2_shuffled_peer','G3_context_mlp'):
            d=join(x,load(gp,gm))
            for (fold,seed),v in d.groupby(['fold','seed']):
                if fold not in (12,18): raise ValueError('Unexpected fold')
                cal=v[v.split=='cal']; test=v[v.split=='test']
                start=int(pd.Timestamp('2019-04-25',tz='UTC').timestamp())//10
                cut=start+int(fold)*360
                if not cal.w.between(cut-720+12,cut-13).all() or not test.w.between(cut+12,cut+2159).all(): raise ValueError('Unexpected temporal split')
                counts={k:[int(q.y.sum()),int((q.y==0).sum())] for k,q in [('cal',cal),('test',test)]}
                if min(counts['cal'])<100: raise ValueError('Insufficient calibration')
                audits.append(dict(fold=int(fold),seed=int(seed),xgb=xm,graph=gm,counts=counts))
                for variant,alpha in [('xgb',0.),('graph',1.),('fixed',.5)]:
                    t=threshold(cal.y.to_numpy(),combine(cal,alpha),.01)
                    for split,q in [('cal',cal),('test',test)]:
                        rows.append(dict(fold=int(fold),seed=int(seed),xgb=xm,graph=gm,variant=variant,split=split,alpha=alpha,**metrics(q.y.to_numpy(),combine(q,alpha),t)))
    df=pd.DataFrame(rows); df.to_csv(a.out/'metrics.csv',index=False)
    s=df[df.split=='test'].groupby(['fold','xgb','graph','variant'])[['AP','AUROC','recall','FPR']].mean()
    s.to_csv(a.out/'summary.csv'); (a.out/'audit.json').write_text(json.dumps(audits,indent=2))
    print(s.loc[(slice(None),'M3_full',slice(None),'fixed'),:].to_string())
    print('\nPrimary comparison (M3 + real peer):')
    print(s.loc[(slice(None),'M3_full','G1_real_peer',slice(None)),:].to_string())
    print('\nStrict operational gate, M3 + real peer:')
    print(df.query("split=='test' and xgb=='M3_full' and graph=='G1_real_peer'")[[ 'fold','seed','variant','recall','FPR','TP','FP']].to_string(index=False))
    print('Development only. Saved:',a.out)
if __name__=='__main__': main()
