#!/usr/bin/env python3
"""Exploratory diagnostic of existing predictions; no retraining or day C."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.input/'predictions.csv')
    assert set(d.split)=={'cal','test_A','shift_B'}
    assert d.y.isin([0,1]).all() and np.isfinite(d.score).all()
    assert (d.y==(d.attack_flows>0)).all() and (d.flows>=d.attack_flows).all()
    assert not d.duplicated(['model','split','src','dst','w']).any()
    rows=[]; quant=[]; blocks=[]
    for model,g in d.groupby('model'):
        cal=g[g.split=='cal']; neg=np.sort(cal.loc[cal.y==0,'score'].to_numpy())[::-1]
        assert len(neg)
        for split,h in g.groupby('split'):
            for label,v in h.groupby('y'):
                quant.append(dict(model=model,split=split,y=label,n=len(v),**{f'q{q}':float(v.score.quantile(q/100)) for q in [0,10,50,90,95,99,100]}))
            for alpha in [.01,.05,.10]:
                t=np.nextafter(neg[int(np.floor(alpha*len(neg)))],np.inf)
                alert=h.score>=t; pos=h.y==1; benign=h.flows-h.attack_flows
                rows.append(dict(model=model,split=split,calibration_fpr_budget=alpha,threshold=t,recall=(alert&pos).sum()/pos.sum(),fpr=(alert&~pos).sum()/(~pos).sum(),alerts=int(alert.sum()),mixed_alerts=int((alert&pos&(benign>0)).sum()),retrospective_benign_flows_in_alerts=int(benign[alert].sum()),retrospective_attack_flows_in_alerts=int(h.attack_flows[alert].sum())))
            for b,v in h.groupby(h.w//180):
                blocks.append(dict(model=model,split=split,block=int(b),n=len(v),positives=int(v.y.sum()),AP=average_precision_score(v.y,v.score) if v.y.nunique()==2 else np.nan))
    r=pd.DataFrame(rows); r.to_csv(a.out/'operating_points.csv',index=False)
    pd.DataFrame(quant).to_csv(a.out/'score_quantiles.csv',index=False)
    pd.DataFrame(blocks).to_csv(a.out/'block_diagnostics.csv',index=False)
    print('Calibration-only thresholds; 5% and 10% exploratory, not replacement success criteria.')
    print(r[['model','split','calibration_fpr_budget','threshold','recall','fpr','alerts']].to_string(index=False))
    print('Score quantiles (99%, max):'); print(pd.DataFrame(quant)[['model','split','y','q99','q100']].to_string(index=False))
    print('Retrospective flows in alert windows are NOT blocked flows.')
    print('Saved:',a.out)
if __name__=='__main__': main()
