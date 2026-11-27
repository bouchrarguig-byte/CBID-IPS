"""Retrospective, flow-count weighted exposure audit; never a deployment threshold."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
KEY=['seed','src','dst','w']
MODELS=['M0_local','M1_temporal','M2_context','M3_full','G0_no_message','G1_real_peer','G2_shuffled_peer','G3_context_mlp','H0_fixed','H1_fixed','H2_fixed','H3_fixed']
BUDGETS=[.001,.005,.01,.02,.05]
def frontier(score,a,b):
    valid=np.isfinite(score)
    q=pd.DataFrame({'score':score[valid],'a':a[valid],'b':b[valid]}).groupby('score').sum().sort_index(ascending=False).cumsum()
    for budget in BUDGETS:
        z=q[q.b<=budget*b.sum()]
        yield (budget, float(z.index[-1]),float(z.a.iloc[-1]/a.sum()),float(z.b.iloc[-1]/b.sum())) if len(z) else (budget,float('inf'),0.,0.)
def audit(d):
    rows=[]
    for seed,g in d.groupby('seed'):
        ref=g[g.model==MODELS[0]].set_index(KEY[1:]).sort_index()
        assert len(ref)==91893 and ref.flows.sum()==4827456 and ref.attack_flows.sum()==2686781
        a=ref.attack_flows.to_numpy(); b=ref.flows.to_numpy()-a
        assert (a>=0).all() and (b>=0).all()
        idx=ref.index; src,dst,w=[idx.get_level_values(k) for k in KEY[1:]]
        for model in MODELS:
            v=g[g.model==model].set_index(KEY[1:]).sort_index()
            assert idx.equals(v.index) and v[['flows','attack_flows','y']].equals(ref[['flows','attack_flows','y']])
            s=v.score; assert s.between(0,1).all()
            for delay in [0,1]:
                eff=np.full(len(ref),-np.inf)
                for k in range(1,7):
                    keys=pd.MultiIndex.from_arrays([src,dst,w-k-delay],names=idx.names)
                    eff=np.maximum(eff,s.reindex(keys,fill_value=-np.inf).to_numpy())
                    if k not in (1,3,6): continue
                    for budget,t,ar,br in frontier(eff,a,b):
                        rows.append(dict(seed=seed,model=model,horizon_s=k*10,extra_delay_s=delay*10,budget=budget,oracle_threshold=t,attack_exposure_rate=ar,benign_exposure_rate=br))
    return pd.DataFrame(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    d=pd.read_csv(args.input/'predictions.csv',usecols=KEY+['model','score','flows','attack_flows','y']);assert not d.duplicated(KEY+['model']).any();assert set(d.seed)=={42,43,44};assert set(d.model)==set(MODELS)
    r=audit(d);args.out.mkdir(parents=True,exist_ok=True);r.to_csv(args.out/'oracle_metrics.csv',index=False)
    cols=['attack_exposure_rate','benign_exposure_rate'];s=r.groupby(['model','horizon_s','extra_delay_s','budget'])[cols].agg(['mean','std']);s.to_csv(args.out/'oracle_summary.csv')
    print('=== 1% budget, no additional delay (retrospective) ===');print(s.xs((0,.01),level=('extra_delay_s','budget')).round(6).to_string())
    v=r.pivot(index=['seed','horizon_s','extra_delay_s','budget'],columns='model',values='attack_exposure_rate'); pairs=[('G1_real_peer','G0_no_message'),('G1_real_peer','G2_shuffled_peer'),('G1_real_peer','G3_context_mlp'),('H1_fixed','H0_fixed'),('H1_fixed','H2_fixed'),('H1_fixed','H3_fixed')];delta=pd.DataFrame({f'{x} minus {y}':v[x]-v[y] for x,y in pairs});delta.to_csv(args.out/'paired_deltas.csv');print('\n=== Paired differences, 1% budget, no additional delay ===');print(delta.xs((0,.01),level=('extra_delay_s','budget')).round(6).to_string());print('Saved:',args.out)
if __name__=='__main__':main()
