#!/usr/bin/env python3
"""Frozen E/F inference and open-loop exposure; no fitting or threshold tuning."""
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import cbid_tcag_v10_04_confirmation as tc
from cbid_apre_v1_05_confirm import exposure

PROTOCOL_HASH='28ddf8fb93986db0fa99ee5f79378e3c4ec777b8358c6e9033f78a47ea0b8e2e'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--day',choices=['E','F'],required=True); a=p.parse_args()
    root=Path.home()/'Downloads'; base=root/'CBID_FINAL_AUDIT'; protocol=base/'09_final_protocol/final_protocol.json'
    assert tc.digest(protocol)==PROTOCOL_HASH
    cfg=json.loads(protocol.read_text()); frozen=root/'CBID_TCAG_V10/04_frozen_models'; policy=root/'CBID_APRE_V1/04_frozen_policy'
    mf=frozen/'freeze.json'; pf=policy/'policy_freeze.json'; th=policy/'thresholds.csv'
    assert tc.digest(mf)==cfg['model_freeze_sha256']; assert tc.digest(pf)==cfg['policy_freeze_sha256']
    m=json.loads(mf.read_text()); pol=json.loads(pf.read_text())
    assert m['sources']==tc.sources()==cfg['sources']
    for n,h in m['files'].items(): assert tc.digest(frozen/n)==h,n
    assert tc.digest(th)==pol['thresholds_sha256']
    assert tc.digest(a.archive)==cfg['dataset_sha256']
    assert tc.digest(root/'CBID_TCAG_V10/05_confirmation/predictions.csv')==pol['prediction_sha256']
    assert tc.digest(root/'cbid_apre_v1_04_freeze_policy.py')==pol['script_sha256']
    assert tc.digest(root/'cbid_apre_v1_03_relation_gate.py')==pol['audit_script_sha256']
    assert torch.cuda.is_available(); torch.set_num_threads(8)
    thresholds=pd.read_csv(th,float_precision='round_trip').set_index(['seed','model','horizon_s'])
    models=cfg['models']; seeds=cfg['seeds']; assert thresholds.index.is_unique
    assert set(thresholds.index)=={(s,n,h) for s in seeds for n in models for h in [10,60]}
    assert thresholds.budget.eq(.01).all() and thresholds.extra_delay_s.eq(0).all()
    day=cfg['confirmation_days'][0 if a.day=='E' else 1]; out=base/'10_final_confirmation'/a.day
    out.mkdir(parents=True,exist_ok=False)
    tc.dump(out/'run_protocol.json',dict(day=day,protocol_sha256=PROTOCOL_HASH,script_sha256=tc.digest(__file__),model_hash=cfg['model_freeze_sha256'],policy_hash=cfg['policy_freeze_sha256']))
    tc.C=day
    tc.confirm(SimpleNamespace(frozen=frozen,freeze_sha256=cfg['model_freeze_sha256'],out=out/'inference',archive=a.archive,device='cuda',threads=8))
    report=json.loads((out/'inference/features/feature_manifest.json').read_text())['days'][day]
    assert report['flows']==cfg['expected_flow_counts'][day],report
    d=pd.read_csv(out/'inference/predictions.csv',float_precision='round_trip')
    assert not d.duplicated(['seed','model','src','dst','w']).any()
    assert set(d.seed)==set(seeds) and set(d.model)==set(models)
    rows=[]
    for seed,g in d.groupby('seed'):
        ref=g[g.model==models[0]].set_index(['src','dst','w']).sort_index(); idx=ref.index
        A=ref.attack_flows.to_numpy(); B=ref.flows.to_numpy()-A
        assert (A>=0).all() and (B>=0).all()
        for model in models:
            q=g[g.model==model].set_index(idx.names).sort_index()
            assert q.index.equals(idx) and q[['flows','attack_flows','y']].equals(ref[['flows','attack_flows','y']]) and q.score.between(0,1).all()
            for h in [10,60]:
                theta=float(thresholds.loc[(seed,model,h),'oracle_threshold'])
                for delay in [0,10]:
                    mask=exposure(q.score,idx,h,delay,theta); ae=int(A[mask].sum()); be=int(B[mask].sum()); ar=ae/A.sum() if A.sum() else np.nan; br=be/B.sum() if B.sum() else np.nan
                    rows.append(dict(seed=seed,model=model,horizon_s=h,extra_delay_s=delay,threshold=theta,total_attack=int(A.sum()),total_benign=int(B.sum()),attack_exposed=ae,benign_exposed=be,attack_exposure_rate=ar,benign_exposure_rate=br,budget_pass=bool(br<=.01) if np.isfinite(br) else None))
    r=pd.DataFrame(rows); assert len(r)==144
    r.to_csv(out/'policy_metrics.csv',index=False)
    r.groupby(['model','horizon_s','extra_delay_s'])[['attack_exposure_rate','benign_exposure_rate']].agg(['mean','std']).to_csv(out/'policy_summary.csv')
    print('\n=== SUPPORT ==='); print(json.dumps(report,indent=2))
    print('\n=== PRIMARY APRE G1 ==='); print(r[(r.model=='G1_real_peer')&(r.horizon_s==10)&(r.extra_delay_s==0)].to_string(index=False))
    print('Saved:',out)
if __name__=='__main__': main()
