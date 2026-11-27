#!/usr/bin/env python3
"""Evaluate frozen APRE exposure policies on D. No fitting or threshold selection."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cbid_tcag_v10_04_confirmation as tc
from cbid_apre_v1_03_relation_gate import MODELS

MODEL_HASH='92198e78d35047750afc34edf96c9f482a81a746c302b7170fc8a857cbc2952e'
POLICY_HASH='d148b49e190382565810ca580d598a853fe769ad50a841250f306becc2aaf023'
DAY='2019-04-27'

def exposure(score, idx, horizon, delay, threshold):
    src,dst,w=[idx.get_level_values(k) for k in ['src','dst','w']]
    active=np.zeros(len(idx),dtype=bool)
    if np.isposinf(threshold): return active
    for k in range(1+delay//10,1+delay//10+horizon//10):
        keys=pd.MultiIndex.from_arrays([src,dst,w-k],names=idx.names)
        active |= score.reindex(keys,fill_value=-np.inf).to_numpy() >= threshold
    return active

def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,required=True);p.add_argument('--policy',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(a.threads)
    if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    f=a.frozen/'freeze.json'; pf=a.policy/'policy_freeze.json'; t=a.policy/'thresholds.csv'; here=Path(__file__).resolve().parent
    if tc.digest(f)!=MODEL_HASH or tc.digest(pf)!=POLICY_HASH: raise ValueError('Manifest hash mismatch')
    m=json.loads(pf.read_text())
    for path,h in [(t,m['thresholds_sha256']),(a.calibration/'predictions.csv',m['prediction_sha256']),(here/'cbid_apre_v1_04_freeze_policy.py',m['script_sha256']),(here/'cbid_apre_v1_03_relation_gate.py',m['audit_script_sha256'])]:
        if tc.digest(path)!=h: raise ValueError('Hash mismatch: '+str(path))
    if m['original_model_freeze_sha256']!=MODEL_HASH or m['candidate_test_day']!=DAY: raise ValueError('Wrong protocol')
    thresholds=pd.read_csv(t,float_precision='round_trip').set_index(['seed','model','horizon_s'])
    expected={(s,n,h) for s in [42,43,44] for n in MODELS for h in [10,60]}
    if len(thresholds)!=72 or set(thresholds.index)!=expected or not thresholds.index.is_unique or not thresholds.budget.eq(.01).all() or not thresholds.extra_delay_s.eq(0).all(): raise ValueError('Invalid threshold grid')
    v=thresholds.oracle_threshold.to_numpy(); assert ((v>=0)&((v<=1)|np.isposinf(v))).all()
    if a.out.exists(): raise SystemExit('Output exists; no overwrite')
    a.out.mkdir(parents=True)
    tc.dump(a.out/'protocol.json',dict(day=DAY,model_hash=MODEL_HASH,policy_hash=POLICY_HASH,evaluation_script_sha256=tc.digest(__file__),rule='frozen thresholds; open-loop completed-flow exposure; no tuning',primary_model='G1_real_peer',primary_horizon_s=10,primary_delay_s=0))
    # Reuse exact frozen inference path, changing only requested calendar day.
    tc.C=DAY
    a.freeze_sha256=MODEL_HASH
    root=a.out; a.out=root/'inference'
    tc.confirm(a)
    d=pd.read_csv(a.out/'predictions.csv',float_precision='round_trip')
    assert not d.duplicated(['seed','model','src','dst','w']).any()
    assert set(d.seed)=={42,43,44} and set(d.model)==set(MODELS)
    rows=[]
    for seed,g in d.groupby('seed'):
        ref=g[g.model==MODELS[0]].set_index(['src','dst','w']).sort_index(); idx=ref.index
        attack=ref.attack_flows.to_numpy(); benign=ref.flows.to_numpy()-attack
        assert (attack>=0).all() and (benign>=0).all()
        for model in MODELS:
            q=g[g.model==model].set_index(idx.names).sort_index()
            assert q.index.equals(idx) and q[['flows','attack_flows','y']].equals(ref[['flows','attack_flows','y']]) and q.score.between(0,1).all()
            for h in [10,60]:
                theta=float(thresholds.loc[(seed,model,h),'oracle_threshold'])
                for delay in [0,10]:
                    mask=exposure(q.score,idx,h,delay,theta); A=int(attack[mask].sum()); B=int(benign[mask].sum()); ar=A/attack.sum() if attack.sum() else np.nan; br=B/benign.sum() if benign.sum() else np.nan
                    rows.append(dict(seed=seed,model=model,horizon_s=h,extra_delay_s=delay,threshold=theta,total_attack=int(attack.sum()),total_benign=int(benign.sum()),attack_exposed=A,benign_exposed=B,attack_exposure_rate=ar,benign_exposure_rate=br,budget_pass=bool(br<=.01) if np.isfinite(br) else False))
    r=pd.DataFrame(rows); assert len(r)==144
    r.to_csv(root/'policy_metrics.csv',index=False)
    s=r.groupby(['model','horizon_s','extra_delay_s'])[['attack_exposure_rate','benign_exposure_rate','budget_pass']].agg(['mean','std','sum']);s.to_csv(root/'policy_summary.csv')
    primary=r[(r.horizon_s==10)&(r.extra_delay_s==0)]
    print('\n=== PRIMARY: 10 s, no additional delay ==='); print(primary.groupby('model')[['attack_exposure_rate','benign_exposure_rate','budget_pass']].mean().to_string());print(primary.to_string(index=False)); print('Saved:',root)
if __name__=='__main__': main()
