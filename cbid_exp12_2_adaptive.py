#!/usr/bin/env python3
"""Exploratory D-only adaptive score gate; offline completed-flow replay, no live IPS."""
import argparse, collections, hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, torch
from threadpoolctl import threadpool_limits
import cbid_exp11_closed_loop as base
import cbid_exp11_2_risk as risk
import cbid_exp11_3_grid as grid
import cbid_exp11_4_transfer as transfer

LOOKBACK=60  # exact 10-second bins; previous ten minutes
MIN_HISTORY=128
QUANTILE=.90
RATE=.01
CAP=10.
POLICIES=['adaptive_score','adaptive_risk','adaptive_residual_risk']
PENALTIES={'M0_local':5.,'G1_real_peer':20.}
# Defined before D execution. Not chosen using an E/F parameter sweep.
MIN_ATTACK=.025  # exploratory nontrivial performance floor, not a guarantee

class ScoreHistory:
    def __init__(self):self.bins=collections.deque()
    def threshold(self,w):
        while self.bins and self.bins[0][0]<w-LOOKBACK:self.bins.popleft()
        if not self.bins or sum(len(p) for _,p in self.bins)<MIN_HISTORY:return None
        return float(np.quantile(np.concatenate([p for _,p in self.bins]),QUANTILE))
    def add(self,w,p):
        if self.bins and w<=self.bins[-1][0]:raise ValueError('Nonmonotonic score history')
        if len(p):self.bins.append((w,np.asarray(p,float).copy()))

def cost_proxy(obs,pa,pb):
    # Conservative persistence heuristic, NOT a bound on benign future volume.
    past=obs.pair_flows_prev60.to_numpy(float)/np.maximum(1.,obs.pair_active_windows_prev60.to_numpy(float))
    volume=np.maximum(obs.flows.to_numpy(float),past)
    return np.maximum(pb,np.maximum(0.,volume-pa))

def choose(obs,p,theta,credit,policy,penalty,pa=None,pb=None):
    cap=risk.available_actions(credit)
    if theta is None or cap<=0:return []
    gate=p>=theta
    if policy=='adaptive_score':rank=p
    elif policy in POLICIES[1:]:
        if pa is None or pb is None or not np.isfinite(np.column_stack([pa,pb])).all() or (pa<0).any() or (pb<0).any():raise ValueError('Invalid forecasts')
        cost=pb if policy=='adaptive_risk' else cost_proxy(obs,pa,pb)
        rank=pa-penalty*cost;gate &= rank>0
    else:raise ValueError(policy)
    return sorted(np.flatnonzero(gate),key=lambda i:(-rank[i],-p[i],str(obs.src.iloc[i]),str(obs.dst.iloc[i])))[:cap]

def simulate(d,scorer,cols,estimators,policy,penalty,reference=None):
    hist=base.History();sh=ScoreHistory();scheduled=set();prev=None;credit=0.;digest=hashlib.sha256()
    a=b=actions=blocked_windows=admitted_windows=changes=warmup_rows=gate_rows=0
    thresholds=[];cost_rows=0;cost_raised=0
    for w,raw in d.groupby('w',sort=True):
        w=int(w);scheduled=scheduled if prev==w-1 else set()
        blocked=np.array([k in scheduled for k in zip(raw.src,raw.dst)],bool)
        a+=int(raw.attack_flows.to_numpy()[blocked].sum());b+=int((raw.flows-raw.attack_flows).to_numpy()[blocked].sum());blocked_windows+=int(blocked.sum())
        obs=hist.build(raw.loc[~blocked,base.BASE],w);admitted_windows+=len(obs);scheduled=set();prev=w
        if obs.empty:continue
        p=np.asarray(scorer(obs),float)
        if not np.isfinite(p).all():raise ValueError('Nonfinite scores')
        if reference is not None:
            q=np.array([reference[(w,s,t)] for s,t in zip(obs.src,obs.dst)]);changes+=int((np.abs(q-p)>1e-6).sum())
        theta=sh.threshold(w)  # ONLY previous admitted scores
        sh.add(w,p)
        credit=min(CAP,credit+RATE*len(obs));pa=pb=None
        if theta is None:warmup_rows+=len(obs)
        else:
            thresholds.append(theta);gate_rows+=int((p>=theta).sum())
            if policy!='adaptive_score' and risk.available_actions(credit)>0:
                X=risk.risk_features(obs,p,cols);pa,pb=(np.maximum(0,m.predict(X)) for m in estimators)
                if policy=='adaptive_residual_risk':
                    cost=cost_proxy(obs,pa,pb);cost_rows+=len(obs);cost_raised+=int((cost>pb).sum())
        selected=choose(obs,p,theta,credit,policy,penalty,pa,pb)
        credit-=len(selected);scheduled={(obs.src.iloc[i],obs.dst.iloc[i]) for i in selected};actions+=len(scheduled)
        for pair in sorted(scheduled):digest.update(json.dumps([w+1,*pair],separators=(',',':')).encode()+b'\n')
    A=int(d.attack_flows.sum());B=int((d.flows-d.attack_flows).sum())
    assert 0<=a<=A and 0<=b<=B and admitted_windows+blocked_windows==len(d)
    return dict(total_attack=A,total_benign=B,blocked_attack=a,blocked_benign=b,attack_suppression=a/A,benign_disruption=b/B,budget_pass=b/B<=.01,actions=actions,blocked_pair_windows=blocked_windows,admitted_pair_windows=admitted_windows,scores_changed_vs_allow=changes,action_schedule_sha256=digest.hexdigest(),warmup_rows=warmup_rows,score_gate_rows=gate_rows,threshold_median=float(np.median(thresholds)) if thresholds else None,cost_evaluated_rows=cost_rows,cost_raised_rows=cost_raised)

def selftest():
    grid.selftest()
    sh=ScoreHistory();assert sh.threshold(1) is None
    sh.add(10,np.linspace(0,1,128));assert abs(sh.threshold(11)-.9)<1e-12
    assert sh.threshold(70) is not None and sh.threshold(71) is None
    obs=pd.DataFrame(dict(src=['a','b'],dst=['x','x'],flows=[100,10],pair_flows_prev60=[600,60],pair_active_windows_prev60=[6,6]));p=np.array([.95,.91]);pa=np.array([20.,12.]);pb=np.array([.1,.1])
    assert choose(obs,p,.9,1.,'adaptive_risk',5.,pa,pb)==[0]
    assert choose(obs,p,.9,1.,'adaptive_residual_risk',5.,pa,pb)==[1]
    assert choose(obs,p,None,10.,'adaptive_score',5.)==[]
    rows=[]
    for w in range(100,250):
        for s in ['a','b','c','d']:
            rows.append(dict(src=s,dst='x',w=w,flows=2,attack_flows=int(s=='a'),**{k:2 for k in base.NUM},distinct_dst_ports=1))
    raw=pd.DataFrame(rows);swapped=raw.copy();swapped['attack_flows']=swapped.flows-swapped.attack_flows
    class Constant:
        def __init__(self,c):self.c=c
        def predict(self,X):return np.full(len(X),self.c)
    def sc(o):
        assert 'attack_flows' not in o
        return np.where(o.src=='a',.9,.2)
    cols=['flows','pair_flows_prev60','pair_active_windows_prev60'];est=[Constant(20.),Constant(.1)]
    for pol in POLICIES:
        m=simulate(raw,sc,cols,est,pol,5.);n=simulate(swapped,sc,cols,est,pol,5.)
        assert m['actions']>0 and m['blocked_pair_windows']>0 and m['action_schedule_sha256']==n['action_schedule_sha256'] and m['blocked_attack']==n['blocked_benign']
        assert m['warmup_rows']>=MIN_HISTORY
    print('EXP12.2 SELFTEST PASS: strictly past quantile, clock expiry, residual cost, closed-loop label invariance',flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path.home()/'Downloads');ap.add_argument('--out',type=Path);ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    torch.set_num_threads(1);selftest()
    freeze=Path(transfer.__file__).with_name('exp11_4_freeze.json');base.check(freeze,transfer.FREEZE_HASH);cfg=json.loads(freeze.read_text())
    for mod in (base,risk,grid):base.check(mod.__file__,cfg['source_hashes'][Path(mod.__file__).name])
    if a.self_test:return
    if a.out is None:ap.error('--out required')
    if sklearn.__version__!='1.7.2':raise RuntimeError('Use original sklearn 1.7.2 environment')
    path=a.root/'CBID_APRE_V1/05_confirmation/inference/features/pair_windows_2019-04-27.csv';base.check(path,grid.TEST_HASH)
    f,g,cal,th=base.load(a.root);cols=g['local']+g['temporal']+g['relational'];d=risk.input_day(path,'D');arts={};hashes={}
    for seed in base.SEEDS:
        for name in base.MODELS:
            est,h,_=grid.risk_artifact(a.root/'CBID_EXP11_2',seed,name,cols)
            if h!=cfg['risk_artifacts'][f'{seed}_{name}']:raise ValueError('Risk hash mismatch')
            arts[seed,name]=est;hashes[f'{seed}_{name}']=h
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'protocol.json').write_text(json.dumps(dict(stage='D development only; design informed by prior E/F diagnostics; no E/F access by this runner',input_sha256=grid.TEST_HASH,source_sha256=base.sha(__file__),previous_freeze_sha256=transfer.FREEZE_HASH,risk_hashes=hashes,lookback_bins=LOOKBACK,min_history_rows=MIN_HISTORY,quantile=QUANTILE,rate=RATE,cap=CAP,penalties=PENALTIES,policies=POLICIES,min_attack_mean=MIN_ATTACK,budget=.01,seeds=base.SEEDS,python=platform.python_version(),sklearn=sklearn.__version__,torch=torch.__version__),indent=2)+'\n')
    rows=[];risk.RATE=RATE;risk.CAP=CAP
    for seed in base.SEEDS:
        for name in base.MODELS:
            scorer=base.Scorer(f,g,seed,name);est=arts[seed,name];theta=float(th.loc[(seed,name,10),'oracle_threshold'])
            with threadpool_limits(limits=1):_,ref=risk.simulate(d,scorer,theta,'allow',cols,est)
            for policy in POLICIES:
                t=time.monotonic()
                with threadpool_limits(limits=1):r=simulate(d,scorer,cols,est,policy,PENALTIES[name],ref)
                r.update(day='D',seed=seed,model=name,policy=policy,rate=RATE,penalty=PENALTIES[name],wall_seconds=time.monotonic()-t);rows.append(r);pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False);print(json.dumps(r),flush=True)
    s=grid.summarize(rows,3);s['development_pass']=s.feasible & (s.attack_mean>=MIN_ATTACK);s.to_csv(a.out/'summary.csv',index=False);print(s.to_string(index=False));print('COMPLETE 18 adaptive D runs; exploratory only')
if __name__=='__main__':main()
