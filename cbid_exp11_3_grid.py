#!/usr/bin/env python3
"""Exploratory parameter sensitivity on previously explored D; no live interventions."""
import argparse, hashlib, itertools, json, math, time, platform
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import torch
import sklearn
from threadpoolctl import threadpool_limits
import cbid_exp11_2_risk as risk
import cbid_exp11_closed_loop as base

RISK_HASH='25d623d81691785326fba8f74b63914b3c092993715ed41a776c20d4965958a5'
BASE_HASH='4d98421298c81278d5df9686aa5511e3c53f03ecf061746e3c455e3e3fb86d20'
TRAIN_HASH='a61e0a6bd8f8c26d726637efbcff90c9e114c833e149d26f3462a37f322b209c'
TEST_HASH='5778eb0d9fa26ec7a1a969011c9dc1b3d4f8837c62f4aecea225087c69c9af01'
MATCH=['blocked_attack','blocked_benign','actions','blocked_pair_windows','scores_changed_vs_allow','action_schedule_sha256']

def summarize(rows, expected):
    d=pd.DataFrame(rows)
    if d.empty:return d
    s=d.groupby(['model','policy','rate','penalty'],dropna=False).agg(n=('seed','nunique'),attack_mean=('attack_suppression','mean'),attack_sd=('attack_suppression','std'),benign_mean=('benign_disruption','mean'),benign_max=('benign_disruption','max'),budget_passes=('budget_pass','sum'),mean_actions=('actions','mean')).reset_index()
    s['all_requested_seeds']=s.n==expected
    s['feasible']=s.all_requested_seeds & (s.budget_passes==expected)
    s['pareto_feasible']=False
    # Global frontier across complete, feasible configurations; use worst observed benign rate.
    for i,a in s[s.feasible].iterrows():
        b=s[s.feasible]
        dominated=((b.attack_mean>=a.attack_mean)&(b.benign_max<=a.benign_max)&((b.attack_mean>a.attack_mean)|(b.benign_max<a.benign_max))).any()
        s.loc[i,'pareto_feasible']=not dominated
    return s.sort_values(['feasible','attack_mean','benign_max'],ascending=[False,False,True])

def selftest():
    risk.selftest()
    rows=[]
    for q,a,b in [(0.,0.,0.),(.01,.04,.009),(.02,.05,.011),(.03,.03,.0095),(.04,.02,.005)]:
        for seed in [42,43]:rows.append(dict(model='test',policy='test',rate=q,penalty=10,seed=seed,attack_suppression=a,benign_disruption=b,budget_pass=b<=.01,actions=0))
    s=summarize(rows,2).set_index('rate');assert s.loc[.01,'pareto_feasible'] and not s.loc[.02,'feasible'] and not s.loc[.03,'pareto_feasible'] and s.loc[.04,'pareto_feasible']
    print('EXP11.3 SELFTEST PASS: complete-seed budget and Pareto classification',flush=True)

def risk_artifact(parent,seed,name,cols):
    folder=parent/('D_pilot_fixed' if seed==42 else 'D_remaining')
    protocol=json.loads((folder/'protocol.json').read_text())
    for k,v in {'stage':'exploratory','train':'C','test':'D','train_sha256':TRAIN_HASH,'test_sha256':TEST_HASH,'script_sha256':RISK_HASH,'base_script_sha256':BASE_HASH,'frozen_models_sha256':base.MODEL_HASH,'frozen_policy_sha256':base.POLICY_HASH}.items():
        if protocol[k]!=v:raise ValueError('Protocol mismatch '+k)
    if seed not in protocol['seeds'] or name not in protocol['models']:raise ValueError('Seed/model mismatch')
    artifact=folder/f'risk_{seed}_{name}.joblib'; h=(folder/f'risk_{seed}_{name}.sha256').read_text().split()[0]
    base.check(artifact,h)
    # Joblib deserialization: use only the user's own previously produced trusted files.
    obj=joblib.load(artifact)
    if obj['seed']!=seed or obj['train_sha256']!=TRAIN_HASH or obj['features']!=cols+['detector_score'] or len(obj['models'])!=2:raise ValueError('Risk artifact metadata mismatch')
    prev=pd.read_csv(folder/'metrics.csv').set_index(['seed','model','policy'])
    return obj['models'],h,prev

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');p.add_argument('--root',type=Path,default=Path.home()/'Downloads');p.add_argument('--test',type=Path);p.add_argument('--out',type=Path);p.add_argument('--seeds',type=int,nargs='+',choices=base.SEEDS,default=[42]);p.add_argument('--rates',type=float,nargs='+',default=[.005,.01,.02,.05]);p.add_argument('--penalties',type=float,nargs='+',default=[5,10,20,50]);args=p.parse_args()
    torch.set_num_threads(1);selftest()
    if args.self_test:return
    if args.test is None or args.out is None:p.error('--test and --out required')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.rates))!=len(args.rates) or len(set(args.penalties))!=len(args.penalties):p.error('Duplicate parameter values')
    if not all(math.isfinite(q) and 0<q<=.05 for q in args.rates) or not all(math.isfinite(l) and l>0 for l in args.penalties):p.error('Invalid grid')
    if .01 not in args.rates or 10. not in args.penalties:p.error('Include reference q=.01 and lambda=10')
    if sklearn.__version__!='1.7.2':raise RuntimeError('Use original sklearn 1.7.2 environment for joblib artifacts')
    base.check(risk.__file__,RISK_HASH);base.check(base.__file__,BASE_HASH);base.check(args.test,TEST_HASH)
    f,g,cal,th=base.load(args.root);cols=g['local']+g['temporal']+g['relational'];d=risk.input_day(args.test,'D')
    args.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(stage='exploratory',test='D already explored',train='C saved risk estimates, no refit',test_sha256=TEST_HASH,train_sha256=TRAIN_HASH,script_sha256=base.sha(__file__),risk_script_sha256=RISK_HASH,base_script_sha256=BASE_HASH,model_freeze_sha256=base.MODEL_HASH,policy_freeze_sha256=base.POLICY_HASH,seeds=args.seeds,rates=args.rates,penalties=args.penalties,credit_cap=10.,horizon_seconds=10,budget=.01,python=platform.python_version(),sklearn=sklearn.__version__,risk_artifacts={})
    def save_protocol():(args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    save_protocol();rows=[]
    # Use the exact existing implementation; only RATE and PENALTY vary.
    for seed in args.seeds:
        for name in base.MODELS:
            scorer=base.Scorer(f,g,seed,name);theta=float(th.loc[(seed,name,10),'oracle_threshold']);est,h,prev=risk_artifact(args.root/'CBID_EXP11_2',seed,name,cols);protocol['risk_artifacts'][f'{seed}_{name}']=h;save_protocol()
            risk.RATE=.01;risk.PENALTY=10.;risk.CAP=10.
            with threadpool_limits(limits=1):allow,ref=risk.simulate(d,scorer,theta,'allow',cols,est)
            configs=[('score_cic',.01,10.),('risk_gated_cic',.01,10.)]+[('score_cic',q,10.) for q in args.rates if q!=.01]+[('risk_gated_cic',q,l) for q,l in itertools.product(args.rates,args.penalties) if (q,l)!=(.01,10.)]
            for policy,q,l in configs:
                risk.RATE=q;risk.PENALTY=l;t=time.monotonic()
                with threadpool_limits(limits=1):result,_=risk.simulate(d,scorer,theta,policy,cols,est,ref)
                match=None
                if q==.01 and l==10.:
                    old=prev.loc[(seed,name,policy)];match=all(result[k]==old[k] for k in MATCH)
                    if not match:raise RuntimeError(f'Baseline reproduction failed {seed} {name} {policy}')
                result.update(day='2019-04-27',seed=seed,model=name,policy=policy,rate=q,penalty=l,threshold=theta,reference_match=match,risk_sha256=h,wall_seconds=time.monotonic()-t)
                rows.append(result);pd.DataFrame(rows).to_csv(args.out/'metrics.csv',index=False);summarize(rows,len(args.seeds)).to_csv(args.out/'summary.csv',index=False);print(json.dumps(result),flush=True)
    summary=summarize(rows,len(args.seeds));print(summary.to_string(index=False));print('COMPLETE',len(rows),'configurations; exploratory only')
if __name__=='__main__':main()
