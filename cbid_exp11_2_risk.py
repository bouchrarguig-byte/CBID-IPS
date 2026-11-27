#!/usr/bin/env python3
"""Exploratory EXP-11.2: frozen detector + next-window count risk ranking.
Fit on C no-action trajectories; compare policies on previously explored D.
No live network actions, no test labels in decisions, no claim of budget guarantee.
"""
import argparse, hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import torch
import cbid_exp11_closed_loop as base

RATE=.01; CAP=10.; PENALTY=10.
POLICIES=['allow','score_cic','risk_gated_cic','risk_only_cic']

def input_day(path, day):
    date,n,total=base.DAYS[day]
    d=pd.read_csv(path,dtype={'src':str,'dst':str}).sort_values(['w','src','dst']).reset_index(drop=True)
    start=int(pd.Timestamp(date,tz='UTC').timestamp())//10
    required=base.BASE+['attack_flows']
    if not set(required).issubset(d):raise ValueError('Missing input columns')
    if len(d)!=n or d.flows.sum()!=total or not d.w.between(start,start+8639).all() or d.duplicated(base.KEY).any() or d[base.KEY].isna().any().any():raise ValueError('Unexpected input day')
    numeric=['w','flows','attack_flows']+base.NUM+['distinct_dst_ports']
    if not np.isfinite(d[numeric]).all().all() or not (d.flows>0).all() or not d.attack_flows.between(0,d.flows).all() or not (d[['w','flows','attack_flows']]%1==0).all().all() or not (d[base.NUM+['distinct_dst_ports']]>=0).all().all():raise ValueError('Invalid counts or features')
    expected={'C':(2686781,2140675),'D':(3396651,354324)}[day]
    if (int(d.attack_flows.sum()),int((d.flows-d.attack_flows).sum()))!=expected:raise ValueError('Unexpected totals')
    return d

def targets(d):
    # Only training data. Match next exact clock bin and directed pair; absent = zero.
    future=d[base.KEY+['flows','attack_flows']].copy()
    future['w']-=1;future['next_a']=future.attack_flows;future['next_b']=future.flows-future.attack_flows
    y=d[base.KEY].merge(future[base.KEY+['next_a','next_b']],on=base.KEY,how='left',validate='one_to_one',sort=False)
    assert y[base.KEY].equals(d[base.KEY].reset_index(drop=True))
    return y[['next_a','next_b']].fillna(0).to_numpy(float)

def risk_features(obs,p,cols):
    z=obs[cols].to_numpy(float)
    if not np.isfinite(z).all() or (z<0).any():raise ValueError('Bad risk features')
    return np.column_stack([np.log1p(z), np.asarray(p,float)])

def fit_risk(d, scorer, cols, seed):
    hist=base.History(); blocks=[]
    for w,raw in d.groupby('w',sort=True):
        obs=hist.build(raw[base.BASE],int(w))
        blocks.append(risk_features(obs,scorer(obs),cols))
    X=np.concatenate(blocks); y=targets(d)
    # Exclude final clock (right censoring). No use of D labels in fitting.
    keep=d.w.to_numpy()<d.w.max()
    models=[]
    for j in range(2):
        m=HistGradientBoostingRegressor(loss='poisson',max_iter=100,max_leaf_nodes=31,learning_rate=.08,l2_regularization=1.,min_samples_leaf=30,early_stopping=False,random_state=seed)
        with threadpool_limits(limits=4):m.fit(X[keep],y[keep,j])
        models.append(m)
    return models, int(keep.sum())

def available_actions(credit):
    return max(0, math.floor(credit+1e-12))

def choose(obs,p,theta,credit,policy,pred_a=None,pred_b=None):
    cap=available_actions(credit)
    if cap<=0 or policy=='allow':return []
    if policy=='score_cic':
        candidates=np.flatnonzero(p>=theta);rank=p
    else:
        if pred_a is None or pred_b is None:raise ValueError('Missing forecasts')
        if not np.isfinite(pred_a).all() or not np.isfinite(pred_b).all() or (pred_a<0).any() or (pred_b<0).any():raise ValueError('Bad forecasts')
        # Predefined objective in counts, not a calibrated probability nor a safety guarantee.
        rank=pred_a-PENALTY*pred_b
        mask=rank>0
        if policy=='risk_gated_cic':mask &= p>=theta
        elif policy!='risk_only_cic':raise ValueError(policy)
        candidates=np.flatnonzero(mask)
    return sorted(candidates,key=lambda i:(-rank[i],-p[i],str(obs.src.iloc[i]),str(obs.dst.iloc[i])))[:cap]

def simulate(d, scorer, theta, policy, cols, risk=None, reference=None):
    hist=base.History();scheduled=set();prev=None;credit=0.; a=b=actions=blocked_windows=admitted_windows=changes=0
    ref={};digest=hashlib.sha256()
    for w,raw in d.groupby('w',sort=True):
        w=int(w);pairs=list(zip(raw.src,raw.dst));scheduled=scheduled if prev==w-1 else set()
        blocked=np.array([k in scheduled for k in pairs],bool)
        # Evaluation counters never feed the policy.
        a+=int(raw.attack_flows.to_numpy()[blocked].sum());b+=int((raw.flows-raw.attack_flows).to_numpy()[blocked].sum());blocked_windows+=int(blocked.sum())
        obs=hist.build(raw.loc[~blocked,base.BASE],w);admitted_windows+=len(obs);scheduled=set();prev=w
        if obs.empty:continue
        p=scorer(obs)
        if policy=='allow':
            ref.update({(w,s,t):float(v) for s,t,v in zip(obs.src,obs.dst,p)});continue
        if reference is not None:
            q=np.array([reference[(w,s,t)] for s,t in zip(obs.src,obs.dst)]);changes+=int((np.abs(q-p)>1e-6).sum())
        credit=min(CAP,credit+RATE*len(obs));pa=pb=None
        if policy.startswith('risk_') and available_actions(credit)>0:
            X=risk_features(obs,p,cols)
            pa,pb=(np.maximum(0,m.predict(X)) for m in risk)
        selected=choose(obs,p,theta,credit,policy,pa,pb)
        credit-=len(selected);scheduled={(obs.src.iloc[i],obs.dst.iloc[i]) for i in selected};actions+=len(scheduled)
        for pair in sorted(scheduled):digest.update(json.dumps([w+1,*pair],separators=(',',':')).encode()+b'\n')
    A=int(d.attack_flows.sum());B=int((d.flows-d.attack_flows).sum())
    assert 0<=a<=A and 0<=b<=B and admitted_windows+blocked_windows==len(d)
    return dict(total_attack=A,total_benign=B,blocked_attack=a,blocked_benign=b,attack_suppression=a/A,benign_disruption=b/B,budget_pass=b/B<=.01,actions=actions,blocked_pair_windows=blocked_windows,admitted_pair_windows=admitted_windows,scores_changed_vs_allow=changes,action_schedule_sha256=digest.hexdigest()),ref

def selftest():
    base.selftest()
    d=pd.DataFrame({'src':['a','b','a','a'],'dst':['x','x','x','x'],'w':[10,10,11,13],'flows':[2,3,5,1],'attack_flows':[1,2,4,1]})
    assert targets(d).tolist()==[[4.,1.],[0.,0.],[0.,0.],[0.,0.]]
    obs=d.iloc[:2];p=np.array([.99,.90]);pa=np.array([1.,20.]);pb=np.array([1.,.1])
    assert choose(obs,p,.8,1,'score_cic')==[0]
    assert choose(obs,p,.8,1,'risk_gated_cic',pa,pb)==[1]
    assert choose(obs,p,.95,1,'risk_gated_cic',pa,pb)==[]
    assert choose(obs,p,.95,1,'risk_only_cic',pa,pb)==[1]
    assert choose(obs,p,.8,.5,'risk_only_cic')==[]
    # Regression: floating-point credit can lie just below the integer boundary.
    near_one=np.nextafter(1.,0.)
    assert near_one<1 and available_actions(near_one)==1
    assert choose(obs,p,.8,near_one,'risk_gated_cic',pa,pb)==[1]
    assert available_actions(1.-1e-10)==0
    # Synthetic closed-loop parity and label permutation for risk selection.
    rows=[]
    for w in range(100,165):
        for s in ['a','b']:
            rows.append(dict(src=s,dst='x',w=w,flows=2,attack_flows=int(s=='a'),**{k:2 for k in base.NUM},distinct_dst_ports=1))
    raw=pd.DataFrame(rows)
    class Constant:
        def __init__(self,c):self.c=c
        def predict(self,X):return np.full(len(X),self.c)
    # Validate label-free features and replay decisions under a label permutation.
    h=base.History();obs=h.build(raw[raw.w==100][base.BASE],100)
    assert 'attack_flows' not in obs
    X=risk_features(obs,np.array([.8,.9]),['flows','pair_flows_prev60']);assert X.shape==(2,3)
    sc=lambda x:np.full(len(x),.9)
    risk=[Constant(20.),Constant(.1)]; cols=['flows','pair_flows_prev60']
    m,_=simulate(raw,sc,.8,'risk_gated_cic',cols,risk)
    swapped=raw.copy();swapped['attack_flows']=swapped.flows-swapped.attack_flows
    n,_=simulate(swapped,sc,.8,'risk_gated_cic',cols,risk)
    assert m['actions']>0 and m['blocked_pair_windows']>0 and m['action_schedule_sha256']==n['action_schedule_sha256'] and m['blocked_attack']==n['blocked_benign']
    # Integration regression: 100 windows with one candidate accumulate 0.01 each.
    # Explicitly exercise a sub-integer value accepted by the shared tolerance.
    credit=0.
    for _ in range(100):credit+=RATE
    assert available_actions(credit)==1
    print('EXP11.2 SELFTEST PASS: exact next-window targets, risk ranking, gate, credit, label-free input and risk action invariance')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');p.add_argument('--root',type=Path,default=Path.home()/'Downloads');p.add_argument('--train',type=Path);p.add_argument('--test',type=Path);p.add_argument('--out',type=Path);p.add_argument('--seeds',type=int,nargs='+',choices=base.SEEDS,default=base.SEEDS);args=p.parse_args()
    torch.set_num_threads(1);selftest()
    if args.self_test:return
    if not all([args.train,args.test,args.out]):p.error('--train --test --out required')
    f,g,cal,th=base.load(args.root);c=input_day(args.train,'C');cols=g['local']+g['temporal']+g['relational']
    args.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(stage='exploratory',train='C',test='D',train_sha256=base.sha(args.train),test_sha256=base.sha(args.test),script_sha256=base.sha(__file__),base_script_sha256=base.sha(base.__file__),frozen_models_sha256=base.MODEL_HASH,frozen_policy_sha256=base.POLICY_HASH,models=base.MODELS,seeds=args.seeds,policies=POLICIES,primary='risk_gated_cic vs score_cic',secondary='risk_only_cic',penalty=PENALTY,credit_rate=RATE,credit_cap=CAP,block_seconds=10,features=cols+['detector_score'],target='next exact clock window, same pair',python=platform.python_version(),sklearn=sklearn.__version__,torch=torch.__version__,risk_model={'loss':'poisson','max_iter':100,'max_leaf_nodes':31,'learning_rate':.08,'l2_regularization':1.,'min_samples_leaf':30,'early_stopping':False})
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    # D is already explored; no confirmatory claim. All risk fitting is exclusively on C.
    d=input_day(args.test,'D')
    for seed in args.seeds:
        for name in base.MODELS:
            scorer=base.Scorer(f,g,seed,name);theta=float(th.loc[(seed,name,10),'oracle_threshold']);risk,n=fit_risk(c,scorer,cols,seed);ref=None
            artifact=args.out/f'risk_{seed}_{name}.joblib'
            joblib.dump({'models':risk,'features':cols+['detector_score'],'train_sha256':protocol['train_sha256'],'seed':seed},artifact,compress=3)
            (args.out/f'risk_{seed}_{name}.sha256').write_text(base.sha(artifact)+'  '+artifact.name+'\n')
            print(f'RISK FIT C seed={seed} model={name} rows={n} sha256={base.sha(artifact)}',flush=True)
            for pol in POLICIES:
                t=time.monotonic()
                with threadpool_limits(limits=1):result,trace=simulate(d,scorer,theta,pol,cols,risk,ref)
                if pol=='allow':ref=trace
                result.update(day='2019-04-27',seed=seed,model=name,policy=pol,threshold=theta if pol in ['score_cic','risk_gated_cic'] else None,wall_seconds=time.monotonic()-t,training_rows=n);rows.append(result);pd.DataFrame(rows).to_csv(args.out/'metrics.csv',index=False);print(json.dumps(result),flush=True)
    df=pd.DataFrame(rows);summary=df.groupby(['model','policy']).agg(attack_mean=('attack_suppression','mean'),benign_mean=('benign_disruption','mean'),budget_passes=('budget_pass','sum'),n=('seed','size'),mean_actions=('actions','mean'),mean_changed_scores=('scores_changed_vs_allow','mean')).reset_index();summary.to_csv(args.out/'summary.csv',index=False);print(summary.to_string(index=False))
if __name__=='__main__':main()
