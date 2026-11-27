#!/usr/bin/env python3
"""Read-only post-hoc diagnostic of fixed EGAP. No fitting, actions or tuning."""
import argparse, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, torch
from sklearn.metrics import average_precision_score, mean_absolute_error
from threadpoolctl import threadpool_limits
import cbid_exp11_closed_loop as base
import cbid_exp11_2_risk as risk
import cbid_exp11_3_grid as grid
import cbid_exp11_4_transfer as transfer

PATHS={
 'D':'CBID_APRE_V1/05_confirmation/inference/features/pair_windows_2019-04-27.csv',
 'E':'CBID_FINAL_AUDIT/10_final_confirmation/E/inference/features/pair_windows_2019-04-28.csv',
 'F':'CBID_FINAL_AUDIT/10_final_confirmation/F/inference/features/pair_windows_2019-04-29.csv'}

def next_targets(d):
    # Evaluation only. One current directed pair maps to at most one exact next bin.
    f=d[base.KEY+['flows','attack_flows']].copy(); f['w']-=1
    f['next_a']=f.attack_flows; f['next_b']=f.flows-f.attack_flows; f['exists']=1
    y=d[base.KEY].merge(f[base.KEY+['next_a','next_b','exists']],on=base.KEY,how='left',validate='one_to_one',sort=False)
    assert y[base.KEY].equals(d[base.KEY].reset_index(drop=True))
    return y[['next_a','next_b','exists']].fillna(0).to_numpy(float)

def div(a,b):return float(a/b) if b else np.nan

def summarize_stages(d,p,pa,pb,theta,penalty):
    # Ground truth enters only this diagnostic, after scores and forecasts exist.
    y=next_targets(d); a,b,exists=y.T; A=int(d.attack_flows.sum()); B=int((d.flows-d.attack_flows).sum())
    valid=d.w.to_numpy()<d.w.max() # exclude last observed bin (right boundary)
    gate=p>=theta; util=pa-penalty*pb>0; rows=[]
    for stage,mask in [('all_current',np.ones(len(d),bool)),('score_gate',gate),('positive_utility',util),('joint_gate',gate&util)]:
        m=valid&mask; n=int(m.sum()); ca=int(a[m].sum()); cb=int(b[m].sum())
        rows.append(dict(stage=stage,rows=n,fraction_of_valid=div(n,valid.sum()),next_pair_exists=int(exists[m].sum()),next_pair_exists_fraction=div(exists[m].sum(),n),next_attack_positive=int((a[m]>0).sum()),next_attack_flows=ca,next_benign_flows=cb,attack_share_of_day=div(ca,A),benign_share_of_day=div(cb,B)))
    # Forecast calibration in counts on the unmodified trajectory, not on blocked state.
    z=d.attack_flows.to_numpy()>0; info=dict(n=len(d),valid_next_rows=int(valid.sum()),theta=theta,penalty=penalty,AP=float(average_precision_score(z,p)),score_gate_rows=int(gate.sum()),score_gate_positive=int((gate&z).sum()),score_gate_negative=int((gate&~z).sum()),gate_recall=div((gate&z).sum(),z.sum()),gate_fpr=div((gate&~z).sum(),(~z).sum()),utility_positive_rows=int(util.sum()),joint_gate_rows=int((gate&util).sum()))
    for name,pred,actual in [('attack',pa,a),('benign',pb,b)]:
        info.update({name+'_forecast_sum':float(pred[valid].sum()),name+'_observed_sum':float(actual[valid].sum()),name+'_forecast_mae':float(mean_absolute_error(actual[valid],pred[valid])),name+'_forecast_mean_ratio':div(pred[valid].sum(),actual[valid].sum())})
    qs=[]
    for label,mask in [('all',np.ones(len(d),bool)),('positive',z),('negative',~z)]:
        for v in [0,.1,.5,.9,.95,.99,.999,1]:qs.append(dict(group=label,quantile=v,score=float(np.quantile(p[mask],v)) if mask.any() else np.nan))
    return info,rows,qs

def selftest():
    d=pd.DataFrame({'src':['a','b','a','b','a'],'dst':['x']*5,'w':[10,10,11,11,13],'flows':[2,3,5,4,1],'attack_flows':[1,0,4,0,1]})
    assert next_targets(d).tolist()==[[4.,1.,1.],[0.,4.,1.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
    p=np.array([.99,.9,.2,.1,.99]); pa=np.array([5,0,0,0,9]); pb=np.array([0,3,0,0,0]); info,s,_=summarize_stages(d,p,pa,pb,.95,5)
    assert info['joint_gate_rows']==2 and s[-1]['rows']==1 and s[-1]['next_attack_flows']==4 and s[0]['next_benign_flows']==5
    assert s[0]['attack_share_of_day']==4/6
    print('EXP12.1 SELFTEST PASS: exact next-bin pairing, final-bin exclusion, gate accounting',flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path.home()/'Downloads');ap.add_argument('--out',type=Path);ap.add_argument('--days',nargs='+',choices=PATHS,default=list(PATHS));ap.add_argument('--seeds',nargs='+',type=int,choices=base.SEEDS,default=base.SEEDS);ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    torch.set_num_threads(1);selftest()
    freeze=Path(transfer.__file__).with_name('exp11_4_freeze.json');base.check(freeze,transfer.FREEZE_HASH);cfg=json.loads(freeze.read_text())
    for mod in (base,risk,grid):base.check(mod.__file__,cfg['source_hashes'][Path(mod.__file__).name])
    if a.self_test:return
    if a.out is None:ap.error('--out required')
    if len(set(a.days))!=len(a.days) or len(set(a.seeds))!=len(a.seeds):ap.error('Duplicate day or seed')
    if sklearn.__version__!='1.7.2':raise RuntimeError('Use original sklearn 1.7.2 environment')
    f,g,cal,th=base.load(a.root);cols=g['local']+g['temporal']+g['relational']; arts={}; hashes={}
    for seed in a.seeds:
        for name in base.MODELS:
            est,h,_=grid.risk_artifact(a.root/'CBID_EXP11_2',seed,name,cols)
            if h!=cfg['risk_artifacts'][f'{seed}_{name}']:raise ValueError('Risk hash mismatch')
            arts[seed,name]=est;hashes[f'{seed}_{name}']=h
    inputs={day:a.root/PATHS[day] for day in a.days}; ih={day:base.sha(path) for day,path in inputs.items()}
    if 'D' in ih and ih['D']!=grid.TEST_HASH:raise ValueError('D input hash mismatch')
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'protocol.json').write_text(json.dumps(dict(stage='posthoc diagnosis only; all days previously evaluated',no_action=True,no_fit=True,no_threshold_search=True,input_sha256=ih,risk_hashes=hashes,source_sha256=base.sha(__file__),freeze_sha256=transfer.FREEZE_HASH,model_freeze_sha256=base.MODEL_HASH,policy_freeze_sha256=base.POLICY_HASH,days=a.days,seeds=a.seeds,python=platform.python_version(),sklearn=sklearn.__version__,torch=torch.__version__),indent=2)+'\n')
    results=[];stages=[];quantiles=[]
    for day in a.days:
        d=risk.input_day(inputs[day],'D') if day=='D' else transfer.read_input(inputs[day],cfg['days'][day])
        # Build label-free, no-action observation table once per day, same history semantics.
        hist=base.History(); blocks=[]
        for w,raw in d.groupby('w',sort=True):blocks.append(hist.build(raw[base.BASE],int(w)))
        obs=pd.concat(blocks,ignore_index=True);assert obs[base.KEY].equals(d[base.KEY]);assert np.allclose(obs[cols],d[cols],rtol=1e-8,atol=1e-7),'Feature parity failed'
        # A detector is evaluated per clock bin, preserving the actual peer grouping.
        idx=list(obs.groupby('w',sort=True).indices.values())
        for seed in a.seeds:
            for name in base.MODELS:
                scorer=base.Scorer(f,g,seed,name);theta=float(th.loc[(seed,name,10),'oracle_threshold']);penalty=5 if name=='M0_local' else 20
                with threadpool_limits(limits=1):
                    p=np.concatenate([scorer(obs.iloc[i]) for i in idx]); X=risk.risk_features(obs,p,cols); pa,pb=(np.maximum(0,m.predict(X)) for m in arts[seed,name])
                if not np.isfinite(np.column_stack([p,pa,pb])).all():raise ValueError('Nonfinite predictions')
                info,ss,qq=summarize_stages(d,p,pa,pb,theta,penalty);key=dict(day=day,seed=seed,model=name);results.append(dict(**key,**info));stages.extend(dict(**key,**x) for x in ss);quantiles.extend(dict(**key,**x) for x in qq)
                for fn,rows in [('diagnostic_metrics.csv',results),('stage_coverage.csv',stages),('score_quantiles.csv',quantiles)]:pd.DataFrame(rows).to_csv(a.out/fn,index=False)
                print(day,seed,name,'AP',round(info['AP'],6),'score_gate',info['score_gate_rows'],'joint_gate',info['joint_gate_rows'],'reachable_attack_share',round(ss[0]['attack_share_of_day'],6),'joint_attack_share',round(ss[-1]['attack_share_of_day'],6),flush=True)
    s=pd.DataFrame(results).groupby(['day','model']).mean(numeric_only=True).drop(columns='seed');s.to_csv(a.out/'summary.csv');print(s[['AP','gate_recall','gate_fpr','score_gate_rows','joint_gate_rows','attack_forecast_mean_ratio','benign_forecast_mean_ratio']].to_string())
    t=pd.DataFrame(stages).groupby(['day','model','stage']).mean(numeric_only=True).drop(columns='seed');t.to_csv(a.out/'stage_summary.csv');print(t[['rows','next_pair_exists_fraction','attack_share_of_day','benign_share_of_day']].to_string());print('COMPLETE',len(results),'diagnostics; no tuning or interventions')
if __name__=='__main__':main()
