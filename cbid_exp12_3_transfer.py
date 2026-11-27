#!/usr/bin/env python3
"""Frozen retrospective test of the sole D-selected adaptive policy. Offline replay only."""
import argparse,json,platform,time
from pathlib import Path
import pandas as pd
import sklearn,torch
from threadpoolctl import threadpool_limits
import cbid_exp11_closed_loop as base
import cbid_exp11_2_risk as risk
import cbid_exp11_3_grid as grid
import cbid_exp11_4_transfer as old
import cbid_exp12_2_adaptive as adaptive
FREEZE_HASH='9b66feea71f8096b6101455e3f63a388a92052fe99a9e71d54e3c55cf3d90964'

def verdict(rows,cfg):
    d=pd.DataFrame(rows)
    assert set(d.seed)==set(cfg['seeds']) and len(d)==3 and not d.seed.duplicated().any()
    return dict(budget_passes=int((d.benign_disruption<=cfg['budget']).sum()),positive_suppression_runs=int((d.attack_suppression>0).sum()),all_pass=bool((d.benign_disruption<=cfg['budget']).all() and (d.attack_suppression>0).all()),attack_mean=float(d.attack_suppression.mean()),benign_max=float(d.benign_disruption.max()),development_floor_met=bool(d.attack_suppression.mean()>=cfg['development_floor']),independent_confirmation=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.home()/'Downloads');p.add_argument('--day',choices=['E','F']);p.add_argument('--out',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    here=Path(__file__).resolve().parent;freeze=here/'exp12_3_freeze.json';base.check(freeze,FREEZE_HASH);cfg=json.loads(freeze.read_text())
    for module in (base,risk,grid,old,adaptive):base.check(module.__file__,cfg['source_hashes'][Path(module.__file__).name])
    base.check(here/'exp11_4_freeze.json',cfg['previous_freeze_sha256'])
    assert (adaptive.LOOKBACK,adaptive.MIN_HISTORY,adaptive.QUANTILE,adaptive.RATE,adaptive.CAP)==(cfg['lookback_bins'],cfg['min_history_rows'],cfg['quantile'],cfg['rate'],cfg['cap'])
    assert adaptive.PENALTIES[cfg['model']]==cfg['penalty'] and cfg['policy'] in adaptive.POLICIES
    torch.set_num_threads(1);adaptive.selftest()
    good=[dict(seed=s,benign_disruption=.009,attack_suppression=.03) for s in cfg['seeds']];assert verdict(good,cfg)['all_pass'];good[0]['attack_suppression']=0;assert not verdict(good,cfg)['all_pass'];good[0]['attack_suppression']=.03;good[1]['benign_disruption']=.011;assert not verdict(good,cfg)['all_pass']
    print('EXP12.3 FREEZE AND VERDICT CHECK PASS',FREEZE_HASH,flush=True)
    if a.self_test:return
    if a.day is None or a.out is None:p.error('--day and --out required')
    if sklearn.__version__!='1.7.2':raise RuntimeError('Use original sklearn 1.7.2 environment')
    f,g,cal,th=base.load(a.root);cols=g['local']+g['temporal']+g['relational'];name=cfg['model'];arts={}
    for seed in cfg['seeds']:
        est,h,_=grid.risk_artifact(a.root/'CBID_EXP11_2',seed,name,cols)
        if h!=cfg['risk_hashes'][str(seed)]:raise ValueError('Risk artifact mismatch')
        arts[seed]=est
    spec=cfg['days'][a.day];path=a.root/spec['path'];base.check(path,spec['sha256']);d=old.read_input(path,spec)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'protocol.json').write_text(json.dumps(dict(stage=cfg['stage'],day=a.day,freeze_sha256=FREEZE_HASH,runner_sha256=base.sha(__file__),input_sha256=base.sha(path),python=platform.python_version(),sklearn=sklearn.__version__,torch=torch.__version__),indent=2)+'\n')
    rows=[];risk.RATE=cfg['rate'];risk.CAP=cfg['cap']
    for seed in cfg['seeds']:
        scorer=base.Scorer(f,g,seed,name);est=arts[seed];theta=float(th.loc[(seed,name,10),'oracle_threshold']);t=time.monotonic()
        with threadpool_limits(limits=1): control,ref=risk.simulate(d,scorer,theta,'allow',cols,est)
        control.update(day=a.day,seed=seed,model=name,policy='allow',rate=cfg['rate'],penalty=cfg['penalty'],wall_seconds=time.monotonic()-t)
        rows.append(control);pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False)
        t=time.monotonic()
        with threadpool_limits(limits=1):r=adaptive.simulate(d,scorer,cols,est,cfg['policy'],cfg['penalty'],ref)
        r.update(day=a.day,seed=seed,model=name,policy=cfg['policy'],rate=cfg['rate'],penalty=cfg['penalty'],risk_sha256=cfg['risk_hashes'][str(seed)],wall_seconds=time.monotonic()-t);rows.append(r);pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False);print(json.dumps(r),flush=True)
    df=pd.DataFrame(rows);assert len(df)==6 and not df.duplicated(['seed','model','policy']).any()
    s=grid.summarize(rows,3);s.to_csv(a.out/'summary.csv',index=False);print(s.to_string(index=False));v=verdict(df[df.policy==cfg['policy']],cfg);v['day']=a.day;(a.out/'verdict.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
    print('COMPLETE 3 selected-policy runs and 3 no-action controls; retrospective only')
if __name__=='__main__':main()
