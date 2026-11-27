#!/usr/bin/env python3
"""Frozen offline pair-window replay on previously evaluated E/F. No live actions."""
import argparse, hashlib, json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import torch
from threadpoolctl import threadpool_limits
import cbid_exp11_closed_loop as base
import cbid_exp11_2_risk as risk
import cbid_exp11_3_grid as grid

FREEZE_HASH = 'c48b1feca8df7a3d5bc032f092a773df873563c6586fa7fd26cfee6be6545d0c'

def read_input(path, spec):
    d = pd.read_csv(path, dtype={'src':str,'dst':str}).sort_values(['w','src','dst']).reset_index(drop=True)
    needed = base.BASE + ['attack_flows']
    if not set(needed).issubset(d.columns): raise ValueError('Missing required columns')
    start = int(pd.Timestamp(spec['date'],tz='UTC').timestamp())//10
    nums = ['w','flows','attack_flows']+base.NUM+['distinct_dst_ports']
    if not np.isfinite(d[nums]).all().all(): raise ValueError('Nonfinite inputs')
    if d[base.KEY].isna().any().any() or d.duplicated(base.KEY).any(): raise ValueError('Invalid pair keys')
    if not d.w.between(start,start+8639).all() or not (d[['w','flows','attack_flows']]%1==0).all().all(): raise ValueError('Invalid time/counts')
    if not (d.flows>0).all() or not d.attack_flows.between(0,d.flows).all() or not (d[base.NUM+['distinct_dst_ports']]>=0).all().all(): raise ValueError('Invalid features')
    if (len(d),int(d.attack_flows.sum()),int((d.flows-d.attack_flows).sum())) != (spec['windows'],spec['attack'],spec['benign']): raise ValueError('Unexpected day inventory')
    return d

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.home()/'Downloads')
    p.add_argument('--day',choices=['E','F'])
    p.add_argument('--input',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    freeze=Path(__file__).resolve().with_name('exp11_4_freeze.json')
    base.check(freeze,FREEZE_HASH); cfg=json.loads(freeze.read_text())
    for module in (base,risk,grid):base.check(module.__file__,cfg['source_hashes'][Path(module.__file__).name])
    torch.set_num_threads(1);grid.selftest()
    print('EXP11.4 FREEZE CHECK PASS',FREEZE_HASH,flush=True)
    if a.self_test:return
    if a.day is None or a.input is None or a.out is None:p.error('--day, --input and --out required')
    if sklearn.__version__!='1.7.2':raise RuntimeError('Use original sklearn 1.7.2 environment')
    if cfg['model_freeze_sha256']!=base.MODEL_HASH or cfg['policy_freeze_sha256']!=base.POLICY_HASH:raise ValueError('Frozen artifact identity mismatch')
    f,g,cal,th=base.load(a.root);cols=g['local']+g['temporal']+g['relational']
    # Verify every saved estimator before reading the selected day's feature CSV.
    artifacts={}
    for seed in cfg['seeds']:
        for name in base.MODELS:
            est,h,_=grid.risk_artifact(a.root/'CBID_EXP11_2',seed,name,cols)
            if h!=cfg['risk_artifacts'][f'{seed}_{name}']:raise ValueError('Unexpected risk artifact')
            artifacts[seed,name]=est
    a.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(freeze_sha256=FREEZE_HASH,runner_sha256=base.sha(__file__),day=a.day,stage=cfg['stage'],input_sha256=base.sha(a.input),python=platform.python_version(),sklearn=sklearn.__version__,torch=torch.__version__)
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    d=read_input(a.input,cfg['days'][a.day]); rows=[]
    for seed in cfg['seeds']:
        for name in base.MODELS:
            scorer=base.Scorer(f,g,seed,name);theta=float(th.loc[(seed,name,10),'oracle_threshold']);est=artifacts[seed,name];ref=None
            for item in cfg['configurations']:
                if item['model']!=name:continue
                pol=item['policy'];risk.RATE=cfg['rate'];risk.CAP=cfg['cap'];risk.PENALTY=item['penalty'];t=time.monotonic()
                with threadpool_limits(limits=1):result,trace=risk.simulate(d,scorer,theta,pol,cols,est,ref)
                if pol=='allow':ref=trace
                result.update(day=cfg['days'][a.day]['date'],seed=seed,model=name,policy=pol,rate=cfg['rate'],penalty=item['penalty'],role=item['role'],threshold=theta if pol!='allow' else None,risk_sha256=cfg['risk_artifacts'][f'{seed}_{name}'],wall_seconds=time.monotonic()-t)
                rows.append(result);pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False);print(json.dumps(result),flush=True)
    df=pd.DataFrame(rows);assert len(df)==18
    assert not df.duplicated(['seed','model','policy']).any()
    s=grid.summarize(rows,3);s.to_csv(a.out/'summary.csv',index=False);print(s.to_string(index=False))
    primary=df[df.role=='primary selected policy'];assert len(primary)==3
    verdict={'day':a.day,'primary_budget_passes':int(primary.budget_pass.sum()),'primary_positive_suppression_runs':int((primary.attack_suppression>0).sum()),'primary_all_pass':bool(primary.budget_pass.all() and (primary.attack_suppression>0).all()),'independent_confirmation':False}
    (a.out/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n');print(json.dumps(verdict),flush=True)
    print('COMPLETE 18 configurations; retrospective transfer only',flush=True)
if __name__=='__main__':main()
