#!/usr/bin/env python3
"""Freeze replay policies on already-open C; no access to D."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from cbid_apre_v1_03_relation_gate import MODELS, KEY, audit

def sha(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists(): raise SystemExit('Output exists; no overwrite')
    src=a.input/'predictions.csv';d=pd.read_csv(src,usecols=KEY+['model','score','flows','attack_flows','y'])
    assert not d.duplicated(KEY+['model']).any() and set(d.seed)=={42,43,44} and set(d.model)==set(MODELS)
    start=int(pd.Timestamp('2019-04-26',tz='UTC').timestamp())//10
    assert d.w.between(start,start+8639).all()
    r=audit(d);r=r[(r.budget==.01)&(r.extra_delay_s==0)&r.horizon_s.isin([10,60])].copy()
    assert len(r)==72
    assert (r.benign_exposure_rate<=.01+1e-12).all()
    a.out.mkdir(parents=True);r.to_csv(a.out/'thresholds.csv',index=False)
    manifest={'calibration_day':'2019-04-26','candidate_test_day':'2019-04-27','test_day_accessed':False,'primary_model':'G1_real_peer','primary_horizon_s':10,'secondary_horizon_s':60,'budget':.01,'test_delays_s':[0,10],'seeds':[42,43,44],'models':MODELS,'prediction_sha256':sha(src),'thresholds_sha256':sha(a.out/'thresholds.csv'),'script_sha256':sha(__file__),'audit_script_sha256':sha(Path(__file__).with_name('cbid_apre_v1_03_relation_gate.py')),'original_model_freeze_sha256':'92198e78d35047750afc34edf96c9f482a81a746c302b7170fc8a857cbc2952e','rule':'threshold per model/seed/horizon; never recalibrate on D; infinity means no action; delays use same threshold','limitations':'C already inspected; open-loop exposure of completed flows, not actual prevention'}
    (a.out/'policy_freeze.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    print(r[['seed','model','horizon_s','oracle_threshold','attack_exposure_rate','benign_exposure_rate']].to_string(index=False));print('POLICY FREEZE SHA256:',sha(a.out/'policy_freeze.json'))
if __name__=='__main__': main()
