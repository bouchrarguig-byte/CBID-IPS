#!/usr/bin/env python3
"""CBID-IPS IJ-AI #31617 — DDoS development feasibility V3.1 (25 Apr 2019 ONLY).

Post-hoc REVISED exploratory clock protocol, following V3 NO-GO (no DDoS before 14:24).
TRAIN = 25-Apr 14:24–17:36 UTC; CAL = 17:36–19:12; B = 19:12–24:00.
At each boundary, 120 seconds of embargo on BOTH sides. No 26-Apr records read.
This clock cut was chosen AFTER viewing coarse V3 class supports; it is not preregistered
or independent confirmation. B is feasibility; do not tune after seeing results.

Dependencies: cbid_feasibility_nf_toniot_v2_1.py in the same directory.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import cbid_feasibility_nf_toniot_v2_1 as core

DAY = '2019-04-25'
C_SEALED = '2019-04-26'
CLOCK_TRAIN_START = '2019-04-25T14:24:00Z'
CLOCK_TRAIN_CAL = '2019-04-25T17:36:00Z'
CLOCK_CAL_B = '2019-04-25T19:12:00Z'


def arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--archive',default=str(core.DEFAULT_ARCHIVE))
    p.add_argument('--out',default=str(Path.home()/'Downloads'/'CBID_31617_DDoS_25Apr_feasibility_v3_1'))
    p.add_argument('--profile-only', action='store_true', help='Support and attack-source audit only; no baselines trained')
    p.add_argument('--chunk-size',type=int,default=150_000)
    p.add_argument('--max-groups',type=int,default=2_000_000)
    p.add_argument('--bin-seconds',type=int,default=10)
    p.add_argument('--embargo-seconds',type=int,default=120)
    p.add_argument('--cal-fpr',type=float,default=.01)
    p.add_argument('--min-coord-windows',type=int,default=200)
    p.add_argument('--min-headroom',type=float,default=.10)
    p.add_argument('--verbose',action='store_true')
    args = p.parse_args()
    if args.bin_seconds != 10:
        p.error('V3.1 protocol is fixed at --bin-seconds 10. Reconfiguration is NOT allowed.')
    if args.embargo_seconds != 120:
        p.error('V3.1 protocol is fixed at --embargo-seconds 120. Reconfiguration is NOT allowed.')
    if args.cal_fpr != .01 or args.min_coord_windows != 200 or args.min_headroom != .10:
        p.error('V3.1 1% FPR, >=200 windows, >=10% missed positives are FIXED.')
    if args.chunk_size <= 0 or args.max_groups <= 0:
        p.error('Invalid read parameters')
    args.csv = None
    args.diagnose_only = False
    return args


def count(label, frame):
    group = frame.groupby(['dst','bin'],sort=False).agg(n_src=('src','nunique'),n_pos=('y','sum'))
    pos = frame.loc[frame.y==1]
    co = group.n_pos.ge(2)
    # Source sets describe repeated attack campaigns. Counts are not independent attacks.
    co_keys = group.loc[co].index
    pos_in_co = pos.set_index(['dst','bin']).index.isin(co_keys)
    source_sets = pos.loc[pos_in_co].groupby(['dst','bin'])['src'].agg(lambda x: tuple(sorted(set(x))))
    set_freq = source_sets.value_counts()
    return {
        'period':label, 'n_bins':int(frame.bin.nunique()), 'n_source_windows':int(len(frame)),
        'positive_DDoS':int(pos.shape[0]), 'benign':int((frame.y==0).sum()),
        'n_destination_windows':int(len(group)),
        'coordinated_dest_windows_ge2':int(co.sum()),
        'coordinated_dest_windows_ge5':int(group.n_pos.ge(5).sum()),
        'positive_in_coordinated':int(pos_in_co.sum()),
        'unique_DDoS_source_IPs':int(pos.src.nunique()),
        'unique_DDoS_destination_IPs':int(pos.dst.nunique()),
        'unique_attacking_source_sets_in_coord':int(len(set_freq)),
        'max_windows_sharing_same_attacking_source_set':int(set_freq.max()) if len(set_freq) else 0,
        'share_coord_windows_largest_source_set':float(set_freq.max()/len(source_sets)) if len(source_sets) else 0.0,
    }


def main():
    args=arg_parser();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    print('CBID-IPS #31617 — DDoS V3.1 REVISED EXPLORATORY feasibility')
    print('V3 NO-GO reason: 0 positive DDoS observations in 00:00–14:24 TRAIN.')
    print('REVISED POST-HOC CLOCK: train 14:24–17:36 / calibration 17:36–19:12 / B 19:12–24:00 UTC (25 Apr).')
    print('EMBARGO: 120 s before and after train/cal and cal/B boundaries.')
    print('SEALED TEST C = 26-Apr-2019. This script reads ONLY 25-Apr labels/data.')
    print('B is NOT a blind test, since coarse B labels were already inspected in V3.')
    obs, context = core.aggregate_selected(args,{DAY},'ddos')
    b0=int(pd.Timestamp(CLOCK_TRAIN_START).timestamp()//10)
    b1=int(pd.Timestamp(CLOCK_TRAIN_CAL).timestamp()//10)
    b2=int(pd.Timestamp(CLOCK_CAL_B).timestamp()//10)
    e=int(np.ceil(args.embargo_seconds/10))
    train=obs.loc[(obs.bin>=b0)&(obs.bin < b1-e)].copy()
    cal=obs.loc[(obs.bin>=b1+e)&(obs.bin < b2-e)].copy()
    B=obs.loc[obs.bin>=b2+e].copy()
    audit=pd.DataFrame([count('A_train',train),count('A_calibration',cal),count('B_feasibility',B)])
    print('\nEXPLORATORY FIXED-CLOCK SPLIT SUPPORT AND SOURCE-SET REPLICATION:')
    print(audit.to_string(index=False))
    audit.to_csv(out/'split_support_and_sources.csv', index=False)
    metad={
        'study':'Exploratory revised protocol after seeing V3 support; NOT preregistered',
        'code_version':'V3.1','development_day':DAY,'sealed_day_C':C_SEALED,
        'attack':'ddos','train_start_utc':CLOCK_TRAIN_START,
        'train_cal_boundary_utc':CLOCK_TRAIN_CAL,'cal_B_boundary_utc':CLOCK_CAL_B,
        'embargo_seconds':120,'bin_seconds':10,'calibration_FPR_target':.01,
        'GO_min_coord_windows':200,'GO_min_headroom':.10,
        'reason_for_revision':'V3 first 60% of 25-Apr had zero DDoS; its calibration window contained both classes',
        'B_used_for_model_selection':False,'B_coarse_label_support_previously_examined':True,
        'C_data_or_labels_read':False,
    }
    (out/'study_log.json').write_text(json.dumps(metad,indent=2),encoding='utf-8')
    if args.profile_only:
        print('\nPROFILE COMPLETE. No models trained. C unchanged/unread.')
        return
    if any(len(v)==0 or v.y.nunique()<2 for v in (train,cal,B)):
        print('\nNO-GO: The FIXED revised splits still lack both DDoS and benign classes.')
        print('Stop modifying temporal boundaries. Return to manuscript revision.')
        return
    results=[]; predictions={}; yt=B.y.to_numpy(); coordinated=B.coordinated.to_numpy()
    models=[('rule_n_sources',None,['ctx_n_sources']),
            ('XGB_local',core.make_model()[0],core.LOCAL),
            ('XGB_local_context',core.make_model()[0],core.LOCAL+context)]
    for name, model, features in models:
        if model is None:
            score_cal=cal.ctx_n_sources.to_numpy()
            score_B=B.ctx_n_sources.to_numpy()
        else:
            model.fit(train[features],train.y)
            score_cal=model.predict_proba(cal[features])[:,1]
            score_B=model.predict_proba(B[features])[:,1]
        threshold=core.threshold_fpr(cal.y.to_numpy(),score_cal,.01)
        metrics,pred=core.evaluate(yt,score_B,threshold,coordinated)
        metrics.update(model=name,threshold_from_calibration=threshold)
        predictions[name]=pred
        results.append(metrics)
    res=pd.DataFrame(results)
    print('\nMODEL BASELINES ON B (fixed CAL 1% FPR threshold):')
    print(res.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    res.to_csv(out/'baselines_B.csv',index=False)
    coord_positive=(yt==1)&(coordinated==1)
    npos=int(coord_positive.sum())
    missed_context=int((coord_positive&~predictions['XGB_local_context']).sum())
    missed_all=int((coord_positive&~predictions['XGB_local_context']&~predictions['XGB_local']&~predictions['rule_n_sources']).sum())
    headroom=missed_context/npos if npos else 0.0
    ncoord=int(audit.loc[audit.period.eq('B_feasibility'),'coordinated_dest_windows_ge2'].iloc[0])
    verdict='GO_RESEARCH_ONLY' if ncoord>=200 and headroom>=.10 else 'NO_GO'
    print('\nFEASIBILITY:',verdict)
    print('Coordinated windows in B:',ncoord,'(>=200 required)')
    print('Coordinated positive source-windows:',npos)
    print('Missed by XGB contextual:',missed_context,f'({headroom:.1%}; >=10% required)')
    print('Missed by all 3 baselines:',missed_all)
    print('Headroom is potential improvement only, NOT evidence that graph attention helps.')
    print('Note high repetition of same attacking-source pairs if source-set audit reports it.')
    (out/'feasibility_verdict.json').write_text(json.dumps({'verdict':verdict,'B_coordinated_windows':ncoord,
        'B_coordinated_positive_source_windows':npos,'missed_by_context':missed_context,
        'missed_all':missed_all,'headroom':headroom},indent=2),encoding='utf-8')
    print('Saved',out,'; C SEALED.')


if __name__=='__main__':
    main()
