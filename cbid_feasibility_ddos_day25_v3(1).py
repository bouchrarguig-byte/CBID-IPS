#!/usr/bin/env python3
"""CBID-IPS #31617: predeclared within-day DDoS feasibility on 25 Apr 2019.

C=26-Apr-2019 remains sealed. A=00:00-14:24 UTC on 25-Apr (first 60% of the UTC day); calibration=
14:24-19:12 UTC (next 20%); B=19:12-24:00 UTC (last 20%), with 120s
temporal embargo at each boundary. Boundaries are fixed by the clock, not labels.
No GAT training; this is exploratory feasibility, not independent confirmation.

Requires cbid_feasibility_nf_toniot_v2_1.py alongside this script.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import cbid_feasibility_nf_toniot_v2_1 as core
except ImportError as e:
    raise SystemExit('Place cbid_feasibility_nf_toniot_v2_1.py in the same directory.') from e

DEV_DAY='2019-04-25'
SEALED_DAY='2019-04-26'
TRAIN_FRACTION=0.60
CAL_FRACTION=0.20


def build_args():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--archive',default=str(core.DEFAULT_ARCHIVE),help='Original NF-ToN-IoT-v3 ZIP')
    p.add_argument('--out',default=str(Path.home()/'Downloads'/'CBID_31617_DDoS_25Apr_feasibility'))
    p.add_argument('--profile-only',action='store_true',help='No models fitted; print class and coordination support per predefined split')
    p.add_argument('--chunk-size',type=int,default=150_000)
    p.add_argument('--max-groups',type=int,default=2_000_000)
    p.add_argument('--bin-seconds',type=int,default=10)
    p.add_argument('--embargo-seconds',type=int,default=120)
    p.add_argument('--cal-fpr',type=float,default=.01)
    p.add_argument('--min-coord-windows',type=int,default=200)
    p.add_argument('--min-headroom',type=float,default=.10)
    p.add_argument('--verbose',action='store_true')
    args=p.parse_args()
    if args.bin_seconds<=0 or args.embargo_seconds<0 or args.chunk_size<=0:
        p.error('bin/chunk must be positive and embargo nonnegative')
    if not 0<args.cal_fpr<1 or not 0<=args.min_headroom<=1:
        p.error('cal-fpr and min-headroom out of bounds')
    # required by the already reviewed V2.1 raw aggregator
    args.csv=None
    args.diagnose_only=False
    return args


def count_period(label,obs):
    windows=obs.groupby(['dst','bin'],sort=False).agg(
        n_sources=('src','nunique'), n_positive=('y','sum'))
    co=windows.n_positive>=2
    return {
        'period':label,
        'n_10s_bins':int(obs['bin'].nunique()),
        'n_source_windows':int(len(obs)),
        'positive_DDoS_source_windows':int(obs.y.sum()),
        'benign_source_windows':int((obs.y==0).sum()),
        'n_destination_windows':int(len(windows)),
        'coordinated_destination_windows_ge2':int(co.sum()),
        'coordinated_destination_windows_ge5':int((windows.n_positive>=5).sum()),
        'positive_in_coordinated_windows':int(((obs.y==1)&(obs.coordinated==1)).sum()),
    }


def main():
    args=build_args(); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    print('CBID-IPS DDoS 25-Apr-2019 — WITHIN-DAY DEVELOPMENT FEASIBILITY')
    print('PREDECLARED: UTC clock 00:00-14:24 train / 14:24-19:12 calibration / 19:12-24:00 B; embargo 120 s by default.')
    print('SEALED FINAL DAY: 26-Apr-2019; its labels and model predictions are NEVER read.')
    print('WARNING: the DDoS task was chosen after examining 25-Apr labels; this is exploratory design.')
    print('Input:',args.archive,flush=True)
    obs,context=core.aggregate_selected(args,{DEV_DAY},'ddos')
    # Fixed UTC clock boundaries independent of observed data and attack labels.
    midnight_ms=int(pd.Timestamp(DEV_DAY+'T00:00:00Z').timestamp()*1000)
    midnight_bin=midnight_ms//(args.bin_seconds*1000)
    bins_per_day=86400//args.bin_seconds
    b1=int(midnight_bin+round(bins_per_day*TRAIN_FRACTION))
    b2=int(midnight_bin+round(bins_per_day*(TRAIN_FRACTION+CAL_FRACTION)))
    emb=int(np.ceil(args.embargo_seconds/args.bin_seconds))
    train=obs.loc[obs.bin<b1-emb].copy()
    cal=obs.loc[(obs.bin>=b1+emb)&(obs.bin<b2-emb)].copy()
    ev=obs.loc[obs.bin>=b2+emb].copy()
    def fmt(bin_num):
        return datetime.fromtimestamp(int(bin_num)*args.bin_seconds,timezone.utc).isoformat()
    print('\nTIME BOUNDARIES (chosen without labels):')
    print(' fixed train/cal boundary:',fmt(b1),'fixed cal/B boundary:',fmt(b2))
    print(' embargo (seconds):',args.embargo_seconds)
    stats=pd.DataFrame([count_period('A_train',train),count_period('A_calibration',cal),count_period('B_feasibility',ev)])
    print('\nPREDECLARED SPLIT SUPPORT:')
    print(stats.to_string(index=False))
    stats.to_csv(out/'support_by_temporal_split.csv',index=False)
    protocol={
        'study':'exploratory DDoS development feasibility; NOT independent test',
        'dataset':'NF-ToN-IoT-v3','development_day':DEV_DAY,'sealed_test_day':SEALED_DAY,
        'attack':'ddos','time_bin_seconds':args.bin_seconds,
        'split_rule':'fixed 60% / 20% / 20% of UTC clock day independent of labels and observed volumes',
        'train_cal_boundary_utc':fmt(b1),'cal_B_boundary_utc':fmt(b2),
        'embargo_seconds':args.embargo_seconds,
        'threshold_FPR_calibration':args.cal_fpr,
        'min_coordinated_B_windows':args.min_coord_windows,
        'min_headroom':args.min_headroom,
        'selected_attack_after_reviewing_development_day_labels':True,
        'historical_scanning_V8_models_not_evaluable_as_DDoS_detectors':True,
        'test_day_opened':False,
    }
    (out/'protocol_details.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if args.profile_only:
        print('\nPROFILE COMPLETE: models not trained; no C labels opened.')
        return
    if any(len(z)==0 or z.y.nunique()!=2 for z in (train,cal,ev)):
        print('\nNO-GO: at least one fixed chronological split lacks both DDoS and benign.')
        print('Do not move boundaries based on observed labels to manufacture a feasible split.')
        return
    yt=ev.y.to_numpy();co=ev.coordinated.to_numpy();results=[];preds={}
    models=[('rule_n_sources',None,['ctx_n_sources']),
            ('model_local',core.make_model()[0],core.LOCAL),
            ('model_local_context',core.make_model()[0],core.LOCAL+context)]
    print('\nBASELINES ON B: threshold fixed on calibration portion of A')
    for name,model,feats in models:
        if model is None:
            s_cal=cal.ctx_n_sources.to_numpy();s_b=ev.ctx_n_sources.to_numpy()
        else:
            model.fit(train[feats],train.y)
            s_cal=model.predict_proba(cal[feats])[:,1]
            s_b=model.predict_proba(ev[feats])[:,1]
        threshold=core.threshold_fpr(cal.y.to_numpy(),s_cal,args.cal_fpr)
        row,pred=core.evaluate(yt,s_b,threshold,co)
        row.update(model=name,threshold_from_calibration=threshold)
        results.append(row);preds[name]=pred
    frame=pd.DataFrame(results)
    print(frame.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    frame.to_csv(out/'baselines_B_day25.csv',index=False)
    cp=(yt==1)&(co==1)
    ncp=int(cp.sum())
    ctxmiss=int((cp&~preds['model_local_context']).sum())
    allmiss=int((cp&~preds['model_local_context']&~preds['model_local']&~preds['rule_n_sources']).sum())
    headroom=ctxmiss/ncp if ncp else 0.0
    ncoord=int(stats.loc[stats.period=='B_feasibility','coordinated_destination_windows_ge2'].iloc[0])
    verdict='GO_FOR_RESEARCH_ONLY' if ncoord>=args.min_coord_windows and headroom>=args.min_headroom else 'NO_GO'
    print('\nFEASIBILITY:',verdict)
    print(f'B coordinated dest-windows: {ncoord} (required >= {args.min_coord_windows})')
    print(f'B coordinated positives: {ncp}; missed by context model: {ctxmiss} ({headroom:.1%}); missed by ALL baselines: {allmiss}')
    print('Headroom is NOT a graph-model performance estimate.')
    print('The final 26-Apr day is STILL SEALED; no final evaluation performed.')
    (out/'verdict.json').write_text(json.dumps({'verdict':verdict,
            'coordinated_windows_B':ncoord,'coordinated_positives_B':ncp,
            'missed_by_context_baseline_B':ctxmiss,'missed_by_all_baselines_B':allmiss,
            'headroom_B':headroom},indent=2),encoding='utf-8')
    print('Results:',out)

if __name__=='__main__': main()
