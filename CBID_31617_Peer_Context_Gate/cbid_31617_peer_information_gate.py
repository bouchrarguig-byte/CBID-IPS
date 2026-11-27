#!/usr/bin/env python3
"""CBID-IPS IJ-AI #31617 — INFORMATION GATE for inter-host behavioral context.

This is a DEVELOPMENT-ONLY controlled experiment, NOT proof of graph novelty.
Input: V4 source-IP/destination-IP/10-second aggregated CSV, processed with
V6's exact 23-Apr 2019 train/validation boundaries and 120-s embargo.
The subset of 11 destinations was selected after inspecting labels in earlier
work. 24-Apr and later rows are excluded by timestamp before label access for fitting/scoring (CSV parser still reads columns).

At the end of a time bin, classify each (source IP,destination IP,bin) record.
Compare:
  local:       XGBoost on the target's own nine features
  local_degree: target features + number of other sources targeting same dst/bin
  real_peer:   target features + degree + other sources' mean, target-peer
               absolute discrepancy, and cosine coherence for SAME dst/bin
  random_peer: identical feature dimensions, but with matched-degree random
               other-source records from the SAME bin (possibly another dst)

The target is excluded from BOTH real and random neighbor context. No class
labels are used in topology/features. Degree-matched random control uses
random sampling without replacement among other source hosts in the same bin.
No hyperparameter / threshold tuning on validation or any later date.

If there is no evidence that real_peer consistently beats local_degree AND
random_peer, STOP: a GATv2 graph model is unlikely to have a defensible
incremental relational contribution in this selected scope.
"""
from __future__ import annotations
import argparse
import json
import itertools
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)

try:
    import cbid_gatv2_validation_v6 as v6
except ImportError as exc:
    raise SystemExit('Place cbid_gatv2_validation_v6.py in the same directory.') from exc

KEY_GROUP = ['window_start_ms_META', 'destination_ip_META']
GROUP_TIME = 'window_start_ms_META'


def context_by_destination(part: pd.DataFrame, x: np.ndarray):
    """Label-blind leave-one-out real peer mean, plus matching peer counts."""
    if not isinstance(part.index, pd.RangeIndex) or len(part) != len(x):
        raise ValueError('Expected contiguous RangeIndex matching feature rows')
    if not np.isfinite(x).all():
        raise ValueError('Features must be finite')
    n, dim = x.shape
    peer_count = np.zeros(n, dtype=np.int32)
    peer_mean = np.zeros((n, dim), dtype=np.float32)
    for idx in part.groupby(KEY_GROUP, sort=False).indices.values():
        idx = np.asarray(idx, dtype=np.int64)
        k = len(idx) - 1
        peer_count[idx] = k
        if k:
            total = x[idx].astype(np.float64).sum(axis=0)
            peer_mean[idx] = ((total - x[idx]) / k).astype(np.float32)
    return peer_mean, peer_count


def context_random_timebin(part: pd.DataFrame, x: np.ndarray,
                           peer_count: np.ndarray, seed: int):
    """Degree-matched randomized other-HOST peers within the same time bin.

    Focal source IP is excluded from candidates, even if the same source has
    other destination records in this bin. No true label or destination is
    used to rank the randomized peers; only timestamp and source identity
    prevent target-self contamination. Matching count k is kept exactly.
    """
    if len(part) != len(x) or len(x) != len(peer_count):
        raise ValueError('Mismatching lengths')
    out = np.zeros_like(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    sources=part['source_ip_META'].astype(str).to_numpy()
    for idx in part.groupby(GROUP_TIME, sort=False).indices.values():
        idx = np.asarray(idx, dtype=np.int64)
        n = len(idx)
        if n == 1:
            if peer_count[idx[0]]:
                raise ValueError('A time bin with one record cannot have peers')
            continue
        local_sources=sources[idx]
        for position,target_row in enumerate(idx):
            k=int(peer_count[target_row])
            if k==0:
                continue
            valid=np.flatnonzero(local_sources != local_sources[position])
            if len(valid) < k:
                raise ValueError('Too few distinct-source candidate records')
            chosen=rng.choice(valid,size=k,replace=False)
            out[target_row]=x[idx[chosen]].mean(axis=0)
    return out


def features(x: np.ndarray, degree: np.ndarray, peer: np.ndarray | None):
    """Comparable information budgets for real and randomized peer contexts.

    Besides peer means, include target-peer discrepancy and cosine coherence.
    The focal row is NOT part of peer stats. When no peer exists, all relational
    features are set to zero (and local + log degree remain available).
    """
    extra = np.log1p(degree).astype(np.float32)[:, None]
    if peer is None:
        return np.concatenate((x, extra), axis=1)
    diff = np.abs(x - peer).astype(np.float32)
    norm_x=np.linalg.norm(x,axis=1)
    norm_p=np.linalg.norm(peer,axis=1)
    cosine=np.sum(x*peer,axis=1)/(np.maximum(norm_x*norm_p,1e-12))
    has_peer=(degree>0)
    diff[~has_peer]=0
    cosine[~has_peer]=0
    coherence=np.clip(cosine,-1,1).astype(np.float32)[:,None]
    result=np.concatenate((x,extra,peer,diff,coherence),axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite relational feature')
    return result


def score(y, prob):
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= 0.5).astype(int)
    return {
        'average_precision': float(average_precision_score(y, prob)),
        'macro_f1_fixed_0_5': float(f1_score(y, pred, average='macro', labels=[0,1], zero_division=0)),
        'mcc_fixed_0_5': float(matthews_corrcoef(y, pred)),
        'positive_recall_fixed_0_5': float(recall_score(y, pred, zero_division=0)),
        'positive_precision_fixed_0_5': float(precision_score(y, pred, zero_division=0)),
    }


def bootstrap_delta(y, p_a, p_b, block_ids, *, B=300, seed=20261124):
    """Exploratory 30-min time-block bootstrap (NOT independent campaigns)."""
    rng = np.random.default_rng(seed)
    block, inv = np.unique(block_ids, return_inverse=True)
    indices = [np.where(inv == j)[0] for j in range(len(block))]
    deltas = []
    for _ in range(B):
        sampled = rng.integers(0, len(block), size=len(block))
        ids = np.concatenate([indices[j] for j in sampled])
        yy = y[ids]
        if len(np.unique(yy)) != 2:
            continue
        deltas.append(average_precision_score(yy, p_a[ids]) -
                      average_precision_score(yy, p_b[ids]))
    return {'block_size_minutes':30, 'blocks':len(block),
            'replicates_valid':len(deltas),
            'delta_ap_point':float(average_precision_score(y,p_a)-average_precision_score(y,p_b)),
            'delta_ap_bootstrap_q025':float(np.quantile(deltas, .025)) if deltas else None,
            'delta_ap_bootstrap_q975':float(np.quantile(deltas, .975)) if deltas else None,
            'limitation':'time-block bootstrap; not independent devices, campaigns, or external test'}


def test_no_self():
    """Synthetic topology test. No real network dataset is accessed."""
    p=pd.DataFrame({
        'window_start_ms_META':[1000,1000,1000,1000,2000,2000,2000,3000],
        'destination_ip_META':['a','a','b','b','a','a','c','z'],
        'source_ip_META':['u0','u1','u0','u3','u0','u1','u2','u9']})
    x=np.array([[1.,0],[3.,0],[8.,2],[9.,2],[13.,1],[17.,1],[22.,2],[30.,1]],dtype=np.float32)
    peer,deg=context_by_destination(p,x)
    assert deg.tolist()==[1,1,1,1,1,1,0,0]
    assert np.allclose(peer[:2], x[[1,0]])
    assert np.allclose(peer[2:4], x[[3,2]])
    assert np.allclose(peer[4:6], x[[5,4]])
    assert np.all(peer[6:]==0)
    for s in (0,5,1024):
        rand=context_random_timebin(p,x,deg,s)
        assert np.all(rand[deg==0]==0)
        for j in range(len(p)):
            if deg[j] == 0:
                continue
            other = [ii for ii in range(len(p))
                     if p.at[ii, 'window_start_ms_META'] == p.at[j, 'window_start_ms_META']
                     and p.at[ii, 'source_ip_META'] != p.at[j, 'source_ip_META']]
            possibilities = [x[list(ids)].mean(axis=0)
                             for ids in itertools.combinations(other,int(deg[j]))]
            assert any(np.allclose(rand[j], pm, atol=1e-5)
                       for pm in possibilities), (s,j,'self included or count mismatch')
    assert np.array_equal(context_random_timebin(p,x,deg,7),context_random_timebin(p,x,deg,7))
    print('SELF-TEST OK: real peers exclude target, null excludes target, per-bin degree matched')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--input',type=Path,default=v6.DEFAULT_INPUT)
    parser.add_argument('--output-root',type=Path,default=Path.home()/'Downloads'/'CBID_31617_Peer_Information_Gate')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--algorithm',choices=['xgb','rf'],default='xgb')
    parser.add_argument('--null-repeats',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--embargo-seconds',type=int,default=120)
    args=parser.parse_args()
    if args.self_test:
        test_no_self();return
    if args.null_repeats < 1 or args.null_repeats > 10 or args.embargo_seconds < 0:
        parser.error('Invalid repeat count or embargo')
    if not args.input.is_file():
        parser.error('Missing V4 CSV '+str(args.input))
    print('DEVELOPMENT ONLY. V4 label-selected 11 targets. April24+ excluded from fitting/scoring (CSV parser may parse their columns).')
    print('Decision point: END of ten-second bin. Peers at same bin are already observed.')
    split1=v6.utc_ms('2019-04-23T16:00:00+00:00')
    split2=v6.utc_ms('2019-04-24T00:00:00+00:00')
    parts=v6.read_train_validation(args.input,split1,split2,args.embargo_seconds*1000)
    v6.summarize_split(parts)
    x, _,scaler=v6.preprocess(parts)
    groups={};counts={};valid={};random_context={}
    for part in ('train','validation'):
        groups[part],counts[part]=context_by_destination(parts[part],x[part])
        n=len(parts[part]); is_multi=(counts[part]>0)
        valid[part]=dict(n_rows=n,n_records_with_peer=int(is_multi.sum()),
                         frac_with_peer=float(is_multi.mean()),
                         max_peer_count=int(counts[part].max()),
                         n_timestamps=int(parts[part].window_start_ms_META.nunique()),
                         n_destinations=int(parts[part].destination_ip_META.nunique()))
        print(part,valid[part],flush=True)
    # Validation groups defined from labels ONLY FOR REPORTING, not for building features.
    val=parts['validation'];yv=val[v6.LABEL].to_numpy(dtype=np.int64)
    coord=(val.groupby(KEY_GROUP,sort=False)[v6.LABEL].transform('sum').to_numpy()>=2)
    subgroup=(yv==1)&coord
    print('Validation coordinated-positive source-windows:',int(subgroup.sum()),
          'unique group-windows:',int((val.groupby(KEY_GROUP)[v6.LABEL].sum()>=2).sum()))
    if args.check_only:
        print('CHECK-ONLY COMPLETE, no model training.');return
    if len(np.unique(yv))<2:
        raise ValueError('Validation must have both classes')
    yt=parts['train'][v6.LABEL].to_numpy(dtype=np.int64)
    if len(np.unique(yt))<2:
        raise ValueError('Training must have both classes')
    out=args.output_root/('development_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    meta={'scope':'EXPLORATORY 23Apr validation only, label-selected destinations; no independent test',
          'input':str(args.input),'train_before_UTC':'2019-04-23T15:58:00Z',
          'validation_from_UTC':'2019-04-23T16:02:00Z',
          'validation_to_UTC':'2019-04-24T00:00:00Z',
          'preprocessing':'train-only V6 StandardScaler after log1p seven features',
          'focal_exclusion':'true for ALL peer contexts',
          'graph':'same destination and same 10-s time bin, other source windows only; relational features peer mean, absolute target-peer difference, cosine coherence',
          'control':'random same-bin different-source records, exactly degree matched; different destination not enforced',
          'null_repeats':args.null_repeats,'seed':args.seed,'algorithm':args.algorithm,
          'split_support':valid,'holdout_day':'24Apr previously examined; this study does not retain/use its labels for fitting or scoring',
          'limitations':['11 destinations selected after seeing class labels in previous work',
                         'Same-time peers available only at end of ten-second bin',
                         'No actual GATv2 here; this is a signal-value feasibility test',
                         'Same IPs/campaign may recur across train/validation',
                         'Multiple samples are not independent source/campaign observations']}
    (out/'protocol.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    Xs={}
    Xs['local']=(x['train'],x['validation'])
    Xs['local_degree']=(features(x['train'],counts['train'],None),
                        features(x['validation'],counts['validation'],None))
    Xs['real_peer']=(features(x['train'],counts['train'],groups['train']),
                     features(x['validation'],counts['validation'],groups['validation']))
    for r in range(args.null_repeats):
        name='random_peer_'+str(r+1)
        rt=context_random_timebin(parts['train'],x['train'],counts['train'],args.seed+10000+r)
        rv=context_random_timebin(parts['validation'],x['validation'],counts['validation'],args.seed+20000+r)
        Xs[name]=(features(x['train'],counts['train'],rt),features(x['validation'],counts['validation'],rv))
    try:
        if args.algorithm=='xgb':
            from xgboost import XGBClassifier
        else:
            from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise SystemExit('Missing algorithm dependency. Use --algorithm rf as alternative.') from exc
    rows=[]; predictions={}
    weights=float((yt==0).sum()/max((yt==1).sum(),1))
    for mode,(xx,vv) in Xs.items():
        if args.algorithm=='xgb':
            model=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,
                                subsample=.85,colsample_bytree=.85,n_jobs=8,
                                objective='binary:logistic',tree_method='hist',
                                random_state=args.seed,scale_pos_weight=weights)
        else:
            model=RandomForestClassifier(n_estimators=300,random_state=args.seed,
                                         min_samples_leaf=3,n_jobs=8,
                                         class_weight='balanced_subsample')
        start=time.perf_counter()
        model.fit(xx,yt)
        prob=np.asarray(model.predict_proba(vv)[:,1],dtype=np.float64)
        assert prob.shape==(len(yv),) and np.isfinite(prob).all()
        metrics=score(yv,prob)
        report={'model':mode,'features':xx.shape[1],**metrics,
                'recall_coordinated_pos_fixed_0_5':
                    float(np.mean(prob[subgroup]>=0.5)) if subgroup.any() else None,
                'training_and_inference_seconds':round(time.perf_counter()-start,2)}
        rows.append(report);predictions[mode]=prob
        print(mode, 'AP=',round(report['average_precision'],5),
              'macroF1=',round(report['macro_f1_fixed_0_5'],5),
              'MCC=',round(report['mcc_fixed_0_5'],5),flush=True)
    metrics=pd.DataFrame(rows)
    metrics.to_csv(out/'validation_metrics.csv',index=False)
    pred_csv=pd.DataFrame({'window_start_ms':val.window_start_ms_META.to_numpy(),
                           'y_true':yv,'peers':counts['validation'],
                           'coordinated_positive_DiagnosticOnly':subgroup.astype(int)})
    for name,p in predictions.items():
        pred_csv['p_'+name]=p
    pred_csv.to_csv(out/'validation_predictions.csv',index=False)
    # Bootstrap uses time blocks instead of iid flow records. This uncertainty
    # remains exploratory because times are from one day and sources can repeat.
    timeblock=(val.window_start_ms_META.to_numpy(dtype=np.int64)//(30*60*1000))
    checks={}
    for name in ('local','local_degree') + tuple('random_peer_'+str(j+1) for j in range(args.null_repeats)):
        checks['real_minus_'+name]=bootstrap_delta(yv,predictions['real_peer'],predictions[name],
                                                    timeblock,B=300,seed=args.seed)
    (out/'paired_timeblock_bootstrap.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    real=float(metrics.loc[metrics.model=='real_peer','average_precision'].iloc[0])
    degree=float(metrics.loc[metrics.model=='local_degree','average_precision'].iloc[0])
    nulls=[float(metrics.loc[metrics.model=='random_peer_'+str(j+1),'average_precision'].iloc[0])
           for j in range(args.null_repeats)]
    verdict='POTENTIAL_SIGNAL_EXPLORATORY' if real > degree and real > max(nulls) else 'NO_RELATIONAL_ADVANTAGE_ON_VALIDATION'
    report={'verdict':verdict,'AP_real':real,'AP_local_degree':degree,
            'AP_random_nulls':nulls,'AP_gain_real_minus_degree':real-degree,
            'AP_gain_real_minus_best_null':real-max(nulls),
            'independent_test_performed':False,
            'meaning':'Developmental screening only; cannot substantiate publication claims.'}
    (out/'feasibility_verdict.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n=== RELATIONAL SIGNAL SCREEN, DEVELOPMENT ONLY ===')
    print(json.dumps(report,indent=2));print('Outputs:',out)
    print('No 24-Apr labels retained or used for fitting/scoring; no external generalization demonstrated.')

if __name__=='__main__':
    main()
