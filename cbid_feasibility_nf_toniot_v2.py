#!/usr/bin/env python3
"""CBID-IPS IJ-AI #31617 — feasibility v2 (development only).

Improvements over the supplied Claude prototype:
- Can stream NF-ToN-IoT-v3.csv from the original ZIP, no 5-GB extraction.
- Inventory exposes label-blind day volumes (never attack counts for candidate test days).
- In check mode, A and B are disjoint; C is explicitly excluded. 2019-04-24
  is blocked as a fresh B/C period because it was previously inspected in V5.
- Context is computed from ALL observed groups, before labels restrict which
  observations can be scored for the 'scanning vs benign' binary question.
- Groups are aggregated chunkwise without concatenating millions of raw flows.
- Calibration uses whole chronological 10-second bins with an embargo.
- NO GNN training. B feasibility results are NOT independent final-test results.

Important limitations: choosing B/C using their label counts invalidates a blind C;
B is a *feasibility* set for design, not a blind test. Coordinated groups are
identified using ground-truth labels *for analysis only* (not features).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)

FIELDS = {
    'src': 'IPV4_SRC_ADDR', 'dst': 'IPV4_DST_ADDR',
    'dport': 'L4_DST_PORT', 'proto': 'PROTOCOL',
    'in_bytes': 'IN_BYTES', 'out_bytes': 'OUT_BYTES',
    'in_pkts': 'IN_PKTS', 'out_pkts': 'OUT_PKTS',
    'dur': 'FLOW_DURATION_MILLISECONDS',
    'end': 'FLOW_END_MILLISECONDS', 'attack': 'Attack',
}
LOCAL = ['n_flows', 'in_bytes', 'out_bytes', 'in_pkts', 'out_pkts',
         'mean_dur', 'n_dports', 'frac_tcp', 'frac_udp']
DEFAULT_ARCHIVE = Path.home() / 'Downloads' / '02934b58528a226b_NFV3DATA-A11964_A11964.zip'
KNOWN_TESTED = {'2019-04-23', '2019-04-24'}


def parse_days(value):
    return {x.strip() for x in value.split(',') if x.strip()}


def day_code(day):
    return int(datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() // 86400)


def validate_dates(A, B, C, used):
    for period in [A, B, C, used]:
        for d in period:
            try:
                day_code(d)
            except ValueError:
                raise SystemExit(f'Invalid UTC date: {d!r} (expected YYYY-MM-DD)')
    if not A or not B or not C:
        raise SystemExit('CHECK requires nonempty --fit-days (A), --eval-days (B), and --exclude-days (C).')
    if A & B or A & C or B & C:
        raise SystemExit('A/B/C dates must be completely disjoint.')
    if (B | C) & used:
        raise SystemExit('A previously inspected day cannot serve as fresh B or C: '
                         + ', '.join(sorted((B | C) & used)))


def iter_raw(args, cols):
    """Stream requested CSV columns either from path or original ZIP member."""
    raw_path = Path(args.csv) if args.csv else Path(args.archive)
    if not raw_path.is_file():
        raise SystemExit(f'Dataset file not found: {raw_path}')
    try:
        if raw_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(raw_path) as zf:
                members = [m for m in zf.infolist() if not m.is_dir() and
                           Path(m.filename).name.lower() == 'nf-ton-iot-v3.csv']
                if len(members) != 1:
                    raise SystemExit(f'Expected one NF-ToN-IoT-v3.csv in archive; found {len(members)}')
                with zf.open(members[0]) as fh:
                    yield from pd.read_csv(fh, usecols=cols, chunksize=args.chunk_size,
                                           dtype={k: 'string' for k in
                                                  (FIELDS['src'], FIELDS['dst'], FIELDS['dport'], FIELDS['attack'])
                                                  if k in cols}, low_memory=False)
        else:
            yield from pd.read_csv(raw_path, usecols=cols, chunksize=args.chunk_size,
                                   dtype={k: 'string' for k in
                                          (FIELDS['src'], FIELDS['dst'], FIELDS['dport'], FIELDS['attack'])
                                          if k in cols}, low_memory=False)
    except ValueError as e:
        raise SystemExit('Missing or invalid CSV columns. Verify source header. Details: '+str(e)) from e


def inventory(args):
    """Read no attack label column: safe candidate-day traffic volume inventory."""
    counts = Counter()
    total = 0
    for c in iter_raw(args, [FIELDS['end']]):
        t = pd.to_numeric(c[FIELDS['end']], errors='coerce')
        days = (t.dropna().astype('int64') // 86_400_000).value_counts()
        for d, n in days.items():
            counts[int(d)] += int(n)
        total += len(c)
        if args.verbose:
            print(f'Parsed {total:,} rows...', flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res = []
    for d, n in sorted(counts.items()):
        day = datetime.fromtimestamp(d * 86400, timezone.utc).strftime('%Y-%m-%d')
        res.append({'utc_day': day, 'total_flows': n,
                    'previously_used_V5_V8': day in KNOWN_TESTED})
    pd.DataFrame(res).to_csv(out/'inventory_by_day_LABEL_BLIND.csv', index=False)
    print('\nLABEL-BLIND UTC DAY INVENTORY (NO CLASS COUNTS):')
    print(pd.DataFrame(res).to_string(index=False))
    print(f'\nTotal rows scanned: {total:,}; output: {out / "inventory_by_day_LABEL_BLIND.csv"}')
    print('Do not choose final C based on its attack-label distribution.')


def aggregate_selected(args, allowed, target):
    """Keep one aggregation per (destination, 10s time bin, source), not all raw rows.

    Accumulator: n, sum inbound/outbound bytes, packets, duration, tcp count,
    udp count, max target flag, max other-attack flag, latest time, ports set.
    """
    cols = list(FIELDS.values())
    codes = {day_code(d) for d in allowed}
    acc = {}
    kept = 0; scanned = 0; malformed = 0
    groupkeys = [FIELDS['dst'], 'time_bin', FIELDS['src']]
    for chunk in iter_raw(args, cols):
        scanned += len(chunk)
        t = pd.to_numeric(chunk[FIELDS['end']], errors='coerce')
        ok = t.notna() & ((t.fillna(-1).astype('int64') // 86_400_000).isin(codes))
        part = chunk.loc[ok].copy()
        if part.empty:
            continue
        kept += len(part)
        part[FIELDS['end']] = t.loc[ok].astype('int64')
        part['time_bin'] = part[FIELDS['end']] // (args.bin_seconds * 1000)
        for k in ('in_bytes','out_bytes','in_pkts','out_pkts','dur','proto'):
            v = FIELDS[k]
            part[v] = pd.to_numeric(part[v], errors='coerce').fillna(0)
        part = part.dropna(subset=[FIELDS['src'], FIELDS['dst']])
        malformed += int(len(chunk.loc[ok]) - len(part))
        if part.empty:
            continue
        labels = part[FIELDS['attack']].astype(str).str.strip().str.lower()
        part['_target'] = labels.eq(target).astype('int8')
        part['_other'] = (~labels.isin((target,'benign'))).astype('int8')
        part['_tcp'] = part[FIELDS['proto']].eq(6).astype('int8')
        part['_udp'] = part[FIELDS['proto']].eq(17).astype('int8')
        # Sort by observed event time not needed; associative sums permit unordered CSV.
        gr = part.groupby(groupkeys, sort=False, observed=True)
        agg = gr.agg(
            n=(FIELDS['end'],'size'), in_b=(FIELDS['in_bytes'],'sum'),
            out_b=(FIELDS['out_bytes'],'sum'), in_p=(FIELDS['in_pkts'],'sum'),
            out_p=(FIELDS['out_pkts'],'sum'), dur=(FIELDS['dur'],'sum'),
            tcp=('_tcp','sum'), udp=('_udp','sum'),
            positive=('_target','max'), other=('_other','max'),
            max_ts=(FIELDS['end'],'max'))
        ports = gr[FIELDS['dport']].agg(lambda x: set(x.dropna().astype(str)))
        # Python dict update across CSV chunks avoids loading raw A+B rows together.
        for (dst, time_bin, src), row in agg.iterrows():
            key = (str(dst), int(time_bin), str(src))
            vals = [int(row['n']), float(row['in_b']), float(row['out_b']),
                    float(row['in_p']), float(row['out_p']), float(row['dur']),
                    int(row['tcp']), int(row['udp']), int(row['positive']),
                    int(row['other']), int(row['max_ts']), ports.loc[(dst,time_bin,src)]]
            old = acc.get(key)
            if old is None:
                acc[key] = vals
            else:
                for j in range(8):
                    old[j] += vals[j]
                old[8] = max(old[8], vals[8]); old[9] = max(old[9], vals[9])
                old[10] = max(old[10], vals[10]); old[11].update(vals[11])
        if len(acc) > args.max_groups:
            raise SystemExit(f'Exceeded --max-groups={args.max_groups:,}; '
                             'reduce selected days or increase available RAM.')
        if args.verbose:
            print(f'Scanned {scanned:,} flows; selected {kept:,}; groups {len(acc):,}',flush=True)
    if not acc:
        raise SystemExit(f'No data found for selected dates: {sorted(allowed)}')
    print(f'Raw flows scanned {scanned:,}; selected {kept:,}; '
          f'groups {len(acc):,}; rows missing src/dst {malformed:,}')
    records = []
    for (dst,bin_i,src),v in acc.items():
        n=v[0]
        records.append((dst,bin_i,src,n,v[1],v[2],v[3],v[4],v[5]/max(n,1),
                        len(v[11]),v[6]/max(n,1),v[7]/max(n,1),v[8],v[9]))
    cols_out = ['dst','bin','src']+LOCAL+['y','other']
    obs = pd.DataFrame.from_records(records,columns=cols_out)
    # Label-blind contextual characteristics across ALL groups, including other attacks.
    dw = obs.groupby(['dst','bin'],sort=False,observed=True)
    raw_total_flows = dw['n_flows'].transform('sum')
    ctx = dw[LOCAL].transform('mean').add_prefix('ctx_mean_')
    obs = pd.concat([obs,ctx],axis=1)
    obs['ctx_n_sources'] = np.log1p(dw['src'].transform('nunique'))
    obs['ctx_total_flows'] = np.log1p(raw_total_flows)
    # ground-truth analysis only: other attack groups still in network context.
    obs['n_pos_in_window'] = dw['y'].transform('sum')
    obs['coordinated'] = (obs['n_pos_in_window'] >= 2).astype('int8')
    for c in LOCAL[:7]:
        obs[c]=np.log1p(obs[c].clip(lower=0))
    for c in ['ctx_mean_'+f for f in LOCAL[:7]]:
        obs[c]=np.log1p(obs[c].clip(lower=0))
    # NOTE context mean of raw values then log1p, not mean(log1p(X)).
    # Restrict evaluation *after* all label-blind context features have been built.
    dropped=int(obs['other'].sum())
    obs=obs[obs['other']==0].copy()
    obs['day_code']=(obs['bin']*args.bin_seconds)//86400
    ctx_cols = ['ctx_mean_'+c for c in LOCAL]+['ctx_n_sources','ctx_total_flows']
    print(f'Excluded mixed/other-attack labeled groups AFTER context computation: {dropped:,}')
    return obs, ctx_cols


def coord_stats(obs,label):
    w=obs.groupby(['dst','bin']).agg(n_src=('src','nunique'), n_pos=('y','sum')).reset_index()
    pos=obs[obs['y']==1]; coord=w[w['n_pos']>=2]
    return {'period':label,'source_windows':len(obs),'positive_source_windows':len(pos),
            'dest_windows':len(w),'dest_windows_with_attack':int((w.n_pos>=1).sum()),
            'coordinated_windows_ge2_attack_sources':len(coord),
            'coordinated_windows_ge5_attack_sources':int((w.n_pos>=5).sum()),
            'destinations_with_coordination':coord.dst.nunique(),
            'positive_share_in_coord':float(pos.coordinated.mean()) if len(pos) else 0.0}


def make_model():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,
                             subsample=0.8,colsample_bytree=0.8,
                             eval_metric='logloss',random_state=0,n_jobs=-1), 'XGBoost'
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300,learning_rate=0.1,
                                              random_state=0),'HistGradientBoosting'


def threshold_fpr(y, s, alpha=.01):
    neg=np.sort(np.asarray(s)[np.asarray(y)==0])
    if len(neg)<100:
        print('WARNING: <100 benign calibration observations; 1% FPR calibration coarse.')
    if len(neg)==0:
        raise SystemExit('Calibration set has no benign observations.')
    # Exactly controls empirical negative FPR <= alpha (ties removed).
    idx=min(int(np.floor((1-alpha)*len(neg))),len(neg)-1)
    return float(np.nextafter(neg[idx], np.inf))


def evaluate(y,s,thr,coordinated):
    pred=np.asarray(s)>=thr; y=np.asarray(y); coordinated=np.asarray(coordinated)
    neg=y==0; pos=y==1; co=pos&(coordinated==1); iso=pos&(coordinated==0)
    return {'macro_F1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'FPR':float(pred[neg].mean()) if neg.any() else None,
            'average_precision':float(average_precision_score(y,s)) if pos.any() else None,
            'recall_coord':float(pred[co].mean()) if co.any() else None,
            'recall_isolated':float(pred[iso].mean()) if iso.any() else None},pred


def check(args):
    A=parse_days(args.fit_days);B=parse_days(args.eval_days);C=parse_days(args.exclude_days)
    used=parse_days(args.known_used_days)
    validate_dates(A,B,C,used)
    target=args.attack.strip().lower()
    obs, context=aggregate_selected(args,A|B,target)
    fit=obs[obs.day_code.isin({day_code(d) for d in A})].copy()
    ev=obs[obs.day_code.isin({day_code(d) for d in B})].copy()
    if not len(fit) or not len(ev) or fit.y.nunique()<2 or ev.y.nunique()<2:
        raise SystemExit('A/B lack enough positive and benign source-windows; no GO can be established.')
    bins=np.sort(fit.bin.unique())
    if len(bins)<10: raise SystemExit('Not enough distinct A time bins for calibration.')
    split=bins[int(len(bins)*.8)]
    embargo=int(math.ceil(args.embargo_seconds/args.bin_seconds))
    tr=fit[fit.bin<split-embargo]
    cal=fit[fit.bin>=split+embargo]
    if tr.y.nunique()<2 or cal.y.nunique()<2:
        raise SystemExit('Chronological A training/calibration with embargo lacks one class.')
    print(f'\nCALIBRATION: train={len(tr):,} (bins <{split-embargo}), '
          f'calibration={len(cal):,} (bins >={split+embargo}); embargo={args.embargo_seconds}s')
    stats=[coord_stats(fit,'A_fit'),coord_stats(ev,'B_feasibility')]
    print('\nCOORDINATION STATISTICS (labels for analysis, never as features):')
    print(pd.DataFrame(stats).set_index('period').T.to_string())
    results=[]; preds={}; yt=ev.y.to_numpy(); coord=ev.coordinated.to_numpy()
    baselines=[('rule_n_sources',None,['ctx_n_sources']),
               ('model_local',make_model()[0],LOCAL),
               ('model_local_context',make_model()[0],LOCAL+context)]
    # Rule is monotone in #sources; other baselines fitted on A training only.
    for name,model,features in baselines:
        if model is None:
            cal_score=cal.ctx_n_sources.to_numpy(); score=ev.ctx_n_sources.to_numpy()
        else:
            model.fit(tr[features],tr.y)
            cal_score=model.predict_proba(cal[features])[:,1]
            score=model.predict_proba(ev[features])[:,1]
        thr=threshold_fpr(cal.y.to_numpy(),cal_score,args.cal_fpr)
        row,pred=evaluate(yt,score,thr,coord)
        row.update({'model':name,'threshold_from_A':thr})
        results.append(row); preds[name]=pred
    res=pd.DataFrame(results)
    print('\nBASELINES ON B (threshold fixed ONLY on A calibration):')
    print(res.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    cp=(yt==1)&(coord==1);ncp=int(cp.sum())
    miss_context=int((cp&~preds['model_local_context']).sum())
    miss_all=int((cp&~preds['model_local_context']&~preds['model_local']&~preds['rule_n_sources']).sum())
    head=miss_context/ncp if ncp else 0.0
    windows=stats[1]['coordinated_windows_ge2_attack_sources']
    verdict='GO_FOR_RESEARCH_ONLY' if windows>=args.min_coord_windows and head>=args.min_headroom else 'NO_GO'
    print(f'\nB feasibility: coordinated windows={windows} (min {args.min_coord_windows}), '
          f'context-model missed fraction={head:.1%} (min {args.min_headroom:.0%}); {verdict}')
    print(f'Missed by all 3 non-graph baselines: {miss_all} / {ncp} coordinated positives.')
    print('Headroom is a missed-positive fraction, NOT guaranteed GNN improvement.')
    print('C was NOT used to select/train/calibrate models; B is NOT a final blind test.')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(stats).to_csv(out/'coordination_stats_A_B.csv',index=False)
    res.to_csv(out/'baselines_period_B.csv',index=False)
    result={'fit_days':sorted(A),'feasibility_days':sorted(B),'sealed_days_C':sorted(C),
            'known_used_days':sorted(used),'protocol':'development feasibility only',
            'target':target,'min_coord_windows':args.min_coord_windows,
            'min_headroom':args.min_headroom,'coordinated_windows_B':int(windows),
            'positive_coordinated_B':ncp,'context_missed_B':miss_context,
            'missed_by_all_B':miss_all,'headroom_B':head,'verdict':verdict}
    (out/'verdict.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'\nSaved to {out.resolve()}')


def main():
    p=argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('mode',choices=['inventory','check'])
    group=p.add_mutually_exclusive_group()
    group.add_argument('--csv',default=None,help='Extracted NF-ToN-IoT-v3 CSV')
    group.add_argument('--archive',default=str(DEFAULT_ARCHIVE),help='Original raw NF-ToN-IoT-v3 ZIP')
    p.add_argument('--out',default=str(Path.home()/'Downloads'/'CBID_31617_feasibility_v2'))
    p.add_argument('--chunk-size',type=int,default=150_000)
    p.add_argument('--fit-days',default='',help='A: UTC dates, comma separated')
    p.add_argument('--eval-days',default='',help='B: UTC dates, comma separated')
    p.add_argument('--exclude-days',default='',help='C: reserved FINAL UTC dates (required in check)')
    p.add_argument('--known-used-days',default='2019-04-23,2019-04-24',
                   help='Dates previously inspected; cannot be B or C')
    p.add_argument('--attack',default='scanning')
    p.add_argument('--bin-seconds',type=int,default=10)
    p.add_argument('--embargo-seconds',type=int,default=120)
    p.add_argument('--cal-fpr',type=float,default=0.01)
    p.add_argument('--min-coord-windows',type=int,default=200)
    p.add_argument('--min-headroom',type=float,default=0.10)
    p.add_argument('--max-groups',type=int,default=2_000_000)
    p.add_argument('--verbose',action='store_true')
    args=p.parse_args()
    if args.bin_seconds<=0 or args.chunk_size<=0 or args.embargo_seconds<0:
        p.error('Invalid bin, chunk or embargo size.')
    if not 0<args.cal_fpr<1 or not 0<=args.min_headroom<=1:
        p.error('Invalid FPR or headroom parameter.')
    if args.mode=='inventory':inventory(args)
    else:check(args)

if __name__=='__main__': main()
