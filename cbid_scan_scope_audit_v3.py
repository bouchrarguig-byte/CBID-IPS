#!/usr/bin/env python3
"""CBID-IPS / IJ-AI 31617: exploratory, READ-ONLY scan of NF-ToN-IoT-v3.

Purpose: determine whether other destinations/time blocks on 23-24 Apr 2019
contain sufficiently many observed multisource scanning windows AND benign-only
windows to justify a *new, pre-defined* evaluation protocol.

- Flow availability time: FLOW_END_MILLISECONDS (not start time).
- Windows: non-overlapping 10s bins by end time; blocks: 15 minutes.
- No datasets are extracted, sampled, trained, deleted or overwritten.
- NO feature columns, inferred attack coordination or revised scores produced.
- All counts use ground-truth labels, and are FOR EXPLORATORY DESIGN ONLY.
- Later hyperparameter selection must never rely on labelled final test sets.

Reads the ZIP in two streaming passes; memory usage relates to the number of
unique (time bin, destination) pairs, not 27.5 million raw rows.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ZIP = (Path.home() / 'Downloads'
               / '02934b58528a226b_NFV3DATA-A11964_A11964.zip')


def iso_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec='seconds')


def stream_rows(archive: Path):
    with zipfile.ZipFile(archive, 'r') as z:
        csv_files = [n for n in z.namelist() if n.endswith('/data/NF-ToN-IoT-v3.csv')]
        if len(csv_files) != 1:
            raise ValueError(f'Expected one NF-ToN-IoT-v3.csv inside {archive}, found {csv_files}')
        with z.open(csv_files[0], 'r') as raw:
            with io.TextIOWrapper(raw, encoding='utf-8-sig', newline='', errors='replace') as fh:
                rd = csv.DictReader(fh)
                needed = {'Label', 'Attack', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR'}
                missing = needed - set(rd.fieldnames or [])
                if missing:
                    raise ValueError(f'Missing fields: {sorted(missing)}')
                yield from rd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, default=DEFAULT_ZIP)
    ap.add_argument('--days', nargs='+', default=['2019-04-23', '2019-04-24'],
                    help='UTC dates to audit; defaults to scanning dates 23 and 24 April 2019')
    ap.add_argument('--window-s', type=int, default=10)
    ap.add_argument('--block-min', type=int, default=15)
    ap.add_argument('--max-bins', type=int, default=1_500_000,
                    help='Fail safely rather than exhaust RAM if too many unique attacked windows')
    ap.add_argument('--out', type=Path, default=(Path.home() / 'Downloads'
                    / 'CBID_NFToN_Scope_Audit_v3'))
    args = ap.parse_args()
    if not args.archive.is_file():
        ap.error(f'Archive not found: {args.archive}')
    if args.window_s < 1 or args.block_min < 1:
        ap.error('window-s and block-min must be positive')
    if (args.block_min * 60) % args.window_s != 0:
        ap.error('block duration must be divisible by window duration')
    day_range = set(args.days)
    for ds in day_range:
        datetime.strptime(ds, '%Y-%m-%d')
    win_ms = args.window_s * 1000
    block_ms = args.block_min * 60 * 1000
    day_start = {ds: int(datetime.strptime(ds, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()*1000)
                 for ds in day_range}
    ranges = [(val, val+86_400_000) for val in day_start.values()]

    def get_time(row):
        try:
            t = int(float(row['FLOW_END_MILLISECONDS']))
        except (TypeError, ValueError, OverflowError):
            return None
        return t if any(start <= t < end for start, end in ranges) else None

    print('Archive:', args.archive, flush=True)
    print('Dates UTC:', ', '.join(sorted(day_range)), 'window:', args.window_s,
          'seconds; block:', args.block_min, 'minutes', flush=True)

    # pass 1: ATTACK-labelled scanning only, preserving source identities inside
    # separate (destination, 10-s window) bins.
    attack_bins: dict[tuple[int,str], list] = {}
    attack_total = Counter()
    total = 0
    print('\nPASS 1/2 — scanning source multiplicity', flush=True)
    for row in stream_rows(args.archive):
        total += 1
        if total % 3_000_000 == 0:
            print(f'  Rows: {total:,}; attack bins: {len(attack_bins):,}', flush=True)
        if row['Label'].strip() != '1' or row['Attack'].strip().lower() != 'scanning':
            continue
        t = get_time(row)
        if t is None:
            continue
        dst, src = row['IPV4_DST_ADDR'].strip(), row['IPV4_SRC_ADDR'].strip()
        if not dst or not src:
            continue
        key = (t // win_ms, dst)
        rec = attack_bins.get(key)
        if rec is None:
            if len(attack_bins) >= args.max_bins:
                raise MemoryError('Too many attacked bins; use fewer --days or shorter scope')
            rec = [0, set()]
            attack_bins[key] = rec
        rec[0] += 1
        rec[1].add(src)
        attack_total[iso_ms((t//86_400_000)*86_400_000)[:10]] += 1
    dst_set = {dst for _,dst in attack_bins}
    print(f'  Scanning flows: {sum(attack_total.values()):,}; '
          f'attack bins: {len(attack_bins):,}; destinations: {len(dst_set):,}', flush=True)

    # pass 2: benign-only counts (including bins with no attacks). Do not keep
    # individual benign flows or arbitrary IP identifiers in RAM.
    benign_bins: Counter[tuple[int,str]] = Counter()
    benign_count = 0
    bad_time = 0
    print('\nPASS 2/2 — benign windows at scanning destinations', flush=True)
    for i,row in enumerate(stream_rows(args.archive), 1):
        if i % 3_000_000 == 0:
            print(f'  Rows: {i:,}; benign bins: {len(benign_bins):,}', flush=True)
        if row['Label'].strip() != '0' or row['Attack'].strip().lower() != 'benign':
            continue
        dst = row['IPV4_DST_ADDR'].strip()
        if dst not in dst_set:
            continue
        t = get_time(row)
        if t is None:
            bad_time += 1
            continue
        benign_bins[(t//win_ms,dst)] += 1
        benign_count += 1
    print(f'  Benign flows at candidates: {benign_count:,}; '
          f'benign bins: {len(benign_bins):,}', flush=True)

    # Aggregate into (15-min block, destination) only after both passes.
    block_stats: dict[tuple[int,str], dict] = defaultdict(lambda: dict(
        attack_windows=0, multisrc2=0, multisrc3=0, multisrc5=0,
        attack_flows=0, max_attack_sources=0, attack_and_benign_windows=0,
        benign_only_windows=0, benign_flows=0, attack_source_ids=set()))
    for (bin_no,dst), (n,srcs) in attack_bins.items():
        key = ((bin_no*win_ms)//block_ms,dst)
        st=block_stats[key]
        k=len(srcs)
        st['attack_windows'] += 1
        st['attack_flows'] += n
        st['multisrc2'] += k >= 2
        st['multisrc3'] += k >= 3
        st['multisrc5'] += k >= 5
        st['max_attack_sources'] = max(st['max_attack_sources'],k)
        st['attack_source_ids'].update(srcs)
        st['attack_and_benign_windows'] += benign_bins[(bin_no,dst)] > 0
    for (bin_no,dst),n in benign_bins.items():
        key = ((bin_no*win_ms)//block_ms,dst)
        st=block_stats[key]
        st['benign_flows'] += n
        if (bin_no,dst) not in attack_bins:
            st['benign_only_windows'] += 1

    # Eligible blocks are a *feasibility heuristic* only, not a model selection.
    rows=[]
    for (block_no,dst),st in block_stats.items():
        rows.append({
            'start_utc':iso_ms(block_no*block_ms),
            'destination':dst,
            'attack_windows':st['attack_windows'],
            'benign_only_windows':st['benign_only_windows'],
            'attack_and_benign_windows':st['attack_and_benign_windows'],
            'multi_source_windows_ge2':st['multisrc2'],
            'multi_source_windows_ge3':st['multisrc3'],
            'multi_source_windows_ge5':st['multisrc5'],
            'max_concurrent_attack_sources':st['max_attack_sources'],
            'distinct_attack_sources_in_block':len(st['attack_source_ids']),
            'attack_flows':st['attack_flows'],
            'benign_flows':st['benign_flows'],
            'eligible_exploratory':int(st['multisrc2'] >= 5 and st['benign_only_windows'] >= 5)
        })
    rows.sort(key=lambda r:(r['start_utc'],r['destination']))

    by_dst = defaultdict(list)
    for r in rows:
        by_dst[r['destination']].append(r)
    rankings=[]
    for dst,rr in by_dst.items():
        rankings.append({
            'destination':dst,
            'eligible_15min_blocks':sum(r['eligible_exploratory'] for r in rr),
            'blocks_with_any_multi_source':sum(r['multi_source_windows_ge2']>0 for r in rr),
            'multi_source_10s_windows_ge2':sum(r['multi_source_windows_ge2'] for r in rr),
            'multi_source_10s_windows_ge3':sum(r['multi_source_windows_ge3'] for r in rr),
            'all_scanning_10s_windows':sum(r['attack_windows'] for r in rr),
            'benign_only_10s_windows':sum(r['benign_only_windows'] for r in rr),
            'attack_flows':sum(r['attack_flows'] for r in rr),
            'benign_flows':sum(r['benign_flows'] for r in rr),
        })
    rankings.sort(key=lambda x:(-x['eligible_15min_blocks'],
                                -x['multi_source_10s_windows_ge2'],x['destination']))

    args.out.mkdir(parents=True, exist_ok=True)
    block_file=args.out/'scanning_15min_blocks_exploratory.csv'
    ranked_file=args.out/'destination_ranking_exploratory.csv'
    summary_file=args.out/'summary_exploratory.json'
    def to_csv(path, data):
        with path.open('w', encoding='utf-8', newline='') as out:
            if not data:
                return
            wr=csv.DictWriter(out, fieldnames=list(data[0].keys()))
            wr.writeheader()
            wr.writerows(data)
    to_csv(block_file,rows)
    to_csv(ranked_file,rankings)
    report={
        'dataset':'NF-ToN-IoT-v3', 'date_utc': sorted(day_range),
        'read_only_dataset':True, 'timestamp':'FLOW_END_MILLISECONDS',
        'window_seconds':args.window_s, 'block_minutes':args.block_min,
        'notes':[
            'EXPLORATORY label-assisted assessment of scope; it is not a blind test.',
            'Multiple labelled attacking sources in same window do NOT prove a coordinated botnet.',
            'Benign-only bins are observed traffic bins, not zero-traffic periods.',
            'IP source sets are used only for descriptive source diversity, never as ML predictor features.',
            'No model has been trained; new held-out tests must be fixed before model tuning.',
        ],
        'total_csv_rows':total, 'scanning_flows_by_day':dict(attack_total),
        'scanning_destinations':len(dst_set), 'attack_bins':len(attack_bins),
        'benign_bins':len(benign_bins),
        'destinations_with_at_least_3_eligible_blocks':sum(r['eligible_15min_blocks']>=3 for r in rankings),
        'top_destinations':rankings[:30],
    }
    summary_file.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')

    print('\n===== RESULTS =====',flush=True)
    print('Destinations with >=3 eligible 15-minute blocks:',
          report['destinations_with_at_least_3_eligible_blocks'])
    print('TOP DESTINATIONS (exploratory, NOT proof of predictive gain):')
    for r in rankings[:15]:
        print(f"  {r['destination']:18s} eligible_blocks={r['eligible_15min_blocks']:3d} "
              f"multi_10s_ge2={r['multi_source_10s_windows_ge2']:5d} "
              f"multi_10s_ge3={r['multi_source_10s_windows_ge3']:5d} "
              f"benign_only_10s={r['benign_only_10s_windows']:5d}")
    print('\nOUTPUTS:')
    print(' -',block_file)
    print(' -',ranked_file)
    print(' -',summary_file)
    print('\nNo training, no changes to ZIP files.')

if __name__=='__main__':
    main()
