#!/usr/bin/env python3
"""CBID-IPS / IJ-AI 31617: broaden a *pilot* scanning-versus-benign host-window dataset.

Two modes:
  --preflight (default): read ONLY the preceding exploratory audit CSVs; print
     day-by-day source multiplicity and benign coverage for pre-selected targets.
  --extract: read the NF-ToN-IoT-v3 ZIP once (without extracting 5 GB), aggregate
     completed flows in 10-s UTC windows by (source IP, destination IP).

No learning, no splitting, no reporting of performance, no original-file edits.
Targets were selected using attack labels in a previous exploratory audit; do
NOT claim target selection was blind. Ignore *other attack types* rather than
calling them benign. *_META and *_AUDIT_ONLY columns are NOT model inputs.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

HOME = Path.home() / 'Downloads'
DEFAULT_ARCHIVE = HOME / '02934b58528a226b_NFV3DATA-A11964_A11964.zip'
DEFAULT_AUDIT = HOME / 'CBID_NFToN_Scope_Audit_v3'
DEFAULT_OUTPUT = HOME / 'CBID_NFToN_Scanning_Expanded_v4'
DAYS = ('2019-04-23', '2019-04-24')
TIME_COLUMN = 'FLOW_END_MILLISECONDS'


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec='seconds')


def numeric(s) -> float:
    try:
        v = float(s)
        return max(0.0, v) if math.isfinite(v) and v < 1e16 else 0.0
    except (ValueError, TypeError, OverflowError):
        return 0.0


def day_from_ms(ms: int) -> str:
    return iso(ms)[:10]


@dataclass(slots=True)
class Acc:
    n: int = 0
    in_bytes: float = 0.
    out_bytes: float = 0.
    in_pkts: float = 0.
    out_pkts: float = 0.
    duration_ms: float = 0.
    tcp_flows: int = 0
    udp_flows: int = 0
    dst_ports: set[str] = field(default_factory=set)
    attacks: int = 0
    benign: int = 0

    def add(self, row, malicious: bool):
        self.n += 1
        self.in_bytes += numeric(row.get('IN_BYTES'))
        self.out_bytes += numeric(row.get('OUT_BYTES'))
        self.in_pkts += numeric(row.get('IN_PKTS'))
        self.out_pkts += numeric(row.get('OUT_PKTS'))
        self.duration_ms += numeric(row.get('FLOW_DURATION_MILLISECONDS'))
        proto = str(row.get('PROTOCOL', '')).strip().lower()
        self.tcp_flows += proto in ('6', 'tcp')
        self.udp_flows += proto in ('17', 'udp')
        port = str(row.get('L4_DST_PORT', '')).strip()
        if port:
            self.dst_ports.add(port)
        self.attacks += int(malicious)
        self.benign += int(not malicious)


def choose_destinations(ranked_file: Path, min_blocks: int):
    if not ranked_file.is_file():
        raise FileNotFoundError(f'Exploratory ranking CSV not found: {ranked_file}')
    with ranked_file.open(encoding='utf-8-sig', newline='') as fh:
        ranking = list(csv.DictReader(fh))
    chosen = [r['destination'] for r in ranking
              if int(r['eligible_15min_blocks']) >= min_blocks]
    if not chosen:
        raise RuntimeError('No destinations match min eligible blocks')
    return chosen


def load_block_audit(path: Path, chosen: set[str], days: set[str]):
    if not path.is_file():
        raise FileNotFoundError(f'Exploratory block CSV not found: {path}')
    by_day_dst = defaultdict(lambda: Counter())
    per_hour = defaultdict(lambda: Counter())
    with path.open(encoding='utf-8-sig', newline='') as fh:
        for r in csv.DictReader(fh):
            dst, day = r['destination'], r['start_utc'][:10]
            if dst not in chosen or day not in days:
                continue
            key = (day, dst)
            for attr in ('attack_windows', 'benign_only_windows',
                         'attack_and_benign_windows', 'multi_source_windows_ge2',
                         'multi_source_windows_ge3', 'multi_source_windows_ge5',
                         'eligible_exploratory'):
                value = int(r[attr])
                by_day_dst[key][attr] += value
                per_hour[(day, r['start_utc'][11:13])][attr] += value
            by_day_dst[key]['blocks'] += 1
    return by_day_dst, per_hour


def preflight(audit_dir: Path, min_blocks: int):
    chosen = choose_destinations(audit_dir / 'destination_ranking_exploratory.csv', min_blocks)
    by, hours = load_block_audit(audit_dir / 'scanning_15min_blocks_exploratory.csv',
                                 set(chosen), set(DAYS))
    print('=== EXPLORATORY PREFLIGHT — NO MODEL TRAINED ===', flush=True)
    print('Targets selected by label-assisted exploration:', len(chosen))
    print('IMPORTANT: these targets are NOT an independent blinded hold-out.')
    print('Days:', ', '.join(DAYS), '| bin=10 seconds, aligned UTC | clock=FLOW_END_MILLISECONDS')
    print(f"{'destination':18} {'day':10} {'multi2':>7} {'multi3':>7} {'benign':>8} {'eligible15':>10}")
    for dst in chosen:
        for day in DAYS:
            c = by[(day, dst)]
            print(f"{dst:18} {day:10} {c['multi_source_windows_ge2']:7d} "
                  f"{c['multi_source_windows_ge3']:7d} "
                  f"{c['benign_only_windows']:8d} {c['eligible_exploratory']:10d}")
    print('\nUTC hour summary (across selected targets; counts are target-windows, not unique global windows):')
    print(f"{'date / hour':20} {'multi2':>7} {'benign_only':>12} {'eligible15':>10}")
    for key in sorted(hours):
        c = hours[key]
        if c['multi_source_windows_ge2'] or c['eligible_exploratory']:
            print(f"{key[0]} {key[1]}:00  {c['multi_source_windows_ge2']:7d} "
                  f"{c['benign_only_windows']:12d} {c['eligible_exploratory']:10d}")
    print('\nNEXT: --extract aggregates ALL observed windows for these targets on both days,')
    print('not just eligible blocks; it does not sample based on their labels.')
    return chosen


def extract(archive: Path, outdir: Path, days: tuple[str,...], chosen: list[str],
            window_s: int, max_groups: int):
    if not archive.is_file():
        raise FileNotFoundError(f'ZIP not found: {archive}')
    win_ms = window_s * 1000
    date_ranges = []
    for d in days:
        t = int(datetime.strptime(d,'%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()*1000)
        date_ranges.append((t,t+86_400_000))
    selected = set(chosen)
    aggregate: dict[tuple[int,str,str], Acc] = {}
    counts = Counter()
    print('\n=== EXTRACTING — READ ONLY ZIP, WRITES ONLY NEW OUTPUT FILES ===', flush=True)
    print('Archive:', archive)
    print('Selected destinations:',len(chosen),'| days:', days, '| bin:',window_s,'s')
    with zipfile.ZipFile(archive) as z:
        paths = [n for n in z.namelist() if n.endswith('/data/NF-ToN-IoT-v3.csv')]
        if len(paths) != 1:
            raise RuntimeError('Expected exactly one NF-ToN-IoT-v3.csv in ZIP')
        with z.open(paths[0]) as raw, io.TextIOWrapper(raw,encoding='utf-8-sig',newline='',errors='replace') as fh:
            reader = csv.DictReader(fh)
            required = {'Label','Attack','IPV4_SRC_ADDR','IPV4_DST_ADDR',TIME_COLUMN,
                        'IN_BYTES','OUT_BYTES','IN_PKTS','OUT_PKTS','PROTOCOL','L4_DST_PORT',
                        'FLOW_DURATION_MILLISECONDS'}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError('CSV missing columns: '+str(sorted(missing)))
            for row in reader:
                counts['rows_seen'] += 1
                if counts['rows_seen'] % 3_000_000 == 0:
                    print(f"  Rows: {counts['rows_seen']:,} | selected: {counts['selected_flows']:,} "
                          f"| host-windows: {len(aggregate):,}", flush=True)
                dst = row['IPV4_DST_ADDR'].strip()
                if dst not in selected:
                    continue
                try:
                    ts = int(float(row[TIME_COLUMN]))
                except (ValueError, TypeError, OverflowError):
                    counts['bad_time_at_selected_destination'] += 1
                    continue
                if not any(begin <= ts < end for begin,end in date_ranges):
                    continue
                counts['in_scope_all_labels'] += 1
                lab = row['Label'].strip()
                attack = row['Attack'].strip().lower()
                if lab=='1' and attack=='scanning':
                    is_attack = True
                elif lab=='0' and attack=='benign':
                    is_attack = False
                else:
                    counts['excluded_other_attacks_or_inconsistent_labels'] += 1
                    continue
                src = row['IPV4_SRC_ADDR'].strip()
                if not src:
                    counts['missing_source'] += 1
                    continue
                win = ts//win_ms * win_ms
                key=(win,dst,src)
                if key not in aggregate:
                    if len(aggregate) >= max_groups:
                        raise MemoryError(f'Memory guard: >{max_groups} unique source-destination-time groups; '
                                          'reduce destinations or raise --max-groups if RAM permits')
                    aggregate[key]=Acc()
                aggregate[key].add(row,is_attack)
                counts['selected_flows'] += 1
                counts['selected_scanning_flows' if is_attack else 'selected_benign_flows'] += 1

    if not aggregate:
        raise RuntimeError('No records selected. Confirm date ranges and labels.')
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('Refusing to overwrite nonempty output directory: '+str(outdir))
    outdir.mkdir(parents=True,exist_ok=True)
    out_csv = outdir / f'host_window_end_2019-04-23_24_{window_s}s.csv'
    graph_csv = outdir / f'destination_window_audit_2019-04-23_24_{window_s}s.csv'
    group_stats: dict[tuple[int,str], list[int]] = defaultdict(lambda:[0,0,0,0])
    with out_csv.open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f)
        wr.writerow(['window_start_ms_META','window_utc_META','destination_ip_META','source_ip_META',
                    'n_flows','sum_in_bytes','sum_out_bytes','sum_in_pkts','sum_out_pkts',
                    'mean_flow_duration_ms','n_unique_dst_ports','fraction_tcp','fraction_udp',
                    'attack_flows_AUDIT_ONLY','benign_flows_AUDIT_ONLY','y_scanning_host_window'])
        for (win,dst,src),a in sorted(aggregate.items()):
            wr.writerow([win,iso(win),dst,src,a.n,int(a.in_bytes),int(a.out_bytes),
                         int(a.in_pkts),int(a.out_pkts),round(a.duration_ms/a.n,6),
                         len(a.dst_ports),round(a.tcp_flows/a.n,7),round(a.udp_flows/a.n,7),
                         a.attacks,a.benign,int(a.attacks>0)])
            st=group_stats[(win,dst)]
            st[0] += 1
            st[1] += (a.attacks > 0)
            st[2] += a.attacks
            st[3] += a.benign
    del aggregate
    with graph_csv.open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f)
        wr.writerow(['window_start_ms_META','window_utc_META','destination_ip_META',
                     'source_count','positive_hosts_AUDIT_ONLY','scanning_flows_AUDIT_ONLY',
                     'benign_flows_AUDIT_ONLY'])
        for (win,dst),st in sorted(group_stats.items()):
            wr.writerow([win,iso(win),dst,*st])
    day_roll = defaultdict(Counter)
    for (win,_),st in group_stats.items():
        c=day_roll[day_from_ms(win)]
        c['graph_windows'] += 1
        c['graphs_with_scanning'] += int(st[1]>0)
        c['graphs_multi_source_ge2'] += int(st[1]>=2)
        c['graphs_multi_source_ge3'] += int(st[1]>=3)
        c['benign_only_graphs'] += int(st[1]==0)
    report = {
       'status':'EXPLORATORY label-assisted target choice, not independent blind test',
       'dataset':'NF-ToN-IoT-v3', 'selected_destinations':chosen, 'days_utc':list(days),
       'timestamp_column':TIME_COLUMN,'window_seconds':window_s,
       'model_feature_names':['n_flows','sum_in_bytes','sum_out_bytes','sum_in_pkts',
                              'sum_out_pkts','mean_flow_duration_ms','n_unique_dst_ports',
                              'fraction_tcp','fraction_udp'],
       'excluded_columns':'all *_META, *_AUDIT_ONLY and y_scanning_host_window',
       'counters':dict(counts),'day_graph_summary':{k:dict(v) for k,v in day_roll.items()},
       'warnings':[
           'All observed scanning-vs-benign windows retained for selected targets and days; not just eligible blocks.',
           'Other attacks excluded, not recoded as benign; this is a restricted binary task, not a multiclass IDS.',
           'Sources and targets are graph identities, not confirmed compromised IoT devices.',
           'Window features use FLOW_END_MILLISECONDS and are available only at the end of each 10s bin.',
           'These source/destination targets were selected after reviewing labels: do not claim independent test.',
           'Training data must be fit independently of evaluation data; hold out by time AND/OR destination.',
           'An ablation of relations needs a comparable XGBoost model with neighborhood aggregates.'
       ]}
    (outdir/'summary_extraction_v4.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n=== EXTRACTION SUMMARY ===')
    print('Raw flow records selected:',f"{counts['selected_flows']:,}")
    print('Host-window rows:',f'{sum(st[0] for st in group_stats.values()):,}')
    print('Graph windows:',f'{len(group_stats):,}')
    for day,c in sorted(day_roll.items()):
        print(f"{day}: graphs={c['graph_windows']:,}, positives={c['graphs_with_scanning']:,}, "
              f"multi2={c['graphs_multi_source_ge2']:,}, benign_only={c['benign_only_graphs']:,}")
    print('Outputs:',outdir)
    print('No train/test modeling performed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--preflight',action='store_true',help='Default; only inspect existing audit CSVs')
    g.add_argument('--extract',action='store_true',help='Stream the 5-GB CSV within ZIP into host-windows')
    p.add_argument('--archive',type=Path,default=DEFAULT_ARCHIVE)
    p.add_argument('--audit-dir',type=Path,default=DEFAULT_AUDIT)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--min-eligible-blocks',type=int,default=3)
    p.add_argument('--window-s',type=int,default=10)
    p.add_argument('--max-groups',type=int,default=500_000)
    args=p.parse_args()
    if args.window_s<1 or args.min_eligible_blocks<1 or args.max_groups<1:
        p.error('window-s, min-eligible-blocks, max-groups must all be positive')
    chosen=preflight(args.audit_dir,args.min_eligible_blocks)
    if args.extract:
        extract(args.archive,args.output,DAYS,chosen,args.window_s,args.max_groups)
    else:
        print('\nPreflight-only. To proceed, run this same script with --extract.')


if __name__=='__main__':
    main()
