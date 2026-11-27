#!/usr/bin/env python3
"""CBID-IPS IJ-AI #31617. Read-only inspection of the NF-ToN scanning pilot.

Reports temporal distribution shift, graph composition, actual multi-source
positives in the held-out period and comparison with a trivial always-benign
baseline. Does not train, select thresholds, edit data, or assert significance.

Usage:
  python cbid_pilot_diagnostic_v2.py
  python cbid_pilot_diagnostic_v2.py --input /path/to/host_window_end_2019-04-24_10s.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT = (Path.home() / 'Downloads' / 'CBID_NFToN_Scanning_Audit'
           / 'host_window_end_2019-04-24_10s.csv')
TARGET = 'y_scanning_host_window'
FEATURES = ['n_flows', 'sum_in_bytes', 'sum_out_bytes',
            'sum_in_pkts', 'sum_out_pkts', 'mean_flow_duration_ms',
            'n_unique_dst_ports', 'fraction_tcp', 'fraction_udp']

def one_split(name, d):
    from sklearn.metrics import f1_score
    d = d.copy()
    d['_group'] = d['window_start_ms_META'].astype(str) + '|' + d['destination_ip_META'].astype(str)
    groups = d.groupby('_group', sort=False)
    counts = groups.size()
    positive_sources = groups[TARGET].sum()
    positive_graphs = positive_sources > 0
    pos = int(d[TARGET].sum())
    n = len(d)
    from collections import Counter
    print(f'\n=== {name.upper()} ===')
    print(f'host-windows: {n:,}; positives: {pos:,} ({pos/max(n, 1):.2%}); '
          f'negatives: {n-pos:,}; graph-windows: {len(counts):,}')
    if not n: return
    print(f'graph sizes: mean={counts.mean():.2f}, median={counts.median():.1f}, '
          f'max={counts.max()}, singleton={int((counts==1).sum())} '
          f'({(counts==1).mean():.1%} of graphs)')
    print('graph positives: ', int(positive_graphs.sum()),
          '| >=2 positive sources:', int((positive_sources>=2).sum()),
          '| >=4:', int((positive_sources>=4).sum()),
          '| >=5:', int((positive_sources>=5).sum()))
    print('source-window positives in graphs with:')
    for title, cond in [('k=1', positive_sources == 1),
                        ('k=2-3', positive_sources.between(2,3)),
                        ('k>=4', positive_sources >= 4)]:
        print('  ', title, ':', int(positive_sources.loc[cond].sum()))
    print('distinct IP sources:', d['source_ip_META'].nunique(),
          '| distinct targets:', d['destination_ip_META'].nunique())
    print('positive sources per destination:')
    print(d.groupby('destination_ip_META')[TARGET].agg(['size', 'sum']).to_string())
    always_benign = np.zeros(n, dtype=np.int64)
    print('always-benign benchmark: macro-F1=',
          f'{f1_score(d[TARGET], always_benign, labels=[0,1], average="macro", zero_division=0):.4f}',
          '| Scanning recall=0.0000')
    print('selected medians by class [train-to-test drift diagnostics, not for threshold tuning]:')
    for f in ['n_flows', 'sum_in_bytes', 'mean_flow_duration_ms', 'n_unique_dst_ports']:
        med = d.groupby(TARGET)[f].median().to_dict()
        print(' ', f, 'benign=', med.get(0, 'NA'), 'scan=', med.get(1, 'NA'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT)
    parser.add_argument('--embargo-seconds', type=int, default=120)
    args = parser.parse_args()
    if args.embargo_seconds < 0:
        parser.error('embargo-seconds must be nonnegative')
    if not args.input.is_file():
        parser.error(f'Input not found: {args.input}')
    df = pd.read_csv(args.input, low_memory=False)
    req = {'window_start_ms_META', 'source_ip_META', 'destination_ip_META', TARGET, *FEATURES}
    if req - set(df.columns):
        parser.error(f'Missing columns: {sorted(req - set(df.columns))}')
    df['window_start_ms_META'] = pd.to_numeric(df['window_start_ms_META'], errors='raise').astype('int64')
    df[TARGET] = pd.to_numeric(df[TARGET], errors='raise').astype('int64')
    if not df[TARGET].isin([0, 1]).all():
        parser.error('Invalid labels, must be 0 or 1')
    ts = np.sort(df['window_start_ms_META'].unique())
    if len(ts) < 40: parser.error('Insufficient time bins')
    c1, c2 = ts[int(len(ts) * 0.6)], ts[int(len(ts) * 0.8)]
    e = args.embargo_seconds * 1000
    t = df['window_start_ms_META']
    masks = {'train': t < c1-e,
             'validation': (t >= c1+e) & (t < c2-e),
             'test': t >= c2+e}
    parts = {name: df.loc[mask].copy() for name,mask in masks.items()}
    from datetime import datetime, timezone
    print('INPUT:', args.input)
    print('TIME CUTOFFS UTC:', *[datetime.fromtimestamp(x/1000, timezone.utc).isoformat() for x in (c1, c2)])
    print('EMBARGO: +/-', args.embargo_seconds, 'seconds')
    print('This is an exploratory diagnosis; test may not be used to tune the model.')
    for name, p in parts.items(): one_split(name, p)
    sources = {name: set(p['source_ip_META']) for name,p in parts.items()}
    print('\n=== SOURCE IDENTITY OVERLAP (descriptive only) ===')
    print('train/test:', len(sources['train'] & sources['test']),
          '| train/validation:', len(sources['train'] & sources['validation']),
          '| validation/test:', len(sources['validation'] & sources['test']))
    print('No claim of host-disjoint evaluation is made.')
    print('\nINTERPRETATION: Graphs with only one source cannot demonstrate inter-host effects.')
    print('If the held-out test has very few k>=2 positives, expand to other periods/targets')
    print('and reserve a NEW blind test before claiming a multi-source detection advantage.')
    print('No files were created or modified.')

if __name__ == '__main__': main()
