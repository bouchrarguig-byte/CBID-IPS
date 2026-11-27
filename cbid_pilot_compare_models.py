#!/usr/bin/env python3
"""CBID-IPS IJ-AI 31617: exploratory, temporally held-out host-window benchmark.

Input: host_window_end_2019-04-24_10s.csv made by cbid_extract_multisource_scanning.py

Compares:
  - Random Forest on LOCAL host-window statistics
  - XGBoost on LOCAL and CONTEXT features (if xgboost available)
  - GATv2 with communication star edges (if torch_geometric available)
  - GATv2 with exactly the same nodes but NO inter-host communication edges

IMPORTANT: This is an exploratory, *label-selected two-destination* pilot, NOT a
validated claim about general IoT generalisation, verified attack coordination,
or deployed prevention. No attack-derived or identifying columns are predictors.
Temporal forward split with a time embargo; test is evaluated only after model
selection on validation. Use independent collection for generalization claims.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

FEATURES = [
    'n_flows', 'sum_in_bytes', 'sum_out_bytes', 'sum_in_pkts',
    'sum_out_pkts', 'mean_flow_duration_ms', 'n_unique_dst_ports',
    'fraction_tcp', 'fraction_udp',
]
LOG_FEATURES = FEATURES[:7]
KEYS = ['window_start_ms_META', 'destination_ip_META', 'source_ip_META']
TARGET = 'y_scanning_host_window'
DEFAULT_INPUT = (Path.home() / 'Downloads' / 'CBID_NFToN_Scanning_Audit'
                 / 'host_window_end_2019-04-24_10s.csv')


def group_keys(df):
    return df['window_start_ms_META'].astype('int64').astype(str) + '|' + df['destination_ip_META'].astype(str)


def safe_metric(y, pred, prob=None):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    ans = {
        'support': int(len(y)), 'positive_support': int(np.sum(y == 1)),
        'negative_support': int(np.sum(y == 0)),
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y, pred, labels=[0, 1], average='weighted', zero_division=0)),
        'scan_precision': float(precision_score(y, pred, pos_label=1, zero_division=0)),
        'scan_recall': float(recall_score(y, pred, pos_label=1, zero_division=0)),
        'scan_f1': float(f1_score(y, pred, pos_label=1, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'mcc': float(matthews_corrcoef(y, pred)),
        'confusion_matrix_0_1': cm.tolist(),
    }
    if prob is not None and len(np.unique(y)) == 2:
        ans['roc_auc'] = float(roc_auc_score(y, np.asarray(prob, float)))
    return ans


def print_metric(name, metrics):
    print(f"{name:27s} N={metrics['support']:5d} "
          f"macroF1={metrics['macro_f1']:.4f} "
          f"scanR={metrics['scan_recall']:.4f} "
          f"scanP={metrics['scan_precision']:.4f} "
          f"MCC={metrics['mcc']:.4f}", flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=Path.home() / 'Downloads' / 'CBID_NFToN_Pilot_Results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embargo-seconds', type=int, default=120)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--baselines-only', action='store_true', help='Skip GATv2 / PyG')
    parser.add_argument('--diagnostic-only', action='store_true', help='Stop after split feasibility diagnostics')
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'CSV missing: {args.input}')
    if args.embargo_seconds < 0:
        parser.error('--embargo-seconds must be >= 0')
    if args.epochs < 1 or args.patience < 1:
        parser.error('--epochs and --patience must be >= 1')
    seed_all(args.seed)

    print('CBID-IPS exploratory source-window benchmark')
    print('Input:', args.input)
    df = pd.read_csv(args.input, low_memory=False)
    required = set(FEATURES + KEYS + [TARGET])
    if missing := sorted(required.difference(df.columns)):
        raise ValueError(f'Missing CSV columns: {missing}')
    if df[KEYS].isna().any().any():
        raise ValueError('Found missing host / target / timestamp key')
    if df.duplicated(KEYS).any():
        raise ValueError('Duplicate source-window-destination rows; aggregation is invalid')
    y_orig = pd.to_numeric(df[TARGET], errors='coerce')
    if y_orig.isna().any() or not set(y_orig.unique()).issubset({0, 1}):
        raise ValueError('Expected binary y_scanning_host_window')
    df[TARGET] = y_orig.astype(int)
    df['window_start_ms_META'] = pd.to_numeric(df['window_start_ms_META'], errors='raise').astype('int64')
    df = df.sort_values(['window_start_ms_META', 'destination_ip_META', 'source_ip_META']).reset_index(drop=True)
    df['_group'] = group_keys(df)

    # Same wall-clock cutoffs for *both* destinations, not per-destination splits.
    timestamps = np.sort(df['window_start_ms_META'].unique())
    if len(timestamps) < 40:
        raise ValueError(f'Too few unique time bins ({len(timestamps)}) for blocked 60/20/20 split')
    cut1, cut2 = timestamps[int(len(timestamps)*0.6)], timestamps[int(len(timestamps)*0.8)]
    embargo_ms = args.embargo_seconds * 1000
    t = df['window_start_ms_META'].to_numpy(dtype='int64')
    split_masks = {
        'train': t < cut1 - embargo_ms,
        'validation': (t >= cut1 + embargo_ms) & (t < cut2 - embargo_ms),
        'test': t >= cut2 + embargo_ms,
    }
    parts = {}
    for split, mask in split_masks.items():
        part = df.loc[mask].copy().reset_index(drop=True)
        parts[split] = part
        counts = part[TARGET].value_counts().to_dict()
        first = datetime.fromtimestamp(part['window_start_ms_META'].min()/1000, timezone.utc).isoformat() if len(part) else '-'
        last = datetime.fromtimestamp(part['window_start_ms_META'].max()/1000, timezone.utc).isoformat() if len(part) else '-'
        print(f'{split:10s} host-windows={len(part):6d} groups={part._group.nunique():5d} '
              f'classes={counts} range=[{first}, {last}]', flush=True)
    print('Temporal cutoffs UTC:',
          *(datetime.fromtimestamp(c/1000, timezone.utc).isoformat() for c in (cut1, cut2)),
          '| embargo ±', args.embargo_seconds, 'seconds')
    if any((not len(p) or p[TARGET].nunique() < 2) for p in parts.values()):
        print('STOP: train, validation, or test is empty or missing a class. '
              'Do not train. Widen the observed destinations/time range before evaluation.')
        sys.exit(2)
    gids = [set(p['_group']) for p in parts.values()]
    assert not (gids[0] & gids[1] or gids[0] & gids[2] or gids[1] & gids[2]), 'Graph split leakage!'
    if args.diagnostic_only:
        print('Diagnostic-only: no model trained or fitted.')
        return

    # Strict feature whitelist. IP, time, y, and AUDIT_ONLY columns never enter X.
    raw_arrays = {}
    for split, p in parts.items():
        raw = p[FEATURES].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
        raw[~np.isfinite(raw)] = np.nan
        raw = np.nan_to_num(raw, nan=0, posinf=0, neginf=0)
        raw = np.clip(raw, 0.0, 1e13)
        raw[:, :len(LOG_FEATURES)] = np.log1p(raw[:, :len(LOG_FEATURES)])
        raw_arrays[split] = raw
    scaler = StandardScaler()
    scaler.fit(raw_arrays['train'])
    local = {s: scaler.transform(x).astype(np.float32) for s, x in raw_arrays.items()}

    # Identical relational information for XGBoost/RF-context and GATv2.
    # The group average/number of sources is computed *only from input features*.
    context = {}
    for split, p in parts.items():
        xv = local[split]
        cnt = p.groupby('_group', sort=False)['_group'].transform('size').to_numpy()
        group_mean = pd.DataFrame(xv).groupby(p['_group'], sort=False).transform('mean').to_numpy(dtype=np.float32)
        context[split] = np.concatenate([xv, group_mean, np.log1p(cnt[:, None]).astype(np.float32)], axis=1)

    ys = {s: p[TARGET].to_numpy(dtype=int) for s, p in parts.items()}
    train_class_counts = np.bincount(ys['train'], minlength=2)
    train_class_weights = len(ys['train']) / (2 * np.maximum(train_class_counts, 1))
    train_sample_weights = train_class_weights[ys['train']]
    predictions = {}
    results = {
        'type': 'EXPLORATORY TWO-DESTINATION PILOT; NOT EXTERNAL GENERALIZATION',
        'input_path': str(args.input), 'seed': args.seed, 'embargo_seconds': args.embargo_seconds,
        'split': '60%/20%/20% of unique observed UTC 10-second bins; shared time boundaries + embargo',
        'model_selection': 'validation macro-F1 for GATv2; fixed threshold 0.5',
        'imbalance_handling': 'same inverse-class-frequency training weights in RF, XGB and GATv2',
        'feature_whitelist': FEATURES,
        'excluded_from_features': 'all *_META, *_AUDIT_ONLY, y_scanning_host_window, source_ip, destination_ip',
        'known_limitations': [
            'Destinations selected after inspecting labels: exploratory pilot, not independent external evaluation.',
            'No deduplication of individual raw flows has been performed by this aggregated-CSV benchmark.',
            'IP sources have not been verified as IoT devices; multi-source scanning is not proof of coordinated control.',
            'Temporal split tests later observations, not generalisation to entirely unseen hosts/destinations.',
            'Repeated hosts/window autocorrelation: no inferential significance or confidence intervals are claimed.',
            'Label is any scanning flow in a source-destination-window, not attack attribution of a whole device.',
            'Predictions apply to observed source-windows, not empty time bins.',
            'Node features summarize flows completed by the bin end; not available earlier in real-time.',
        ],
        'split_counts': {s: {'source_windows': int(len(p)), 'graphs': int(p._group.nunique()),
                             'positive': int(p[TARGET].sum())} for s, p in parts.items()},
        'models': {},
    }

    def record(name, prob):
        prob = np.asarray(prob, dtype=np.float64)
        if len(prob) != len(parts['test']) or not np.isfinite(prob).all():
            raise RuntimeError(f'Invalid predictions for {name}')
        label = (prob >= 0.5).astype(int)
        metric = safe_metric(ys['test'], label, prob)
        results['models'][name] = metric
        predictions[name] = prob
        print_metric(name, metric)

    for set_name, use_context in [('RF_local', False), ('RF_with_context', True)]:
        print('\nTraining', set_name, flush=True)
        feature_map = context if use_context else local
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                    max_features='sqrt',
                                    random_state=args.seed, n_jobs=-1)
        rf.fit(feature_map['train'], ys['train'], sample_weight=train_sample_weights)
        record(set_name, rf.predict_proba(feature_map['test'])[:, list(rf.classes_).index(1)])

    try:
        from xgboost import XGBClassifier
        for set_name, use_context in [('XGB_local', False), ('XGB_with_context', True)]:
            print('\nTraining', set_name, flush=True)
            feature_map = context if use_context else local
            xgb = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.06,
                                subsample=0.9, colsample_bytree=0.9,
                                reg_lambda=2.0, objective='binary:logistic',
                                tree_method='hist', n_jobs=4, random_state=args.seed,
                                eval_metric='logloss')
            xgb.fit(feature_map['train'], ys['train'], sample_weight=train_sample_weights)
            record(set_name, xgb.predict_proba(feature_map['test'])[:, list(xgb.classes_).index(1)])
    except ImportError:
        print('WARNING: xgboost missing. Skipping XGBoost; install in cbid-env to compare.', flush=True)

    if not args.baselines_only:
        try:
            import torch
            from torch import nn
            from torch.nn import functional as F
            from torch_geometric.data import Data
            from torch_geometric.loader import DataLoader
            from torch_geometric.nn import GATv2Conv
        except ImportError as exc:
            print(f'WARNING: PyTorch Geometric unavailable ({exc}). Skipping GATv2.', flush=True)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('\nGATv2 device:', device, 'PyTorch:', torch.__version__, flush=True)

            def to_graphs(split, with_edges=True):
                p = parts[split]
                f = local[split]
                graph_list = []
                for _, sub in p.groupby('_group', sort=False):
                    idx = sub.index.to_numpy(dtype=np.int64)
                    n = len(idx)
                    x_src = np.concatenate([f[idx], np.zeros((n, 2), dtype=np.float32)], axis=1)
                    x_hub = np.concatenate([
                        np.mean(f[idx], axis=0),
                        np.asarray([1.0, np.log1p(n)], dtype=np.float32),
                    ])[None, :]
                    x = torch.tensor(np.concatenate([x_src, x_hub], axis=0), dtype=torch.float)
                    y = torch.tensor(np.r_[sub[TARGET].to_numpy(dtype=np.int64), -1], dtype=torch.long)
                    source_mask = torch.tensor(np.r_[np.ones(n, dtype=bool), False])
                    src_row_ids = torch.tensor(np.r_[idx, -1], dtype=torch.long)
                    if with_edges:
                        sources = torch.arange(n, dtype=torch.long)
                        hub = torch.full((n,), n, dtype=torch.long)
                        edge_index = torch.stack([
                            torch.cat([sources, hub]), torch.cat([hub, sources])
                        ], dim=0)
                    else:
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                    graph_list.append(Data(x=x, edge_index=edge_index, y=y,
                                           source_mask=source_mask, source_row_ids=src_row_ids))
                return graph_list

            class GATNet(nn.Module):
                def __init__(self, d_in):
                    super().__init__()
                    self.g1 = GATv2Conv(d_in, 32, heads=2, concat=True, dropout=0.1)
                    self.g2 = GATv2Conv(64, 32, heads=2, concat=False, dropout=0.1)
                    self.classifier = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

                def forward(self, x, edge_index):
                    x = F.elu(self.g1(x, edge_index))
                    x = F.elu(self.g2(x, edge_index))
                    return self.classifier(x)

            def train_gat(name, with_edges):
                seed_all(args.seed)
                graph_data = {s: to_graphs(s, with_edges) for s in ('train','validation','test')}
                loaders = {
                    s: DataLoader(g, batch_size=args.batch_size, shuffle=(s == 'train'))
                    for s, g in graph_data.items()
                }
                model = GATNet(len(FEATURES)+2).to(device)
                weights = torch.tensor(train_class_weights, dtype=torch.float32, device=device)
                criterion = nn.CrossEntropyLoss(weight=weights)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
                best = -1.0
                best_state = None
                best_epoch = -1
                stale = 0

                @torch.no_grad()
                def infer(split):
                    model.eval()
                    preds = np.full(len(parts[split]), np.nan, dtype=np.float64)
                    for batch in loaders[split]:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index)
                        mask = batch.source_mask
                        probs = torch.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
                        indices = batch.source_row_ids[mask].cpu().numpy()
                        preds[indices] = probs
                    if not np.isfinite(preds).all():
                        raise RuntimeError('Missing graph predictions or wrong index alignment')
                    return preds

                for epoch in range(1, args.epochs+1):
                    model.train()
                    losses = []
                    for batch in loaders['train']:
                        batch = batch.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        out = model(batch.x, batch.edge_index)
                        loss = criterion(out[batch.source_mask], batch.y[batch.source_mask])
                        loss.backward()
                        optimizer.step()
                        losses.append(float(loss.item()))
                    val_prob = infer('validation')
                    val_score = f1_score(ys['validation'], val_prob >= 0.5,
                                         labels=[0, 1], average='macro', zero_division=0)
                    if val_score > best + 1e-6:
                        best = float(val_score)
                        best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        best_epoch = epoch
                        stale = 0
                    else:
                        stale += 1
                    if epoch == 1 or epoch % 5 == 0 or stale >= args.patience:
                        print(f'  {name} epoch={epoch:2d} loss={np.mean(losses):.4f} '
                              f'val_macroF1={val_score:.4f} best={best:.4f}', flush=True)
                    if stale >= args.patience:
                        break
                if best_state is None:
                    raise RuntimeError('GAT model did not train')
                model.load_state_dict(best_state)
                prob = infer('test')
                print(f'  {name} selected epoch={best_epoch} validation_macroF1={best:.4f}')
                record(name, prob)
                results['models'][name]['selected_epoch'] = best_epoch
                results['models'][name]['validation_macro_f1'] = best
                results['models'][name]['parameters'] = int(sum(p.numel() for p in model.parameters()))

            train_gat('GATv2_star_edges', with_edges=True)
            train_gat('GATv2_no_host_edges', with_edges=False)

    # Subgroup recall is diagnostic, not a simulated dilution experiment.
    test = parts['test'].copy()
    pos_per_group = test.groupby('_group')[TARGET].transform('sum').to_numpy(dtype=int)
    results['subgroup_recall_by_observed_positive_sources'] = {}
    buckets = [('k=1', (pos_per_group==1)),
               ('k=2-3', (pos_per_group>=2)&(pos_per_group<=3)),
               ('k>=4', (pos_per_group>=4))]
    for model_name, prob in predictions.items():
        subset = {}
        for title, bucket in buckets:
            matching = bucket & (ys['test']==1)
            support = int(matching.sum())
            subset[title] = {'positive_source_windows': support,
                             'recall': (float(np.mean(prob[matching]>=0.5)) if support else None)}
        results['subgroup_recall_by_observed_positive_sources'][model_name] = subset

    # Outputs are local to the user's machine; no raw individual IPs in report CSV.
    outdir = args.output / datetime.now(timezone.utc).strftime('pilot_%Y%m%dT%H%M%SZ')
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    p = test[['window_start_ms_META', 'destination_ip_META', TARGET]].copy()
    p['positive_sources_in_same_window_AUDIT_ONLY'] = pos_per_group
    for model_name, prob in predictions.items():
        p[f'prob_{model_name}'] = prob
    p.to_csv(outdir / 'test_predictions.csv', index=False)
    print('\nSaved:', outdir, '\n  results.json\n  test_predictions.csv')
    print('CAUTION: pilot only. No 50-host dilution test, no statistical significance, no prevention evaluation.')


if __name__ == '__main__':
    main()
