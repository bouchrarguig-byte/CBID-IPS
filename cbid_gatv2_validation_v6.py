#!/usr/bin/env python3
"""IJ-AI #31617 / CBID-IPS: GATv2 *validation-only* reproducibility study.

Purpose: inspect train/validation learning behavior and paired graph-edge ablation
using FIVE paired random seeds, without using the previously inspected 24-Apr-2019
holdout for model selection or evaluation. No claim of independent external test.

Input: host_window_end_2019-04-23_24_10s.csv made by cbid_expand_scanning_v4.py.
The 11 target addresses were selected after using labels; exploratory scope only.

Outputs: training histories, validation predictions and scores, validation-only
reference models, timing of pre-built data inference, methodology metadata.
The original input CSV is read-only. Does not access test labels or fit on test.

Usage:
  python cbid_gatv2_validation_v6.py --check-only
  python cbid_gatv2_validation_v6.py
  python cbid_gatv2_validation_v6.py --seeds 11 22 33 44 55 --epochs 40 --patience 10

Requirements: pandas numpy scikit-learn torch torch-geometric; optionally
xgboost, matplotlib. GPU automatically used if available.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, matthews_corrcoef, balanced_accuracy_score,
    average_precision_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

FEATURES = [
    'n_flows', 'sum_in_bytes', 'sum_out_bytes', 'sum_in_pkts',
    'sum_out_pkts', 'mean_flow_duration_ms', 'n_unique_dst_ports',
    'fraction_tcp', 'fraction_udp',
]
LOG_FEATURES = FEATURES[:7]
KEYS = ['window_start_ms_META', 'destination_ip_META', 'source_ip_META']
LABEL = 'y_scanning_host_window'
DEFAULT_INPUT = Path.home() / 'Downloads' / 'CBID_NFToN_Scanning_Expanded_v4' / 'host_window_end_2019-04-23_24_10s.csv'
DEFAULT_OUTPUT = Path.home() / 'Downloads' / 'CBID_NFToN_Validation_v6'


def fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def utc_ms(s: str) -> int:
    dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
    if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError('Cutoffs must be timezone-aware UTC, e.g., 2019-04-23T16:00:00+00:00')
    return int(dt.timestamp() * 1000)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def metric(y, p, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not len(y) or len(p) != len(y) or not np.isfinite(p).all():
        raise RuntimeError('Probability or label array invalid')
    yh = (p >= threshold).astype(int)
    ans = dict(
        n=int(len(y)), positive=int((y == 1).sum()),
        accuracy=float(accuracy_score(y, yh)),
        macro_f1=float(f1_score(y, yh, average='macro', labels=[0, 1], zero_division=0)),
        scanning_f1=float(f1_score(y, yh, zero_division=0)),
        scanning_recall=float(recall_score(y, yh, zero_division=0)),
        scanning_precision=float(precision_score(y, yh, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, yh)),
        mcc=float(matthews_corrcoef(y, yh)),
        predicted_positive=int(yh.sum()),
        confusion_matrix=confusion_matrix(y, yh, labels=[0, 1]).tolist(),
        threshold=float(threshold),
    )
    if len(np.unique(y)) == 2:
        ans['average_precision'] = float(average_precision_score(y, p))
        ans['roc_auc'] = float(roc_auc_score(y, p))
    return ans


def read_train_validation(path: Path, split1: int, split2: int, embargo: int):
    """Do not retain or inspect any label from the 24-Apr-2019 held-out period."""
    required = set(FEATURES + KEYS + [LABEL])
    pieces = {'train': [], 'validation': []}
    all_columns = None
    total_pre_holdout = 0
    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
        if all_columns is None:
            all_columns = list(chunk.columns)
            missing = required - set(all_columns)
            if missing:
                raise ValueError(f'Missing input columns: {sorted(missing)}')
        tm = pd.to_numeric(chunk['window_start_ms_META'], errors='coerce')
        # Calculate masks from timestamp metadata BEFORE accessing labels or data
        # from any holdout rows. The 24-Apr rows are immediately discarded.
        mask_train = (tm < split1 - embargo).to_numpy()
        mask_val = ((tm >= split1 + embargo) & (tm < split2)).to_numpy()
        if mask_train.any():
            pieces['train'].append(chunk.loc[mask_train, list(required)].copy())
        if mask_val.any():
            pieces['validation'].append(chunk.loc[mask_val, list(required)].copy())
        total_pre_holdout += int(mask_train.sum() + mask_val.sum())
    parts = {}
    for name in ('train', 'validation'):
        if not pieces[name]:
            raise ValueError(f'No {name} rows')
        p = pd.concat(pieces[name], ignore_index=True)
        if p[KEYS].isna().any().any() or p.duplicated(KEYS).any():
            raise ValueError(f'Invalid/duplicate (time,destination,source) keys in {name}')
        p[LABEL] = pd.to_numeric(p[LABEL], errors='raise').astype(int)
        if not set(p[LABEL].unique()) == {0, 1}:
            raise ValueError(f'{name} must have both binary classes')
        p['window_start_ms_META'] = pd.to_numeric(p['window_start_ms_META'], errors='raise').astype('int64')
        p.sort_values(KEYS, inplace=True)
        p.reset_index(drop=True, inplace=True)
        p['_graph'] = p['window_start_ms_META'].astype(str) + '|' + p['destination_ip_META'].astype(str)
        parts[name] = p
    return parts


def preprocess(parts):
    raw = {}
    for split, p in parts.items():
        x = p[FEATURES].apply(pd.to_numeric, errors='coerce').to_numpy(dtype='float64')
        x[~np.isfinite(x)] = 0.0
        x = np.maximum(x, 0.0)
        x = np.minimum(x, 1e13)
        x[:, :len(LOG_FEATURES)] = np.log1p(x[:, :len(LOG_FEATURES)])
        raw[split] = x
    scaler = StandardScaler().fit(raw['train'])
    local = {s: scaler.transform(x).astype('float32') for s, x in raw.items()}
    context = {}
    for split, p in parts.items():
        x = local[split]
        cnt = p.groupby('_graph', sort=False)['_graph'].transform('size').to_numpy(dtype='float32')
        mean = pd.DataFrame(x).groupby(p['_graph'], sort=False).transform('mean').to_numpy(dtype='float32')
        context[split] = np.column_stack((x, mean, np.log1p(cnt))).astype('float32')
    return local, context, scaler


def summarize_split(parts):
    print('STUDY: validation-only, 23-Apr-2019. The 24-Apr-2019 test is NOT evaluated.')
    for key, p in parts.items():
        cnt = p.groupby('_graph')[LABEL].sum()
        print(f'{key:11} source-windows={len(p):6,d} graphs={len(cnt):6,d} '
              f'positive={int(p[LABEL].sum()):5,d} multi-source(>=2 positive)={int((cnt>=2).sum()):4,d} '
              f'range={fmt_ts(int(p.window_start_ms_META.min()))} → {fmt_ts(int(p.window_start_ms_META.max()))}',
              flush=True)


def make_graphs(parts, local, split, torch, Data, with_edges: bool):
    p = parts[split]
    x_all = local[split]
    result = []
    for _, sub in p.groupby('_graph', sort=False):
        idx = sub.index.to_numpy(dtype='int64')
        n = len(idx)
        src_x = np.column_stack((x_all[idx], np.zeros((n, 2), dtype='float32')))
        hub_x = np.r_[x_all[idx].mean(axis=0), np.array([1., np.log1p(n)], dtype='float32')][None, :]
        x = torch.from_numpy(np.vstack((src_x, hub_x)).astype('float32'))
        y = torch.from_numpy(np.r_[sub[LABEL].to_numpy(dtype='int64'), -1])
        mask = torch.from_numpy(np.r_[np.ones(n, dtype=bool), False])
        rid = torch.from_numpy(np.r_[idx, -1].astype('int64'))
        if with_edges:
            s = torch.arange(n, dtype=torch.long)
            h = torch.full((n,), n, dtype=torch.long)
            edges = torch.stack((torch.cat((s, h)), torch.cat((h, s))))
        else:
            edges = torch.empty((2, 0), dtype=torch.long)
        result.append(Data(x=x, y=y, edge_index=edges, source_mask=mask, source_row_ids=rid))
    return result


def graph_infer(model, loader, n_rows, device, torch):
    model.eval()
    probabilities = np.full(n_rows, np.nan, dtype='float64')
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index)
            mask = batch.source_mask
            prob = torch.softmax(outputs[mask], dim=-1)[:, 1].detach().cpu().numpy()
            idx = batch.source_row_ids[mask].cpu().numpy()
            probabilities[idx] = prob
    if not np.isfinite(probabilities).all():
        raise RuntimeError('Invalid/incomplete graph prediction mapping')
    return probabilities


def latency_graph(model, loader, device, torch, repeats, max_batches):
    """Batch inference; includes PyG collation and CPU↔device transfers, no raw CSV aggregation."""
    model.eval()
    count_sources = 0
    count_graphs = 0
    seconds = []
    with torch.inference_mode():
        for pass_num in range(repeats + 1):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            sources = graphs = 0
            for bi, batch in enumerate(loader):
                if bi >= max_batches:
                    break
                graphs += int(batch.num_graphs)
                sources += int(batch.source_mask.sum())
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
                _ = torch.softmax(logits[batch.source_mask], dim=-1)[:, 1].cpu()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if pass_num:
                seconds.append(elapsed)
            count_sources, count_graphs = sources, graphs
    return {
        'device': str(device), 'repeats_excluding_warmup': repeats,
        'max_batches_per_pass': max_batches,
        'sources_timed': count_sources, 'graphs_timed': count_graphs,
        'median_elapsed_sec': float(np.median(seconds)),
        'milliseconds_per_source_window': float(1e3 * np.median(seconds) / max(count_sources, 1)),
        'milliseconds_per_graph': float(1e3 * np.median(seconds) / max(count_graphs, 1)),
        'source_windows_per_sec': float(count_sources / np.median(seconds)),
        'scope': 'precomputed graphs + PyG batch collation + host CPU/GPU transfers + model forward + probabilities, excluding NetFlow extraction, feature aggregation and preprocessing',
    }


def latency_sklearn(model, data, repeats=5):
    x = data[: min(len(data), 4000)]
    _ = model.predict_proba(x)
    secs = []
    for _ in range(repeats):
        t = time.perf_counter()
        _ = model.predict_proba(x)
        secs.append(time.perf_counter() - t)
    return {
        'device': 'cpu', 'source_windows_timed': int(len(x)),
        'milliseconds_per_source_window': float(np.median(secs) * 1e3 / len(x)),
        'median_elapsed_sec': float(np.median(secs)),
        'scope': 'predict_proba on already-computed feature rows; excludes context aggregation/preprocessing',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33, 44, 55])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--dropout', type=float, default=0.15)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--embargo-seconds', type=int, default=120)
    ap.add_argument('--validation-start', default='2019-04-23T16:00:00+00:00')
    ap.add_argument('--holdout-start', default='2019-04-24T00:00:00+00:00')
    ap.add_argument('--skip-baselines', action='store_true')
    ap.add_argument('--skip-latency', action='store_true')
    ap.add_argument('--latency-batches', type=int, default=12)
    ap.add_argument('--latency-repeats', type=int, default=3)
    ap.add_argument('--skip-cpu-latency', action='store_true', help='Faster study (GPU timing only)')
    args = ap.parse_args()
    if not args.input.exists():
        ap.error(f'Missing input: {args.input}')
    if len(args.seeds) != len(set(args.seeds)) or any(s < 0 for s in args.seeds):
        ap.error('--seeds must be distinct non-negative integers')
    if args.epochs < 2 or args.patience < 1 or args.batch_size < 1 or args.lr <= 0:
        ap.error('Invalid training hyperparameters')
    if args.hidden < 2 or args.heads < 1 or not 0 <= args.dropout < 1:
        ap.error('Invalid GAT architecture')
    cut1, cut2 = utc_ms(args.validation_start), utc_ms(args.holdout_start)
    if cut1 >= cut2 or args.embargo_seconds < 0:
        ap.error('Invalid split boundaries/embargo')
    print('CBID-IPS V6: validation-only reproducibility/ablation study')
    print('Input:', args.input)
    print('CSV read in chunks; rows with timestamp >= holdout boundary are discarded BEFORE labels are examined')
    parts = read_train_validation(args.input, cut1, cut2, args.embargo_seconds * 1000)
    summarize_split(parts)
    if not set(parts['train']['_graph']).isdisjoint(set(parts['validation']['_graph'])):
        raise RuntimeError('Group leakage between train/validation')
    if args.check_only:
        print('CHECK COMPLETE: no model fit and no output files. Test labels not used.')
        return

    local, context, scaler = preprocess(parts)
    yt = parts['train'][LABEL].to_numpy(dtype=int)
    yv = parts['validation'][LABEL].to_numpy(dtype=int)
    counts = np.bincount(yt, minlength=2)
    weights = len(yt) / (2 * np.maximum(counts, 1))
    train_weights = weights[yt]
    runid = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = args.output_root / f'validation_v6_{runid}'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'CHECK_INSTRUCTIONS.txt').write_text(
        'EXPLORATORY TRAIN/VALIDATION ONLY. The 24-Apr-2019 holdout has been examined in prior V5 work. '
        'Do not present these runs as new blinded tests. Seed-wise SD measures training stochasticity '
        'on a FIXED split, not uncertainty across networks/sessions. '
        'The 11 destinations were selected after inspecting attack labels. '
        'Real-time latency excludes raw-flow collection and aggregation.\n', encoding='utf-8')
    config = {
        'script_version': 'validation_v6', 'created_utc': runid,
        'data_path': str(args.input), 'input_sha256': 'not_calculated_for_speed',
        'split': {'train': f'< {fmt_ts(cut1 - args.embargo_seconds*1000)}',
                  'validation': f'>= {fmt_ts(cut1 + args.embargo_seconds*1000)} and < {fmt_ts(cut2)}',
                  'holdout_excluded': f'>= {fmt_ts(cut2)}', 'embargo_seconds': args.embargo_seconds},
        'selection_rule': 'best validation macro-F1 at fixed threshold=0.5; no test-based tuning',
        'features': FEATURES, 'excluded': ['no IP as predictor', 'no time as predictor', '*_AUDIT_ONLY', LABEL],
        'seeds': args.seeds, 'epochs': args.epochs, 'patience': args.patience,
        'batch_size': args.batch_size, 'learning_rate': args.lr,
        'weight_decay': args.weight_decay, 'dropout': args.dropout,
        'hidden_per_head': args.hidden, 'heads': args.heads,
        'limitation': 'The relational graph is a per-destination, per-time-bin star with one synthetic mean-feature destination hub; it cannot demonstrate interactions between different destination groups.',
        'status': 'EXPLORATORY VALIDATION ONLY; does not establish superiority, real-time IPS, cognition or generalization.',
    }
    (out / 'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    rows = []
    if not args.skip_baselines:
        print('\nREFERENCE MODELS — VALIDATION ONLY', flush=True)
        variants = [('RF_local', False), ('RF_with_context', True),
                    ('XGB_local', False), ('XGB_with_context', True)]
        for name, use_context in variants:
            x = context if use_context else local
            seed_all(42)
            if name.startswith('RF'):
                model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features='sqrt',
                                               random_state=42, n_jobs=-1)
            else:
                try:
                    from xgboost import XGBClassifier
                except ImportError:
                    print('XGBoost missing: skip', name, flush=True)
                    continue
                model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=.06,
                                      subsample=.9, colsample_bytree=.9, reg_lambda=2.,
                                      objective='binary:logistic', tree_method='hist',
                                      eval_metric='logloss', n_jobs=4, random_state=42)
            t = time.perf_counter()
            model.fit(x['train'], yt, sample_weight=train_weights)
            train_sec = time.perf_counter()-t
            p = model.predict_proba(x['validation'])[:, list(model.classes_).index(1)]
            met = metric(yv, p)
            met.update(model=name, seed=42, selected_epoch=None, train_seconds=float(train_sec),
                       evaluation='validation_only')
            if not args.skip_latency:
                met['latency'] = latency_sklearn(model, x['validation'])
            rows.append(met)
            print(f"  {name:<19} macroF1={met['macro_f1']:.4f} "
                  f"scanR={met['scanning_recall']:.4f} MCC={met['mcc']:.4f}", flush=True)
            pd.DataFrame({'row_index_validation': np.arange(len(p)), 'y_true': yv, 'prob_scan': p}).to_csv(
                out / f'{name}_validation_predictions.csv', index=False)

    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GATv2Conv
    except ImportError as exc:
        raise RuntimeError('GATv2 requires torch and torch_geometric in cbid-env. '
                           'Baseline output has been saved if available.') from exc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nGATv2 PyTorch:', torch.__version__, 'device:', device, flush=True)

    class GATNet(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.g1 = GATv2Conv(n_features, args.hidden, heads=args.heads,
                                concat=True, dropout=args.dropout)
            self.g2 = GATv2Conv(args.hidden * args.heads, args.hidden,
                                heads=args.heads, concat=False, dropout=args.dropout)
            self.classifier = nn.Sequential(nn.Linear(args.hidden, args.hidden),
                                            nn.ReLU(), nn.Linear(args.hidden, 2))
        def forward(self, x, ei):
            x = F.elu(self.g1(x, ei))
            x = F.elu(self.g2(x, ei))
            return self.classifier(x)

    history_rows = []
    variants = [('GATv2_star_edges', True), ('GATv2_no_host_edges', False)]
    # Same seeded initialisation and training scheme for the two variants.
    for name, with_edges in variants:
        print('\nPREPARING GRAPH VARIANT:', name, flush=True)
        graphs = {s: make_graphs(parts, local, s, torch, Data, with_edges)
                  for s in ('train', 'validation')}
        validation_loader = DataLoader(graphs['validation'], batch_size=args.batch_size, shuffle=False)
        for si, seed in enumerate(args.seeds):
            seed_all(seed)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(graphs['train'], batch_size=args.batch_size,
                                      shuffle=True, generator=generator)
            train_eval_loader = DataLoader(graphs['train'], batch_size=args.batch_size, shuffle=False)
            model = GATNet(len(FEATURES)+2).to(device)
            optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
            best = float('-inf')
            best_state = None
            best_epoch = None
            last_improvement = 0
            start = time.perf_counter()
            for epoch in range(1, args.epochs+1):
                model.train()
                batch_losses = []
                for batch in train_loader:
                    batch = batch.to(device)
                    optim.zero_grad(set_to_none=True)
                    logits = model(batch.x, batch.edge_index)
                    loss = criterion(logits[batch.source_mask], batch.y[batch.source_mask])
                    loss.backward()
                    optim.step()
                    batch_losses.append(float(loss.detach()))
                pv = graph_infer(model, validation_loader, len(yv), device, torch)
                val = metric(yv, pv)
                pt = graph_infer(model, train_eval_loader, len(yt), device, torch)
                tr = metric(yt, pt)
                history_rows.append({
                    'model': name, 'seed': seed, 'epoch': epoch,
                    'train_loss': float(np.mean(batch_losses)),
                    'train_macro_f1': tr['macro_f1'],
                    'validation_macro_f1': val['macro_f1'],
                    'validation_average_precision': val.get('average_precision', float('nan')),
                    'validation_scan_recall': val['scanning_recall'],
                    'validation_scan_precision': val['scanning_precision'],
                    'validation_predicted_positive': val['predicted_positive'],
                })
                if val['macro_f1'] > best + 1e-8:
                    best, best_epoch = val['macro_f1'], epoch
                    best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    last_improvement = epoch
                if epoch == 1 or epoch % 5 == 0 or epoch-last_improvement >= args.patience:
                    print(f'  {name} seed={seed} epoch={epoch:2d} '
                          f'train_loss={np.mean(batch_losses):.4f} train_macroF1={tr["macro_f1"]:.4f} '
                          f'val_macroF1={val["macro_f1"]:.4f} best={best:.4f}', flush=True)
                if epoch-last_improvement >= args.patience:
                    break
            model.load_state_dict(best_state)
            pbest = graph_infer(model, validation_loader, len(yv), device, torch)
            met = metric(yv, pbest)
            met.update(model=name, seed=seed, selected_epoch=best_epoch,
                       epochs_trained=epoch, train_seconds=float(time.perf_counter()-start),
                       parameters=int(sum(x.numel() for x in model.parameters())),
                       evaluation='validation_only',
                       graph_edges_present=with_edges)
            if not args.skip_latency and si == 0:
                met['gpu_or_main_device_latency'] = latency_graph(
                    model, validation_loader, device, torch, args.latency_repeats, args.latency_batches)
                if device.type == 'cuda' and not args.skip_cpu_latency:
                    cpu_model = GATNet(len(FEATURES)+2).cpu()
                    cpu_model.load_state_dict(best_state)
                    met['cpu_latency'] = latency_graph(cpu_model, validation_loader, torch.device('cpu'),
                                                       torch, args.latency_repeats, args.latency_batches)
                    del cpu_model
            rows.append(met)
            print(f'  SELECTED {name} seed={seed} epoch={best_epoch} '
                  f'validation_macroF1={met["macro_f1"]:.4f} '
                  f'precision={met["scanning_precision"]:.4f} '
                  f'recall={met["scanning_recall"]:.4f} MCC={met["mcc"]:.4f}', flush=True)
            pd.DataFrame({'row_index_validation': np.arange(len(pbest)), 'y_true': yv,
                          'prob_scan': pbest}).to_csv(
                out / f'{name}_seed{seed}_validation_predictions.csv', index=False)
            # Archival checkpoint is useful for provenance; never loads untrusted files.
            torch.save({'state_dict': best_state, 'seed': seed, 'best_epoch': best_epoch,
                        'training_only': True, 'architecture': 'GATv2 two layers star/no-edge'},
                       out / f'{name}_seed{seed}_validation_checkpoint.pt')
            pd.DataFrame(history_rows).to_csv(out / 'training_history.csv', index=False)
        del graphs

    (out / 'validation_metrics.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    df_rows = pd.DataFrame([{k: v for k,v in r.items() if isinstance(v, (str, int, float, bool, type(None)))} for r in rows])
    df_rows.to_csv(out / 'validation_metrics.csv', index=False)
    print('\n=== TRAINING STOCHASTICITY ON SAME VALIDATION SPLIT (NOT INDEPENDENT STUDY) ===')
    for name, _ in variants:
        v = df_rows.loc[df_rows['model'].eq(name), 'macro_f1'].to_numpy()
        print(f'{name:26} n={len(v)} macro-F1 mean={np.mean(v):.4f} SD={np.std(v,ddof=1) if len(v)>1 else 0.:.4f} '
              f'min={np.min(v):.4f} max={np.max(v):.4f}')
    a = df_rows.loc[df_rows['model'] == variants[0][0], ['seed','macro_f1']]
    b = df_rows.loc[df_rows['model'] == variants[1][0], ['seed','macro_f1']]
    m = a.merge(b, on='seed', suffixes=('_edges','_no_edges'))
    d = m['macro_f1_edges'].to_numpy() - m['macro_f1_no_edges'].to_numpy()
    print(f'Paired (edges - no edges) validation macro-F1 difference: mean={d.mean():+.4f}, '
          f'SD={d.std(ddof=1) if len(d)>1 else 0.:.4f}, n={len(d)} seeds')
    (out / 'paired_seed_differences.json').write_text(json.dumps({
        'seeds': m['seed'].tolist(), 'macro_f1_diff_edges_minus_no_edges': d.tolist(),
        'mean_difference': float(d.mean()),
        'sd_of_seed_differences': float(d.std(ddof=1)) if len(d)>1 else None,
        'warning': 'Single validation split, paired seed variations only. This is NOT a cross-network 95% CI or significance test.'
    }, indent=2), encoding='utf-8')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        hist = pd.DataFrame(history_rows)
        for kind, columns in [('learning_curves', ['train_macro_f1','validation_macro_f1']),
                              ('training_loss', ['train_loss'])]:
            fig, ax = plt.subplots(figsize=(8,5))
            for (name, seed), dd in hist.groupby(['model','seed']):
                for col in columns:
                    ax.plot(dd['epoch'], dd[col], label=f'{name}, seed {seed}, {col}', alpha=.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Macro-F1' if kind == 'learning_curves' else 'Training cross-entropy')
            ax.set_title('Training vs validation (23-Apr-2019 only)' if kind=='learning_curves' else 'Training losses')
            ax.grid(alpha=.2)
            if len(args.seeds) <= 5:
                ax.legend(fontsize=6, ncol=2)
            fig.tight_layout()
            fig.savefig(out / f'{kind}.png', dpi=160)
            plt.close(fig)
    except ImportError:
        print('matplotlib not found: training_history.csv still available')

    print('\nFILES SAVED:', out)
    print('  validation_metrics.csv / .json | training_history.csv | paired_seed_differences.json')
    print('  learning_curves.png / training_loss.png if matplotlib installed')
    print('  per-model validation predictions and GATv2 checkpoints')
    print('NO 24-Apr-2019 holdout predictions have been generated or inspected.')


if __name__ == '__main__':
    main()
