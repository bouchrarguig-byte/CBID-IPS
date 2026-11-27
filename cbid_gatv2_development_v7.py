#!/usr/bin/env python3
"""CBID-IPS #31617 — validation-only controlled diagnostic, V7.

Requires cbid_gatv2_validation_v6.py in the SAME DIRECTORY. Reuses its
READ-ONLY split, nine features, star graphs, metadata exclusion and metrics.

Four *predefined* conditions: class-weighting balanced/none x star/no edges;
all use identical residual GATv2 architecture and paired seeds. This is an
EXPLORATORY DEVELOPMENT experiment, NOT a new independent validation.
The 24-Apr-2019 holdout is NOT read. The 11 destination IPs were selected
using labels in earlier investigations.

Selection: validation average precision (AP), avoiding a fixed 0.5 decision
threshold for early stopping. Fixed-threshold macro F1, recall and precision
are reported as DESCRIPTIVE on the same validation set. Do not optimize the
threshold on these results and call it a new blind test.

Example:
  python cbid_gatv2_development_v7.py --check-only
  python cbid_gatv2_development_v7.py --seeds 11 22 33 44 55
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cbid_gatv2_validation_v6 as v6
except ImportError as exc:
    raise SystemExit('Missing cbid_gatv2_validation_v6.py beside V7. Download both into Downloads.') from exc


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument('--input', type=Path, default=v6.DEFAULT_INPUT)
    pa.add_argument('--output-root', type=Path,
                    default=Path.home() / 'Downloads' / 'CBID_NFToN_Development_v7')
    pa.add_argument('--check-only', action='store_true')
    pa.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33, 44, 55])
    pa.add_argument('--epochs', type=int, default=25)
    pa.add_argument('--patience', type=int, default=6)
    pa.add_argument('--batch-size', type=int, default=128)
    pa.add_argument('--lr', type=float, default=2e-4)
    pa.add_argument('--weight-decay', type=float, default=2e-4)
    pa.add_argument('--hidden', type=int, default=32)
    pa.add_argument('--heads', type=int, default=2)
    pa.add_argument('--dropout', type=float, default=0.2)
    pa.add_argument('--embargo-seconds', type=int, default=120)
    pa.add_argument('--validation-start', default='2019-04-23T16:00:00+00:00')
    pa.add_argument('--holdout-start', default='2019-04-24T00:00:00+00:00')
    args = pa.parse_args()
    if not args.input.is_file():
        pa.error(f'Missing input CSV: {args.input}')
    if args.epochs < 2 or args.patience < 1 or args.batch_size < 1 or args.lr <= 0:
        pa.error('Invalid optimization args')
    if not 0 <= args.dropout < 1 or args.hidden < 2 or args.heads < 1:
        pa.error('Invalid architecture args')
    if len(set(args.seeds)) != len(args.seeds) or any(x < 0 for x in args.seeds):
        pa.error('Seeds must be unique nonnegative integers')
    t1, t2 = v6.utc_ms(args.validation_start), v6.utc_ms(args.holdout_start)
    if t1 >= t2 or args.embargo_seconds < 0:
        pa.error('Invalid temporal boundaries')

    print('CBID-IPS V7: controlled development study (NOT an independent test)', flush=True)
    print('Input:', args.input, flush=True)
    print('Rows >= holdout boundary excluded by timestamp BEFORE labels are examined.', flush=True)
    parts = v6.read_train_validation(args.input, t1, t2, args.embargo_seconds * 1000)
    v6.summarize_split(parts)
    if not set(parts['train']['_graph']).isdisjoint(set(parts['validation']['_graph'])):
        raise RuntimeError('Graph group overlap detected')
    if args.check_only:
        print('CHECK OK. No models fitted, no files written.', flush=True)
        return

    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GATv2Conv
    except ImportError as exc:
        raise SystemExit('PyTorch and PyTorch Geometric required in cbid-env') from exc

    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, 'PyTorch:', torch.__version__, flush=True)
    local, _context, _scaler = v6.preprocess(parts)
    yt = parts['train'][v6.LABEL].to_numpy(dtype=int)
    yv = parts['validation'][v6.LABEL].to_numpy(dtype=int)
    counts = np.bincount(yt, minlength=2)
    class_w = len(yt) / (2 * np.maximum(counts, 1))
    print(f'Train prior={yt.mean():.4f}, val prior={yv.mean():.4f}; '
          f'class weights={class_w.round(3).tolist()}', flush=True)

    class ResidualGAT(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.local = nn.Sequential(
                nn.Linear(len(v6.FEATURES), args.hidden), nn.ReLU(),
                nn.Dropout(args.dropout), nn.Linear(args.hidden, 2))
            self.g1 = GATv2Conv(n_features, args.hidden, heads=args.heads,
                                concat=True, dropout=args.dropout)
            self.g2 = GATv2Conv(args.hidden * args.heads, args.hidden,
                                heads=args.heads, concat=False, dropout=args.dropout)
            self.graph = nn.Sequential(nn.Linear(args.hidden, args.hidden),
                                       nn.ReLU(), nn.Dropout(args.dropout),
                                       nn.Linear(args.hidden, 2))
            # Trainable small initial contribution from message passing.
            self.gate_logit = nn.Parameter(torch.tensor(-1.5, dtype=torch.float32))

        def forward(self, x, edge_index):
            local_logits = self.local(x[:, :len(v6.FEATURES)])
            g = F.elu(self.g1(x, edge_index))
            g = F.elu(self.g2(g, edge_index))
            return local_logits + torch.sigmoid(self.gate_logit) * self.graph(g)

        def gate(self):
            return float(torch.sigmoid(self.gate_logit).detach().cpu())

    out = args.output_root / ('development_v7_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True, exist_ok=False)
    conf = {
        'status': 'EXPLORATORY DEVELOPMENT, NOT EXTERNAL OR BLINDED EVALUATION',
        'data': str(args.input), 'train_before_utc': args.validation_start,
        'validation_before_utc': args.holdout_start,
        'holdout_status': '24-Apr excluded; already inspected in V5 prior work',
        'data_selection': '11 destinations selected by label-assisted prior exploration',
        'graph_definition': '10-second per-target star; artificial mean-feature target hub',
        'model': 'two-layer GATv2 with learned-gate local-logit residual, identical parameterization in each graph ablation',
        'factors': ['balanced versus unweighted cross entropy', 'star versus no host edges'],
        'selection': 'highest average_precision on SAME known 23-Apr validation; not unbiased evaluation',
        'decision_threshold': 0.5,
        'primary_metric': 'average_precision',
        'feature_columns': v6.FEATURES,
        'excluded_as_predictors': ['addresses', 'timestamps', '*AUDIT_ONLY', 'label'],
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'seed_uncertainty': 'variation in initialization only; NOT independent network/session CI',
    }
    (out / 'configuration.json').write_text(json.dumps(conf, indent=2), encoding='utf-8')
    rows = []
    histories = []
    predictions_dir = out / 'predictions'
    predictions_dir.mkdir()
    checkpoints_dir = out / 'checkpoints'
    checkpoints_dir.mkdir()
    for edges, edge_name in [(False, 'no_edges'), (True, 'star_edges')]:
        print('\nGraph variant:', edge_name, flush=True)
        graph = {split: v6.make_graphs(parts, local, split, torch, Data, edges)
                 for split in ('train', 'validation')}
        val_loader = DataLoader(graph['validation'], batch_size=args.batch_size, shuffle=False)
        for weight_name in ('none', 'balanced'):
            w = None if weight_name == 'none' else torch.tensor(class_w, dtype=torch.float32, device=device)
            for seed in args.seeds:
                v6.seed_all(seed)
                gen = torch.Generator().manual_seed(seed)
                train_loader = DataLoader(graph['train'], batch_size=args.batch_size,
                                          shuffle=True, generator=gen)
                model = ResidualGAT(len(v6.FEATURES)+2).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                crit = nn.CrossEntropyLoss(weight=w)
                best_ap, best_ep, best_state = -1., None, None
                last_better = 0
                start = time.perf_counter()
                best_metrics = None
                for epoch in range(1, args.epochs + 1):
                    model.train()
                    losses = []
                    for batch in train_loader:
                        batch = batch.to(device)
                        opt.zero_grad(set_to_none=True)
                        logits = model(batch.x, batch.edge_index)
                        loss = crit(logits[batch.source_mask], batch.y[batch.source_mask])
                        if not torch.isfinite(loss):
                            raise RuntimeError('Non-finite training loss')
                        loss.backward()
                        opt.step()
                        losses.append(float(loss.detach().cpu()))
                    p = v6.graph_infer(model, val_loader, len(yv), device, torch)
                    met = v6.metric(yv, p)
                    score = met['average_precision']
                    histories.append({
                        'edges': edge_name, 'weights': weight_name, 'seed': seed,
                        'epoch': epoch, 'train_loss': float(np.mean(losses)),
                        'validation_AP': score, 'validation_macroF1_fixed05': met['macro_f1'],
                        'validation_recall_fixed05': met['scanning_recall'],
                        'validation_precision_fixed05': met['scanning_precision'],
                        'validation_MCC_fixed05': met['mcc'], 'gate': model.gate(),
                    })
                    if score > best_ap + 1e-8:
                        best_ap, best_ep, last_better = score, epoch, epoch
                        best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
                        best_metrics = dict(met)
                    if epoch in (1, 5, 10) or epoch-last_better >= args.patience:
                        print(f'  {edge_name:10} {weight_name:8} seed={seed:2} '
                              f'epoch={epoch:2} AP={score:.4f} bestAP={best_ap:.4f} '
                              f'macroF1={met["macro_f1"]:.4f} gate={model.gate():.3f}', flush=True)
                    if epoch - last_better >= args.patience:
                        break
                model.load_state_dict(best_state)
                pp = v6.graph_infer(model, val_loader, len(yv), device, torch)
                met = v6.metric(yv, pp)
                assert abs(met['average_precision'] - best_ap) < 1e-6
                row = {'edges': edge_name, 'weights': weight_name, 'seed': seed,
                       'selected_epoch': best_ep, 'epochs_trained': epoch,
                       'parameter_count': int(sum(t.numel() for t in model.parameters())),
                       'gate': model.gate(), 'training_seconds': time.perf_counter()-start,
                       'validation_AP': met['average_precision'],
                       'validation_macroF1_fixed05': met['macro_f1'],
                       'validation_scanningF1_fixed05': met['scanning_f1'],
                       'validation_precision_fixed05': met['scanning_precision'],
                       'validation_recall_fixed05': met['scanning_recall'],
                       'validation_MCC_fixed05': met['mcc'],
                       'validation_predicted_positive_fixed05': met['predicted_positive'],
                       'validation_FP_fixed05': int(met['confusion_matrix'][0][1])}
                rows.append(row)
                tag = f'{edge_name}_{weight_name}_seed{seed}'
                pd.DataFrame({'y_true': yv, 'probability': pp,
                              'validation_row_index': np.arange(len(yv))}).to_csv(
                                  predictions_dir / f'{tag}.csv', index=False)
                torch.save({'state_dict': best_state, 'training_only': True,
                            'seed': seed, 'best_epoch': best_ep, 'graph_variant': edge_name,
                            'class_weights': weight_name}, checkpoints_dir / f'{tag}.pt')
                print(f'  SELECTED {tag}: AP={row["validation_AP"]:.4f} '
                      f'macroF1@0.5={row["validation_macroF1_fixed05"]:.4f} '
                      f'gate={row["gate"]:.3f} epoch={best_ep}', flush=True)
                pd.DataFrame(rows).to_csv(out / 'validation_metrics.csv', index=False)
                pd.DataFrame(histories).to_csv(out / 'training_history.csv', index=False)
        del graph

    df = pd.DataFrame(rows)
    summary = df.groupby(['weights','edges']).agg(
        seeds=('seed','size'), AP_mean=('validation_AP','mean'), AP_sd=('validation_AP','std'),
        macroF1_mean=('validation_macroF1_fixed05','mean'),
        macroF1_sd=('validation_macroF1_fixed05','std'),
        MCC_mean=('validation_MCC_fixed05','mean'),
        gate_mean=('gate','mean')).reset_index()
    summary.to_csv(out / 'summary_by_variant.csv', index=False)
    paired = []
    for weights in ('none', 'balanced'):
        pairs = df.loc[df.weights.eq(weights)].pivot(index='seed', columns='edges',
                                                     values='validation_AP')
        if set(pairs.columns) == {'no_edges','star_edges'}:
            for seed, r in pairs.iterrows():
                paired.append({'weights': weights, 'seed': int(seed),
                               'AP_diff_star_minus_noedges': float(r['star_edges']-r['no_edges'])})
    pd.DataFrame(paired).to_csv(out / 'paired_edge_differences.csv', index=False)
    print('\n=== DEVELOPMENT SUMMARY — SAME validation split ===')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nOutputs:', out)
    print('Do not claim these scores are independent tests or 95% CI; 24-Apr remains excluded.')

if __name__ == '__main__':
    main()
