#!/usr/bin/env python3
"""IJ-AI #31617 — corrected RT-IoT2022 flow-window reproduction (NOT an IoT device graph).

Modes:
  --mode audit       clean, split, profile and write audit; no model fitted
  --mode baselines   evaluate Random Forest (record-majority vote, window means), optionally XGBoost
  --mode gat         evaluate matched GATv2 graph vs self-loops-only ablation
  --mode all         baselines and GATv2

Scientific caveats:
  * Dataset rows are class-sorted without reliable chronology or host identities.
  * Windows are formed WITHOUT using class labels, independently within each split,
    from the source CSV order. The input is already sorted by class, so window
    homogeneity is a structural artefact; not a deployable temporal construction.
  * A class with <window_size rows in a split has no window in that split; rare-class
    coverage is explicitly reported. No fabricated, padded, or borrowed windows.
  * Classes are deduplicated globally by features; ambiguous same-feature labels
    are excluded. Train-only preprocessing. Never fit on val/test.
  * The test set already comes from a dataset explored earlier; do NOT call it a
    new blinded test, and do NOT use test scores to select hyperparameters.
  * No SMOTE. The original flattened-window SMOTE destroys record realism; optional
    inverse-frequency sampling is performed only on genuine training windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = (Path.home() / 'Downloads' / 'telechargement 1' / 'telech' /
               'RT-IoT2022' / 'rt_iot2022_prepared.csv')
DEFAULT_OUT = Path.home() / 'Downloads' / 'CBID_RT2022_Controlled_Window_v2'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def safe_write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def inspect_clean_split(csv_path: Path, output: Path, window_size: int,
                        stride: int, proportions=(.70, .15, .15)):
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(csv_path, low_memory=False)
    if df.columns.duplicated().any():
        raise ValueError('Repeated column names detected')
    candidates = [x for x in ('label', 'Attack_type', 'Label') if x in df.columns]
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one label column: found {candidates}')
    label = candidates[0]
    if df[label].isna().any():
        raise ValueError('Target contains missing values')
    df[label] = df[label].astype(str)
    df['__orig_row'] = np.arange(len(df), dtype=np.int64)
    df['__run'] = df[label].ne(df[label].shift()).cumsum().astype(np.int32)
    features = [x for x in df.columns if x not in (label, 'no', '__orig_row', '__run')]
    if not features:
        raise ValueError('No predictor columns')
    # Hash-only grouping: a 64-bit hash collision is extremely unlikely; audit this limitation.
    fp = pd.util.hash_pandas_object(df[features], index=False)
    group_labels = pd.DataFrame({'fp': fp.to_numpy(), 'label': df[label].to_numpy()})
    ambiguous = group_labels.groupby('fp', sort=False)['label'].nunique()
    ambiguous_keys = ambiguous[ambiguous > 1].index
    bad = fp.isin(ambiguous_keys)
    n_bad = int(bad.sum())
    clean = df.loc[~bad].copy()
    clean['__fp'] = fp.loc[~bad].to_numpy()
    before_dedup = len(clean)
    clean = clean.drop_duplicates(subset='__fp', keep='first').copy()
    n_repeated = before_dedup - len(clean)
    clean.sort_values('__orig_row', inplace=True)
    classes = sorted(df[label].unique().tolist())
    encoder = LabelEncoder().fit(classes)
    clean['__y'] = encoder.transform(clean[label]).astype(np.int64)

    # Split independently INSIDE each contiguous run (not after window generation).
    # Retains class-block artificiality so this is diagnostic, never production.
    split_map = {}
    for _, block in clean.groupby('__run', sort=False):
        positions = block['__orig_row'].to_numpy()
        n = len(positions)
        a = int(np.floor(proportions[0] * n))
        b = int(np.floor((proportions[0] + proportions[1]) * n))
        split_map.update((int(i), 'train') for i in positions[:a])
        split_map.update((int(i), 'validation') for i in positions[a:b])
        split_map.update((int(i), 'test') for i in positions[b:])
    clean['__partition'] = clean['__orig_row'].map(split_map)

    part = {}
    for key in ('train', 'validation', 'test'):
        x = clean[clean['__partition'] == key].copy()
        x.reset_index(drop=True, inplace=True)
        part[key] = x
    fps = [set(part[p]['__fp']) for p in ('train', 'validation', 'test')]
    if not (fps[0].isdisjoint(fps[1]) and fps[0].isdisjoint(fps[2]) and fps[1].isdisjoint(fps[2])):
        raise AssertionError('Feature fingerprint duplication across partitions')

    def window_indices(sub):
        windows, labs, orig_range = [], [], []
        # Use ONLY row order, not class boundaries, to define windows.
        # Subsets are independently ordered by original row number; boundaries
        # remain artificial because class labels were originally grouped in blocks.
        positions = np.arange(len(sub), dtype=np.int64)
        for st in range(0, len(positions) - window_size + 1, stride):
            idx = positions[st:st + window_size]
            cls = sub['__y'].iloc[idx].to_numpy()
            # Same majority-label rule as the original f2.py script.
            windows.append(idx)
            labs.append(int(np.bincount(cls, minlength=len(classes)).argmax()))
            orig_range.append((int(sub['__orig_row'].iloc[idx[0]]),
                               int(sub['__orig_row'].iloc[idx[-1]])))
        if not windows:
            return np.empty((0, window_size), dtype=np.int64), np.empty(0, dtype=np.int64), []
        return np.stack(windows), np.asarray(labs, dtype=np.int64), orig_range

    win = {p: window_indices(part[p]) for p in part}
    # ALL windows pure by construction. Do not call them temporal windows.
    support = []
    for ic, c in enumerate(classes):
        d = {'class': c}
        for p in ('train', 'validation', 'test'):
            d[p + '_records'] = int((part[p]['__y'] == ic).sum())
            d[p + '_windows'] = int((win[p][1] == ic).sum())
        d['eligible_development_train_and_validation'] = all(d[p + '_windows'] > 0 for p in ('train', 'validation'))
        d['has_window_support_in_all_three_partitions'] = all(d[p + '_windows'] > 0 for p in ('train', 'validation', 'test'))
        support.append(d)
    sup = pd.DataFrame(support)
    sup.to_csv(output / 'per_class_split_and_window_support.csv', index=False)
    # Selection label set MUST NOT depend on test labels/support.
    eligible = [ic for ic, row in enumerate(support) if row['eligible_development_train_and_validation']]
    if not eligible:
        raise ValueError('No class has windows in train and validation; reduce window size for audit')
    mix_count = {}
    crossing_run_count = {}
    for p in ('train', 'validation', 'test'):
        w_idx, wy, orig = win[p]
        mixed, crossed = 0, 0
        with (output / f'{p}_window_manifest.csv').open('w', encoding='utf-8') as f:
            f.write('window_id,orig_first_row,orig_last_row,majority_label,n_distinct_classes,n_original_label_runs\n')
            for k, (ids, yy, rr) in enumerate(zip(w_idx, wy, orig)):
                n_classes_in_bag = len(np.unique(part[p]['__y'].iloc[ids]))
                n_runs_in_bag = len(np.unique(part[p]['__run'].iloc[ids]))
                mixed += int(n_classes_in_bag > 1)
                crossed += int(n_runs_in_bag > 1)
                f.write(f'{k},{rr[0]},{rr[1]},{classes[yy]},{n_classes_in_bag},{n_runs_in_bag}\n')
        mix_count[p] = mixed
        crossing_run_count[p] = crossed
    clean[['__orig_row', label, '__partition', '__run']].to_csv(output / 'cleaned_record_splits.csv', index=False)
    diag = {
        'input_sha256': digest(csv_path),
        'input_path': str(csv_path),
        'n_raw': int(len(df)), 'n_clean': int(len(clean)),
        'ambiguous_feature_records_removed': n_bad,
        'duplicate_feature_records_removed': int(n_repeated),
        'n_class_runs_raw': int(df['__run'].nunique()),
        'n_original_classes': len(classes), 'classes': classes,
        'number_of_features_excluding_no': len(features),
        'window_size': window_size, 'stride': stride,
        'window_majority_labels_from_current_split_only': True,
        'overlap_records_within_partition_possible': stride < window_size,
        'between_partition_record_overlap': 0,
        'feature_hash_overlap_between_partitions': 0,
        'mixed_label_windows_per_split': mix_count,
        'windows_crossing_original_label_runs_per_split': crossing_run_count,
        'fraction_windows_mixing_classes': {p: round(mix_count[p] / max(1, len(win[p][1])), 8) for p in win},
        'window_construction_uses_labels': False,
        'windows_majority_labels_used_only_as_targets': True,
        'window_homogeneity_due_to_original_class_sorted_file': True,
        'eligible_window_labels_selected_using_train_validation_ONLY': [classes[i] for i in eligible],
        'ineligible_window_labels': [classes[i] for i in range(len(classes)) if i not in eligible],
        'partitions': {p: {'records': int(len(part[p])), 'windows': int(len(win[p][1])),
                           'labels_with_windows': int(np.unique(win[p][1]).size)} for p in part},
        'limitations': [
            'Class-run-aware partitioning is label-stratified, while window construction is label-blind; CSV ordering remains class-sorted.',
            'No timestamps, no device identities; graph nodes are rows, not IoT devices.',
            'Deduplication occurs on entire dataset, without using test predictions; report this preprocessing.',
            'Repeated-feature detection uses 64-bit Pandas hash; collisions theoretically possible.',
            'No SMOTE; no synthetic/padded windows.',
            'Window rarity makes several labels unassessable at W=10; full multiclass evaluation stays at record level.',
            'Original dataset has been explored previously: test is not truly blind.',
        ],
    }
    safe_write_json(output / 'protocol_audit.json', diag)
    print('\n=== CONTROLLED DATA AUDIT ===')
    print('Raw:', len(df), '| cleaned:', len(clean), '| ambiguous removed:', n_bad, '| duplicates:', n_repeated)
    print('Contiguous class runs:', diag['n_class_runs_raw'], '| classes:', len(classes))
    print('No cross-partition feature hash overlap. Original col "no" excluded.')
    print('WARNING: windows follow class-sorted CSV order, NOT time/device identity.')
    print('Mixed-label window counts:', mix_count, '| fractions:', diag['fraction_windows_mixing_classes'])
    print(sup.to_string(index=False))
    print('Predeclared window-selection labels from TRAIN+VAL only:', [classes[i] for i in eligible])
    print('Excluded from checkpoint selection (no train+val window support):', [classes[i] for i in range(len(classes)) if i not in eligible])
    return part, win, diag, features, classes, eligible


def preprocessing(parts, feature_columns):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    tr = parts['train'][feature_columns]
    numeric = tr.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in feature_columns if c not in numeric]

    def sanitize(df):
        xx = df[feature_columns].copy()
        if numeric:
            xx[numeric] = xx[numeric].replace([np.inf, -np.inf], np.nan)
        return xx

    prep = ColumnTransformer([
        ('numerical', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), numeric),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical),
    ], sparse_threshold=0)
    X = {'train': np.asarray(prep.fit_transform(sanitize(parts['train'])), dtype=np.float32)}
    for p in ('validation', 'test'):
        X[p] = np.asarray(prep.transform(sanitize(parts[p])), dtype=np.float32)
    if not all(np.isfinite(x).all() for x in X.values()):
        raise ValueError('NaN/inf in preprocessed arrays')
    print('Preprocessing fit TRAIN ONLY. Encoded features:', X['train'].shape[1],
          '| numerical:', len(numeric), '| categorical:', categorical)
    return X, {'encoded_feature_dimension': int(X['train'].shape[1]),
               'numeric_columns': numeric, 'categorical_columns': categorical}


def score(y, pred, classes, eligible, partition, model, runtime=None):
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 classification_report, confusion_matrix)
    all_idx = np.arange(len(classes)).tolist()
    # Same labels for all models, otherwise differing supports bias comparisons.
    result = {
        'model': model, 'partition': partition, 'n_windows': int(len(y)),
        'accuracy': float(accuracy_score(y, pred)),
        'macro_F1_all_12_classes': float(f1_score(y, pred, labels=all_idx, average='macro', zero_division=0)),
        'macro_F1_eligible_classes': float(f1_score(y, pred, labels=eligible, average='macro', zero_division=0)),
        'weighted_F1': float(f1_score(y, pred, labels=all_idx, average='weighted', zero_division=0)),
        'MCC': float(matthews_corrcoef(y, pred)) if len(set(y.tolist())) > 1 else None,
        'training_seconds': runtime,
        'per_class': classification_report(y, pred, labels=all_idx, target_names=classes,
                                           output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y, pred, labels=all_idx).tolist(),
    }
    return result


def record_majority_predictions(record_preds, windows, n_classes):
    return np.asarray([np.bincount(record_preds[w], minlength=n_classes).argmax()
                       for w in windows], dtype=np.int64)


def run_baselines(args, output, parts, win, X, classes, eligible):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, f1_score, matthews_corrcoef
    models = {
        'RF': lambda: RandomForestClassifier(n_estimators=args.trees,
                                              random_state=args.seed, n_jobs=args.jobs),
    }
    if args.include_xgb:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError('Install xgboost or omit --include-xgb') from exc
        models['XGB'] = lambda: XGBClassifier(n_estimators=args.trees, max_depth=6,
                        learning_rate=.07, subsample=.85, colsample_bytree=.90,
                        eval_metric='mlogloss', tree_method='hist', n_jobs=args.jobs,
                        random_state=args.seed)

    y = {p: parts[p]['__y'].to_numpy(dtype=np.int64) for p in parts}
    n_classes = len(classes)
    summaries = []
    for mod_name, factory in models.items():
        print('\nFitting', mod_name, 'record-level (then majority vote over SAME windows)...')
        model = factory()
        t0 = time.perf_counter()
        model.fit(X['train'], y['train'])
        elapsed = time.perf_counter() - t0
        for p in ('validation', 'test'):
            y_hat_records = np.asarray(model.predict(X[p]), dtype=np.int64)
            record_metrics = {
                'model': f'{mod_name}_record', 'partition': p, 'n_records': len(y[p]),
                'accuracy': accuracy_score(y[p], y_hat_records),
                'macro_F1_all_12_classes': f1_score(y[p], y_hat_records, labels=list(range(n_classes)), average='macro', zero_division=0),
                'MCC': matthews_corrcoef(y[p], y_hat_records),
                'per_class': classification_report(y[p], y_hat_records, labels=list(range(n_classes)),
                                                   target_names=classes, output_dict=True, zero_division=0),
            }
            safe_write_json(output / f'{mod_name}_record_{p}.json', record_metrics)
            wy = win[p][1]
            if len(wy):
                w_pred = record_majority_predictions(y_hat_records, win[p][0], n_classes)
                metrics = score(wy, w_pred, classes, eligible, p, f'{mod_name}_vote10', elapsed)
                summaries.append({k: v for k, v in metrics.items() if k not in ('per_class','confusion_matrix')})
                safe_write_json(output / f'{mod_name}_vote10_{p}.json', metrics)
                pd.DataFrame({'true': [classes[i] for i in wy],
                              'pred': [classes[i] for i in w_pred]}).to_csv(output / f'{mod_name}_vote10_{p}_pred.csv', index=False)
                print(f'  {p} vote macroF1_eligible={metrics["macro_F1_eligible_classes"]:.4f} '
                      f' all12={metrics["macro_F1_all_12_classes"]:.4f} acc={metrics["accuracy"]:.4f}')

        # Window mean: train on exactly the window dataset GATv2 sees.
        print('Fitting', mod_name, 'on mean of SAME 10 records/window...')
        model_mean = factory()
        w_train = X['train'][win['train'][0]].mean(axis=1)
        # XGBoost requires 0..K-1 labels even if some rare classes lack 10-row windows.
        from sklearn.preprocessing import LabelEncoder as WindowLabelEncoder
        w_enc = WindowLabelEncoder().fit(win['train'][1])
        if len(w_enc.classes_) < 2:
            raise ValueError('Fewer than two class labels appear in training windows')
        t0 = time.perf_counter()
        model_mean.fit(w_train, w_enc.transform(win['train'][1]))
        elapsed_mean = time.perf_counter() - t0
        for p in ('validation', 'test'):
            xx = X[p][win[p][0]].mean(axis=1)
            yy = win[p][1]
            pp = w_enc.inverse_transform(np.asarray(model_mean.predict(xx), dtype=np.int64))
            m = score(yy, pp, classes, eligible, p, f'{mod_name}_window_mean', elapsed_mean)
            summaries.append({k: v for k, v in m.items() if k not in ('per_class','confusion_matrix')})
            safe_write_json(output / f'{mod_name}_window_mean_{p}.json', m)
            pd.DataFrame({'true': [classes[i] for i in yy], 'pred': [classes[i] for i in pp]}).to_csv(
                output / f'{mod_name}_window_mean_{p}_pred.csv', index=False)
            print(f'  {p} mean macroF1_eligible={m["macro_F1_eligible_classes"]:.4f} '
                  f' all12={m["macro_F1_all_12_classes"]:.4f} acc={m["accuracy"]:.4f}')
    return summaries


def run_gat(args, output, win, X, classes, eligible):
    import torch
    from sklearn.metrics import f1_score
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import WeightedRandomSampler
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATv2Conv, global_mean_pool

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('GAT device:', device, '| architecture hidden:', args.hidden, 'heads:', args.heads)
    num_labels = len(classes)

    class Net(nn.Module):
        # Three layers parallel original f2.py/smote.py (4/4/1 heads if --heads 4).
        # Preserve historical x1+x1, x2+x2 as normalization of doubled activations.
        # They are NOT mathematically residual connections; this fact is reported.
        def __init__(self, dim):
            super().__init__()
            self.g1 = GATv2Conv(dim, args.hidden, heads=args.heads, dropout=args.dropout)
            self.n1 = nn.LayerNorm(args.hidden * args.heads)
            self.g2 = GATv2Conv(args.hidden * args.heads, args.hidden,
                                heads=args.heads, dropout=args.dropout)
            self.n2 = nn.LayerNorm(args.hidden * args.heads)
            self.g3 = GATv2Conv(args.hidden * args.heads, args.hidden, heads=1,
                                dropout=args.dropout)
            self.f1 = nn.Linear(args.hidden, args.hidden // 2)
            self.f2 = nn.Linear(args.hidden // 2, num_labels)
            self.drop = nn.Dropout(args.dropout)

        def forward(self, batch):
            z = F.leaky_relu(self.g1(batch.x, batch.edge_index))
            z = self.n1(z + z)
            z = F.leaky_relu(self.g2(z, batch.edge_index))
            z = self.n2(z + z)
            z = F.leaky_relu(self.g3(z, batch.edge_index))
            z = global_mean_pool(z, batch.batch)
            return self.f2(self.drop(F.leaky_relu(self.f1(self.drop(z)))))

    def graph_data(p, edges):
        # From X indexed by the SAME label-blind window indices used by RF/XGB.
        w_idx, labels, _ = win[p]
        if edges == 'upper_triangular':
            # Original f2.py: direction i<j, NOT cosine similarity or undirected graph.
            edge = torch.combinations(torch.arange(args.window_size), r=2).T.contiguous()
        elif edges == 'self_loops':
            edge = torch.empty((2,0), dtype=torch.long)  # GATv2Conv adds self-loops.
        else:
            raise ValueError(edges)
        data = [Data(x=torch.from_numpy(X[p][ids].copy()), edge_index=edge.clone(),
                     y=torch.tensor([int(label)], dtype=torch.long))
                for ids, label in zip(w_idx, labels)]
        return data

    histories = []
    summaries = []
    for variant, edges in [('GATv2_uppertriangular', 'upper_triangular'),
                           ('GATv2_selfloops', 'self_loops')]:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        net = Net(X['train'].shape[1]).to(device)
        datasets = {p: graph_data(p, edges) for p in ('train','validation','test')}
        train_y = win['train'][1]
        if args.balance_sampler:
            counts = np.bincount(train_y, minlength=num_labels)
            weights = 1. / np.maximum(counts[train_y], 1)
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                            len(train_y), replacement=True,
                                            generator=torch.Generator().manual_seed(args.seed))
            train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(datasets['train'], batch_size=args.batch_size,
                                      shuffle=True, generator=torch.Generator().manual_seed(args.seed))
        val_loader = DataLoader(datasets['validation'], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        best_macro, best_state, best_ep, without_gain = -1., None, -1, 0
        started = time.perf_counter()

        def evaluate(loader):
            net.eval()
            ground, preds = [], []
            with torch.inference_mode():
                for d in loader:
                    d = d.to(device)
                    logits = net(d)
                    preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                    ground.extend(d.y.cpu().numpy().tolist())
            return np.asarray(ground, dtype=np.int64), np.asarray(preds, dtype=np.int64)

        for epoch in range(1, args.epochs + 1):
            net.train()
            loss_total, observations = 0., 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = net(batch)
                loss = criterion(logits, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                optimizer.step()
                loss_total += loss.item() * int(batch.num_graphs)
                observations += int(batch.num_graphs)
            scheduler.step()
            yt, yp = evaluate(val_loader)
            fm = f1_score(yt, yp, labels=eligible, average='macro', zero_division=0)
            histories.append({'variant': variant, 'epoch': epoch,
                              'train_loss': loss_total / max(observations, 1),
                              'validation_macroF1_eligible': fm})
            print(f'  {variant} epoch={epoch:2d} loss={loss_total / max(observations,1):.4f} '
                  f'val_macroF1_eligible={fm:.4f}')
            if fm > best_macro + 1e-8:
                best_macro = fm
                best_ep = epoch
                # CPU copy; checkpoint selected by validation only.
                best_state = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
                without_gain = 0
            else:
                without_gain += 1
                if without_gain >= args.patience:
                    break
        train_seconds = time.perf_counter() - started
        net.load_state_dict(best_state)
        torch.save(best_state, output / f'{variant}_selected_state_dict.pt')
        safe_write_json(output / f'{variant}_selection.json', {
            'seed': args.seed, 'best_epoch': best_ep, 'best_validation_macroF1_eligible': best_macro,
            'train_seconds': train_seconds, 'n_parameters': sum(x.numel() for x in net.parameters()),
            'graph_edges': 'i<j (45 directed edges for W=10) + PyG self-loops' if edges=='upper_triangular' else 'self-loops only',
            'smote_used': False, 'training_sampler_balanced': args.balance_sampler,
            'original_doubled_activations_x1_plus_x1_retained': True,
            'original_cosine_annealing_Tmax10_retained': True,
            'torch_version': torch.__version__,
            'checkpoint_file': f'{variant}_selected_state_dict.pt',
        })
        for p, loader in (('validation', val_loader), ('test', test_loader)):
            yy, pp = evaluate(loader)
            result = score(yy, pp, classes, eligible, p, variant, train_seconds)
            safe_write_json(output / f'{variant}_{p}.json', result)
            summaries.append({k: v for k, v in result.items() if k not in ('per_class','confusion_matrix')})
            pd.DataFrame({'true': [classes[i] for i in yy], 'pred': [classes[i] for i in pp]}).to_csv(
                output / f'{variant}_{p}_pred.csv', index=False)
            print(f'  SELECTED {variant} ep={best_ep} {p} macroF1_eligible={result["macro_F1_eligible_classes"]:.4f} '
                  f'all12={result["macro_F1_all_12_classes"]:.4f} acc={result["accuracy"]:.4f}')
        del datasets, net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(histories).to_csv(output / 'gat_training_history.csv', index=False)
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--mode', choices=['audit','baselines','gat','all'], default='audit')
    ap.add_argument('--window-size', type=int, default=10)
    ap.add_argument('--stride', type=int, default=10,
                    help='DEFAULT 10 nonoverlapping; use same as window size for uncorrelated windows')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--trees', type=int, default=150)
    ap.add_argument('--include-xgb', action='store_true')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=64,
                    help='64 for fast diagnostic; --hidden 256 --heads 4 matches original dimensions')
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--lr', type=float, default=.0003)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=.3)
    ap.add_argument('--balance-sampler', action='store_true', help='train-only real-window oversampling')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    if not args.csv.is_file():
        ap.error(f'Input file does not exist: {args.csv}')
    if args.window_size < 2 or args.stride < 1 or args.hidden < 4 or args.epochs < 1:
        ap.error('Invalid W/stride/hidden/epochs')
    if args.stride < args.window_size:
        print('WARNING: overlapping windows within each partition; default non-overlapping is safer.')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.out.expanduser() / f'{args.mode}_{timestamp}'
    output.mkdir(parents=True, exist_ok=False)
    parts, windows, diag, features, classes, eligible = inspect_clean_split(
        args.csv, output, args.window_size, args.stride)
    metadata = {
        'date_utc': timestamp, 'software_python': platform.python_version(),
        'command_arguments': vars(args), 'protocol': diag,
        'mode': args.mode, 'model_hyperparameters_fixed_before_current_run': True,
        'historical_runs_already_consulted': True,
        'not_preregistered_or_blind': True,
        'original_f2_smote_scripts_reused_as_architecture_reference_not_modified': True,
    }
    safe_write_json(output / 'full_run_metadata.json', metadata)
    if args.mode == 'audit':
        print('\nAUDIT COMPLETE. NO MODEL FIT. Output:', output)
        return
    X, preprocess = preprocessing(parts, features)
    safe_write_json(output / 'preprocessing_description.json', preprocess)
    results = []
    if args.mode in ('baselines','all'):
        results.extend(run_baselines(args, output, parts, windows, X, classes, eligible))
    if args.mode in ('gat','all'):
        # Import only when requested, so audit and baseline modes work without PyG.
        results.extend(run_gat(args, output, windows, X, classes, eligible))
    pd.DataFrame(results).to_csv(output / 'window_models_summary.csv', index=False)
    print('\nCOMPLETE: results:', output)
    print('Interpretation: label-BLIND windows over CLASS-SORTED CSV records; not temporal, device or IPS validation.')
    print('Rare classes absent from windows remain covered by record-level baselines if mode includes baselines.')

if __name__ == '__main__':
    main()
