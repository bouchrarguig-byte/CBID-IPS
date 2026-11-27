#!/usr/bin/env python3
"""CBID-IPS / IJ-AI #31617: RT-IoT2022 integrity audit and RECORD-LEVEL baselines.

Important: this is NOT a temporal, device-level, or inter-host graph experiment.
It intentionally does NOT reproduce the original overlapping-window evaluation.
It does not modify either input CSV or any historical experiment.

Examples:
  python cbid_rt_iot2022_clean_protocol_v1.py --audit-only
  python cbid_rt_iot2022_clean_protocol_v1.py --train-baselines
  python cbid_rt_iot2022_clean_protocol_v1.py --train-baselines --n-estimators 120
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

DEFAULT_PREPARED = Path.home() / 'Downloads/telechargement 1/telech/RT-IoT2022/rt_iot2022_prepared.csv'
DEFAULT_RAW = Path.home() / 'Downloads/telechargement 1/telech/RT-IoT2022/RT_IOT2022.csv'
DEFAULT_OUT = Path.home() / 'Downloads/CBID_RT2022_Clean_Baselines_v1'


def sha256_file(path):
    hash_ = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hash_.update(chunk)
    return hash_.hexdigest()


def get_label_column(df):
    found = [name for name in ('label', 'Attack_type', 'Label') if name in df]
    if len(found) != 1:
        raise ValueError(f'Expected one target column; got {found}')
    return found[0]


def info_and_clean(raw_df, source_name):
    df = raw_df.copy()
    label_col = get_label_column(df)
    if df[label_col].isna().any():
        raise ValueError('Missing targets found. Must resolve before training.')
    df[label_col] = df[label_col].astype(str)
    if df.columns.duplicated().any():
        raise ValueError('Duplicated column names are not supported.')

    # 'no' is a potential row ID / label proxy and should not be used as a feature.
    exclude = {label_col, 'no'}
    feature_columns = [c for c in df.columns if c not in exclude]
    feature_frame = df[feature_columns]
    fingerprint = pd.util.hash_pandas_object(feature_frame, index=False)
    label_counts = df[label_col].value_counts()
    unique_keys = fingerprint.nunique()
    rows_with_duplicate_feats = int(fingerprint.duplicated(keep=False).sum())
    removable_duplicates = int(len(df) - unique_keys)
    groups = pd.DataFrame({'key': fingerprint.to_numpy(), 'label': df[label_col].to_numpy()})
    label_variety = groups.groupby('key', sort=False)['label'].nunique()
    conflict_keys = label_variety.index[label_variety > 1]
    conflict_mask = fingerprint.isin(conflict_keys)

    transitions = int(df[label_col].ne(df[label_col].shift()).sum())
    print(f'\n=== {source_name} ===')
    print('Rows:', len(df), '| columns:', len(df.columns), '| target:', label_col)
    print('Label runs:', transitions, '| classes:', len(label_counts))
    print('Class counts:\n' + label_counts.to_string())
    print('Exact duplicates (including no):', int(df.duplicated().sum()))
    print('Duplicate *features* (excluding label and no):', removable_duplicates)
    print('Rows belonging to repeated-feature groups:', rows_with_duplicate_feats)
    print('Feature groups with conflicting labels:', len(conflict_keys))
    print('Rows in conflicting-label groups:', int(conflict_mask.sum()))
    print('Unique feature fingerprints:', unique_keys)
    if 'no' in df:
        print("Column 'no' distinct:", df['no'].nunique(),
              '| unique fraction:', round(df['no'].nunique() / len(df), 6))
        print("Column 'no' min/max:", df['no'].min(), df['no'].max())
    print('Input-feature columns without ID:', len(feature_columns))
    print('Feature types:', feature_frame.dtypes.value_counts().to_dict())
    print('NOTE: fingerprint is a 64-bit feature hash; collisions are theoretically possible.')

    return {
        'df': df, 'label_col': label_col, 'feature_columns': feature_columns,
        'fingerprint': fingerprint, 'conflict_mask': conflict_mask,
        'audit': {
            'source': source_name,
            'rows': len(df), 'columns': len(df.columns),
            'target': label_col,
            'class_counts': label_counts.to_dict(), 'class_runs': transitions,
            'exact_duplicates_including_no': int(df.duplicated().sum()),
            'duplicate_feature_rows_after_first': removable_duplicates,
            'rows_in_repeated_feature_groups': rows_with_duplicate_feats,
            'conflicting_feature_groups': int(len(conflict_keys)),
            'rows_in_conflicting_feature_groups': int(conflict_mask.sum()),
            'feature_count_without_no': len(feature_columns),
            'unique_feature_fingerprints': int(unique_keys),
        }
    }


def raw_comparison(prepared, raw):
    if prepared['feature_columns'] != raw['feature_columns']:
        print('WARNING: Raw/prepared feature column names/order differ.')
        print('Prepared-only:', sorted(set(prepared['feature_columns']) - set(raw['feature_columns'])))
        print('Raw-only:', sorted(set(raw['feature_columns']) - set(prepared['feature_columns'])))
        return
    a, b = prepared['df'], raw['df']
    if len(a) != len(b):
        print('WARNING: raw/prepared row count differs:', len(a), len(b))
        return
    f = prepared['feature_columns']
    feature_identical = a[f].equals(b[f])
    labels_identical = a[prepared['label_col']].equals(b[raw['label_col']])
    ids_identical = ('no' in a and 'no' in b and a['no'].equals(b['no']))
    print('\n=== RAW / PREPARED ROW-BY-ROW COMPARISON ===')
    print('Same values in all features excluding no:', feature_identical)
    print('Same target labels:', labels_identical)
    print("Same 'no' column:", ids_identical)
    if not feature_identical:
        mismatch = (a[f].ne(b[f]) & ~(a[f].isna() & b[f].isna())).any(axis=1)
        print('Rows with one or more differing feature values:', int(mismatch.sum()))
    print('NOTE: a mismatch does not automatically mean the data are invalid; check transforms.')


def make_splits(info, seed):
    from sklearn.model_selection import train_test_split

    df, label_col, features = info['df'], info['label_col'], info['feature_columns']
    # Remove ALL ambiguous feature vectors (identical features but different targets),
    # and retain one occurrence of each remaining feature vector. This ensures that
    # identical feature vectors cannot appear in multiple partitions.
    before = len(df)
    clean = df.loc[~info['conflict_mask']].copy()
    removed_conflicts = before - len(clean)
    clean = clean.drop_duplicates(subset=features, keep='first').copy()
    removed_same_label_dups = before - removed_conflicts - len(clean)
    print('\n=== DATA CLEANING FOR RECORD-LEVEL BASELINES ===')
    print('Removed due to conflicting labels for identical features:', removed_conflicts)
    print('Removed repeated same-label feature vectors:', removed_same_label_dups)
    print('Remaining unique feature vectors:', len(clean))
    print('Counts after cleaning:\n' + clean[label_col].value_counts().to_string())
    if len(clean) < 100 or clean[label_col].value_counts().min() < 10:
        raise ValueError('Not enough samples after cleaning for stratified 60/20/20 split.')

    y = clean[label_col]
    idx = clean.index.to_numpy()
    train_val_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.25, random_state=seed,
        stratify=clean.loc[train_val_idx, label_col]
    )

    # Safety check: no same feature fingerprint in two partitions.
    fp = info['fingerprint']
    tr, va, te = (set(fp.loc[p].tolist()) for p in (train_idx, val_idx, test_idx))
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    print('\n=== STRATIFIED RECORD-LEVEL PARTITIONS ===')
    for name, ix in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
        print(f'{name:11} n={len(ix):6d} |', clean.loc[ix, label_col].value_counts().to_dict())
    print('Feature fingerprint overlap between partitions: ZERO')
    print('LIMITATION: these are NOT time-disjoint or host-disjoint partitions.')
    return clean, train_idx, val_idx, test_idx, {
        'rows_clean': len(clean), 'removed_ambiguous': int(removed_conflicts),
        'removed_duplicates': int(removed_same_label_dups),
        'split_seed': seed, 'split': 'stratified_60_20_20_unique_feature_rows'
    }


def evaluate_model(model, x, y_true, classes, pred_path=None, orig_ids=None):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
        classification_report, confusion_matrix,
    )
    yp = model.predict(x)
    labels = list(range(len(classes)))
    scores = {
        'n': int(len(y_true)), 'accuracy': float(accuracy_score(y_true, yp)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, yp)),
        'macro_f1': float(f1_score(y_true, yp, average='macro', labels=labels, zero_division=0)),
        'weighted_f1': float(f1_score(y_true, yp, average='weighted', labels=labels, zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, yp)),
        'classification_report': classification_report(
            y_true, yp, labels=labels, target_names=classes, output_dict=True, zero_division=0
        ),
        'confusion_matrix': confusion_matrix(y_true, yp, labels=labels).tolist(),
    }
    if pred_path is not None:
        with pred_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['original_dataset_row_index', 'true_label', 'predicted_label'])
            for index, true, pred in zip(orig_ids, y_true, yp):
                writer.writerow([int(index), classes[int(true)], classes[int(pred)]])
    return scores


def train_baselines(info, args, data_hash):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.utils.class_weight import compute_sample_weight

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError('xgboost unavailable. Run: "$HOME/cbid-env/bin/python" -m pip install xgboost') from exc

    clean, train_idx, val_idx, test_idx, cleaning = make_splits(info, args.seed)
    feature_columns = info['feature_columns']
    label_col = info['label_col']
    numeric = clean[feature_columns].select_dtypes(include='number').columns.tolist()
    categorical = [c for c in feature_columns if c not in numeric]

    print('\nNumeric features:', len(numeric), '| categorical features:', categorical)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric),
            ('cat', Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), categorical),
        ], remainder='drop', sparse_threshold=0.0,
    )

    def clean_infinite_values(df):
        x = df[feature_columns].copy()
        if numeric:
            x[numeric] = x[numeric].replace([np.inf, -np.inf], np.nan)
        return x

    print('Fitting feature preprocessing ONLY ON TRAIN...')
    xtr = preprocessor.fit_transform(clean_infinite_values(clean.loc[train_idx])).astype('float32')
    xva = preprocessor.transform(clean_infinite_values(clean.loc[val_idx])).astype('float32')
    xte = preprocessor.transform(clean_infinite_values(clean.loc[test_idx])).astype('float32')
    y_encoder = LabelEncoder()
    y_encoder.fit(clean[label_col])  # target names only; feature transforms never see validation/test.
    classes = y_encoder.classes_.tolist()
    ytr = y_encoder.transform(clean.loc[train_idx, label_col])
    yva = y_encoder.transform(clean.loc[val_idx, label_col])
    yte = y_encoder.transform(clean.loc[test_idx, label_col])
    print('Encoded feature dimension:', xtr.shape[1], '| classes:', len(classes))

    weights = compute_sample_weight(class_weight='balanced', y=ytr) if args.class_weights else None
    print('Class weighting:', 'balanced, on training only' if args.class_weights else 'disabled')
    model_specs = {
        'RF_record': RandomForestClassifier(
            n_estimators=args.n_estimators, max_features='sqrt',
            min_samples_leaf=1, random_state=args.seed, n_jobs=-1,
        ),
        'XGBoost_record': XGBClassifier(
            n_estimators=args.n_estimators, max_depth=6, learning_rate=0.07,
            subsample=0.85, colsample_bytree=0.9,
            objective='multi:softprob', eval_metric='mlogloss',
            tree_method='hist', n_jobs=8, random_state=args.seed,
        ),
    }
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = Path(args.output).expanduser() / f'baseline_{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=False)

    # Original indices can be used to exactly reconstruct the same partitions for GNNs later.
    with (out_dir / 'split_indices.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['original_dataset_row_index', 'partition', 'label'])
        for subset, indices in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
            for ix in indices:
                writer.writerow([int(ix), subset, clean.at[ix, label_col]])

    metadata = {
        'experiment': 'RECORD_LEVEL_BASELINES_ONLY_NOT_GNN_VALIDATION',
        'dataset': str(args.prepared), 'sha256': data_hash,
        'seed': args.seed, 'classes': classes, 'feature_columns': feature_columns,
        'numeric_columns': numeric, 'categorical_columns': categorical,
        'encoded_features': int(xtr.shape[1]), 'class_weights': args.class_weights,
        'n_estimators': args.n_estimators, 'cleaning': cleaning,
        'leakage_controls': [
            "dropped column no", "removed feature-identical duplicates before partitioning",
            "removed ambiguous same-feature vectors", "stratified record split, no shared feature vectors",
            "train-only imputation and one-hot encoding", "no SMOTE", "fixed hyperparameters",
        ],
        'limitations': [
            "Rows are not temporally or host disjoint.",
            "The rows in RT-IoT2022 are class-sorted, NOT temporal windows.",
            "No inter-device graph, cognitive adaptation or prevention is evaluated here.",
            "Dataset-wide deduplication is an explicit cleaning operation, reported by class.",
            "This is one split/seed. No confidence intervals or significance claims.",
        ],
    }
    (out_dir / 'experiment_metadata.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    summary = {}
    for model_name, model in model_specs.items():
        print(f'\nTraining {model_name}... (hyperparameters fixed, test not used for tuning)')
        start = time.perf_counter()
        if weights is None:
            model.fit(xtr, ytr)
        else:
            model.fit(xtr, ytr, sample_weight=weights)
        train_seconds = round(time.perf_counter() - start, 3)
        val_scores = evaluate_model(model, xva, yva, classes)
        test_scores = evaluate_model(
            model, xte, yte, classes, pred_path=out_dir / f'{model_name}_test_predictions.csv',
            orig_ids=test_idx,
        )
        summary[model_name] = {
            'train_seconds': train_seconds, 'validation': val_scores, 'test': test_scores,
        }
        print(f'  validation: macroF1={val_scores["macro_f1"]:.4f} '
              f'acc={val_scores["accuracy"]:.4f} MCC={val_scores["mcc"]:.4f}')
        print(f'  TEST      : macroF1={test_scores["macro_f1"]:.4f} '
              f'acc={test_scores["accuracy"]:.4f} MCC={test_scores["mcc"]:.4f}')
        print('  Rare-class supports and F1 on TEST:')
        for name in sorted(classes, key=lambda c: test_scores['classification_report'][c]['support'])[:5]:
            report = test_scores['classification_report'][name]
            print(f'  {name:30} n={int(report["support"]):4d} F1={report["f1-score"]:.4f}')
        (out_dir / f'{model_name}_metrics.json').write_text(
            json.dumps(summary[model_name], indent=2, ensure_ascii=False), encoding='utf-8'
        )
    (out_dir / 'results_summary.json').write_text(
        json.dumps({
            name: {
                'train_seconds': val['train_seconds'],
                'validation': {m: val['validation'][m] for m in ('accuracy','macro_f1','weighted_f1','mcc')},
                'test': {m: val['test'][m] for m in ('accuracy','macro_f1','weighted_f1','mcc')},
            } for name, val in summary.items()
        }, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print('\nResults saved:', out_dir)
    print('NOTE: performance is a record-level baseline only, NOT evidence for inter-device GATv2.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepared', type=Path, default=DEFAULT_PREPARED)
    parser.add_argument('--raw', type=Path, default=DEFAULT_RAW)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--audit-only', action='store_true', help='Default; read-only data audit.')
    group.add_argument('--train-baselines', action='store_true', help='Train RF and XGBoost record-level baselines.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--class-weights', action='store_true',
                        help='Balanced sample weights for BOTH RF and XGBoost; training only.')
    args = parser.parse_args()
    if not args.prepared.is_file():
        parser.error(f'Prepared CSV missing: {args.prepared}')
    if args.n_estimators < 1:
        parser.error('--n-estimators must be >=1')

    print('=== IJ-AI 31617 — RT-IoT2022 RECORD-LEVEL integrity protocol ===')
    print('Source CSVs are read only. No original projects or model files are modified.')
    print('Prepared SHA256:', sha256_file(args.prepared))
    prepared_info = info_and_clean(pd.read_csv(args.prepared, low_memory=False), 'PREPARED')
    if args.raw.is_file():
        print('Raw SHA256:', sha256_file(args.raw))
        raw_info = info_and_clean(pd.read_csv(args.raw, low_memory=False), 'RAW')
        raw_comparison(prepared_info, raw_info)
    else:
        print('Raw CSV not found, skipping raw comparison:', args.raw)

    if args.train_baselines:
        train_baselines(prepared_info, args, sha256_file(args.prepared))
    else:
        print('\nAUDIT COMPLETE; no training performed, no files created.')
        print('To train RF/XGB on UNIQUE flow records: rerun with --train-baselines')
        print('Do not compare these record-level results directly with the archived overlapping-window scores.')


if __name__ == '__main__':
    main()
