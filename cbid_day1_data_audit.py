#!/usr/bin/env python3
"""Read-only CBID-IPS dataset preflight. Does NOT train, edit, or save source data."""
import argparse
from pathlib import Path
import hashlib
import sys

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}. Activate your existing CBID environment first.")

p = argparse.ArgumentParser(description="CBID-IPS read-only preflight for IJ-AI #31617")
p.add_argument('dataset', nargs='?', default=str(Path.home() / 'Downloads/RT-IoT2022/rt_iot2022_prepared.csv'))
a = p.parse_args()
path = Path(a.dataset).expanduser()
if not path.is_file():
    sys.exit(f"Dataset not found: {path}\nPass its actual path: python cbid_day1_data_audit.py /path/to/dataset.csv")

print('=== DATASET FILE (READ ONLY) ===')
print('Path:', path)
print('Bytes:', path.stat().st_size)
h = hashlib.sha256()
with path.open('rb') as f:
    for chunk in iter(lambda: f.read(1024*1024), b''):
        h.update(chunk)
print('SHA256:', h.hexdigest())

df = pd.read_csv(path, low_memory=False)
print('\n=== SHAPE & COLUMNS ===')
print('Rows:', len(df), 'Columns:', len(df.columns))
print('Column names:', ', '.join(map(str, df.columns)))
print('Column types:\n', df.dtypes.astype(str).to_string())

print('\n=== DATA QUALITY ===')
print('Exact duplicated rows:', int(df.duplicated().sum()))
missing = df.isna().sum().sort_values(ascending=False)
print('Columns with missing values:\n', missing[missing>0].head(20).to_string() if (missing>0).any() else 'None')
if 'label' in df.columns:
    y = df['label'].fillna('<MISSING>').astype(str)
    print('\n=== CLASS COUNTS ===')
    print(y.value_counts(dropna=False).to_string())
    print('\n=== CLASS ORDER / WINDOW VALIDITY ===')
    boundaries = int((y != y.shift()).sum())
    print('Consecutive label runs:', boundaries)
    print('First 30 labels:', y.head(30).to_list())
    print('Last 30 labels:', y.tail(30).to_list())
    print('Note: low run count may mean class-sorted data; consecutive rows may not be chronological or from distinct devices.')
    if len(y) >= 10:
        W, S = 10, 5
        n = (len(y)-W)//S + 1
        print('Windows with W=10, S=5:', n)
        print('Shared source rows between neighboring windows:', W-S)
        print('Random window-level train/test split can reuse source rows across sets.')
else:
    print('WARNING: expected label column not found.')

print('\n=== CANDIDATE TIME / DEVICE / SESSION COLUMNS ===')
candidates = [c for c in df.columns if any(t in c.lower() for t in ('time', 'stamp', 'date', 'device', 'host', 'session', 'capture', 'src_ip', 'dst_ip', 'flow_id', 'id'))]
print(candidates if candidates else 'No obvious time/device/session columns; do not claim temporal or device graphs without separate metadata.')
for c in candidates[:12]:
    if c in df.columns and c != 'label':
        print(f'{c}: unique={df[c].nunique(dropna=True)}, sample={df[c].dropna().head(3).tolist()}')

print('\n=== DATA LEAKAGE ALERTS TO VERIFY IN CODE ===')
print('1. Fit OneHotEncoder/StandardScaler only on training partition.')
print('2. Split original rows/groups/ordered intervals BEFORE making overlapping windows.')
print('3. Keep SMOTE strictly inside training partition (already true in archived smote.py).')
print('4. If no real timestamps/devices, describe window nodes as records, NOT individual IoT devices.')
print('5. Create a separate validation partition for early stopping and tuning.')
print('\nPRECHECK COMPLETE. No files modified.')
