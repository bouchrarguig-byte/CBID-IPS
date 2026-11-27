#!/usr/bin/env python3
"""CBID-IPS IJ-AI #31617: V7 post-hoc multi-source diagnostics and inference cost.

Uses ONLY pre-existing V7 validation predictions and saved checkpoints.
NO training. NO inspection or reading of April 24 data or labels.

Requires cbid_gatv2_validation_v6.py next to this file and existing V7 run.
Output is explicitly EXPLORATORY and must not be labelled a blinded test.

Examples:
  python cbid_v7_posthoc_multisource_latency.py --metrics-only
  python cbid_v7_posthoc_multisource_latency.py --seed 11 --weights balanced
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import cbid_gatv2_validation_v6 as v6


def locate_v7(root: Path) -> Path:
    matches = [p for p in root.glob('development_v7_*') if p.is_dir() and (p/'configuration.json').exists()]
    if not matches:
        raise SystemExit(f'No V7 run found under: {root}')
    return sorted(matches)[-1]


def score_subset(y, p, *, threshold=0.5):
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, average_precision_score
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not len(y):
        return {'n': 0, 'positive': 0}
    predicted = (p >= threshold).astype(int)
    cm = confusion_matrix(y, predicted, labels=[0,1])
    tn, fp, fn, tp = cm.ravel().tolist()
    return {
        'n': int(len(y)), 'positive': int(y.sum()),
        'macroF1_at_05': float(f1_score(y, predicted, labels=[0,1], average='macro', zero_division=0)),
        'scan_recall_at_05': float(recall_score(y, predicted, zero_division=0)),
        'scan_precision_at_05': float(precision_score(y, predicted, zero_division=0)),
        'false_positive_rate_at_05': float(fp/max(fp+tn,1)),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'average_precision': float(average_precision_score(y,p)) if len(set(y)) == 2 else None,
    }


def build_model(torch, args):
    from torch import nn
    from torch.nn import functional as F
    from torch_geometric.nn import GATv2Conv
    hidden = int(args.get('hidden',32))
    heads = int(args.get('heads',2))
    dropout = float(args.get('dropout',0.2))
    n_features = len(v6.FEATURES)+2
    class ResidualGAT(nn.Module):
        def __init__(self):
            super().__init__()
            self.local = nn.Sequential(nn.Linear(len(v6.FEATURES), hidden), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, 2))
            self.g1 = GATv2Conv(n_features, hidden, heads=heads, concat=True, dropout=dropout)
            self.g2 = GATv2Conv(hidden*heads, hidden, heads=heads, concat=False, dropout=dropout)
            self.graph = nn.Sequential(nn.Linear(hidden,hidden), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(hidden,2))
            self.gate_logit = nn.Parameter(torch.tensor(-1.5,dtype=torch.float32))
        def forward(self, x, edge_index):
            local_logits = self.local(x[:, :len(v6.FEATURES)])
            g = F.elu(self.g1(x,edge_index))
            g = F.elu(self.g2(g,edge_index))
            return local_logits + torch.sigmoid(self.gate_logit)*self.graph(g)
    return ResidualGAT()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root',type=Path,default=Path.home()/'Downloads'/'CBID_NFToN_Development_v7')
    parser.add_argument('--run',type=Path,default=None,help='Exact V7 run folder, otherwise latest')
    parser.add_argument('--metrics-only',action='store_true',help='Skip checkpoint loading and CPU/GPU latency')
    parser.add_argument('--weights',choices=['none','balanced'],default='balanced')
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--max-batches',type=int,default=15)
    parser.add_argument('--repeats',type=int,default=4)
    args = parser.parse_args()
    run = args.run if args.run else locate_v7(args.root)
    cfg_path = run/'configuration.json'
    if not cfg_path.is_file():
        raise SystemExit(f'Missing V7 configuration: {cfg_path}')
    config = json.loads(cfg_path.read_text())
    if not str(config.get('status','')).startswith('EXPLORATORY'):
        raise SystemExit('Unexpected V7 configuration status')
    data_path=Path(config['data'])
    train_t=v6.utc_ms(config['train_before_utc'])
    holdout_t=v6.utc_ms(config['validation_before_utc'])
    embargo=int(config['args'].get('embargo_seconds',120))*1000
    parts=v6.read_train_validation(data_path,train_t,holdout_t,embargo)
    val=parts['validation']
    yy=val[v6.LABEL].to_numpy(dtype=int)
    # Group multiplicity based on the true labels; used solely for *post-hoc reporting*.
    kk=val.groupby('_graph',sort=False)[v6.LABEL].transform('sum').to_numpy(dtype=int)
    group_n=val.groupby('_graph',sort=False)[v6.LABEL].transform('size').to_numpy(dtype=int)
    groups={
        'k0_no_scanning_source': kk==0,
        'k1_one_scanning_source': kk==1,
        'k2plus_multisource': kk>=2,
        'k4plus_multisource': kk>=4,
        'all_validation': np.ones(len(yy),dtype=bool),
    }
    print('V7 post-hoc analysis; validation April 23 only; 24-Apr NOT read.')
    print('V7 run:',run)
    print('Validation rows:',len(val),'positive:',int(yy.sum()),'graphs:',val['_graph'].nunique())
    print('k>=2 graphs:',int(val.groupby('_graph')[v6.LABEL].sum().ge(2).sum()))
    allrows=[]
    pdir=run/'predictions'
    for file in sorted(pdir.glob('*.csv')):
        tag=file.stem
        pred=pd.read_csv(file)
        expected={'y_true','probability','validation_row_index'}
        if not expected.issubset(pred.columns):
            raise ValueError(f'Missing prediction columns: {file}')
        pred=pred.sort_values('validation_row_index')
        ix=pred.validation_row_index.to_numpy(dtype=int)
        if not np.array_equal(ix,np.arange(len(yy))):
            raise ValueError(f'Prediction indices do not align: {file}')
        if not np.array_equal(pred.y_true.to_numpy(dtype=int),yy):
            raise ValueError(f'Ground-truth mismatch: {file}')
        pp=pred.probability.to_numpy(dtype=float)
        if not np.isfinite(pp).all():
            raise ValueError(f'Non-finite probability in {file}')
        for group_name, mask in groups.items():
            result=score_subset(yy[mask],pp[mask]); result.update({'model':tag,'subset':group_name})
            allrows.append(result)
    if not allrows:
        raise SystemExit('No V7 predictions found in '+str(pdir))
    out=run/'posthoc_v7_multisource_latency'
    out.mkdir(exist_ok=True)
    df=pd.DataFrame(allrows)
    df.to_csv(out/'subgroup_metrics_all_seeds.csv',index=False)
    dff=df[(df.subset=='k2plus_multisource') & df.model.str.endswith(('seed11','seed22','seed33','seed44','seed55'))]
    print('\nMULTI-SOURCE k>=2 subset, fixed 0.5 threshold (descriptive only):')
    print(dff[['model','n','positive','scan_recall_at_05','scan_precision_at_05','average_precision']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nSaved subgroup metrics:',out/'subgroup_metrics_all_seeds.csv')
    if args.metrics_only:
        print('METRICS-ONLY complete; no model trained, no checkpoints loaded.')
        return
    if args.batch_size < 1 or args.max_batches < 1 or args.repeats < 1:
        parser.error('batch-size, max-batches and repeats must be positive')
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    torch.set_num_threads(min(8,torch.get_num_threads()))
    local,_,_=v6.preprocess(parts)
    lat=[]
    for edge_name,with_edges in [('no_edges',False),('star_edges',True)]:
        tag=f'{edge_name}_{args.weights}_seed{args.seed}'
        ck=run/'checkpoints'/f'{tag}.pt'
        if not ck.is_file():
            raise SystemExit('Missing model checkpoint: '+str(ck))
        saved=torch.load(ck,map_location='cpu',weights_only=True)
        state=saved['state_dict']
        graph=v6.make_graphs(parts,local,'validation',torch,Data,with_edges)
        loader=DataLoader(graph,batch_size=args.batch_size,shuffle=False)
        for dev in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            model=build_model(torch,config['args']).to(torch.device(dev))
            model.load_state_dict(state,strict=True)
            stat=v6.latency_graph(model,loader,torch.device(dev),torch,args.repeats,args.max_batches)
            stat.update({'model':tag,'timestamp_split':'23-Apr-2019 validation',
                         'timing_note':'Preconstructed graphs; excludes parsing, NetFlow extraction and feature aggregation.'})
            lat.append(stat)
            print(f'LATENCY {tag:32s} {dev:4s}: {stat["milliseconds_per_source_window"]:.4f} ms/source-window; '
                  f'{stat["milliseconds_per_graph"]:.4f} ms/graph; n_graphs={stat["graphs_timed"]}')
        del graph
    pd.DataFrame(lat).to_csv(out/'inference_latency_v7.csv',index=False)
    readme=(
        'EXPLORATORY, VALIDATION ONLY. The April 24 day-held-out test was already inspected during V5 '
        'and is not read by this script. Subgroup k is computed from ground truth ONLY for post-hoc analysis; '
        'do not use as input features. Training and model tuning are NOT performed. Seed variation is not '
        'independent-network uncertainty. Source-window latency is based on pre-computed PyG graphs, '
        'batching, transfer and forward pass, not full online detection latency or deployment hardware.\n'
    )
    (out/'README_LIMITATIONS.txt').write_text(readme,encoding='utf-8')
    print('Saved latency:',out/'inference_latency_v7.csv')
    print('COMPLETE: no model trained; 24-Apr holdout not read.')

if __name__=='__main__':
    main()
