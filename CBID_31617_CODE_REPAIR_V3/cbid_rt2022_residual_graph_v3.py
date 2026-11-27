#!/usr/bin/env python3
"""IJ-AI #31617 — V3 experimental *flow-window* graph and residual ablations.

IMPORTANT: This experiment does not claim that the RT-IoT2022 rows describe
IoT devices, genuine temporal sequences or an intrusion prevention mechanism.
It does NOT replace the original published f2.py results. It tests the
hypothesis that a label-blind similarity graph benefits WINDOW classification.

Reuses the audited split, exact-feature dedup, and train-only preprocessing
from cbid_rt2022_controlled_window_v2_1.py. Keep both in the same directory.

Graph controls with identical networks and 10-row bags:
  selfloops       : only each node's own information (no between-row messages)
  uppertriangular : original 45 directed i<j connections (constant weight 1)
  behavioral_knn  : cosine similarities of input features, label-blind, k arcs/node
  shuffled_knn    : same node count/arc count/edge-weight multiset, random receivers

The new architecture has TRUE skip projections (not z+z), uses edge_attr,
learns a graph-level classifier and selects checkpoints on *validation only*.
Random neighbor controls are deterministic and never use targets.

This is a development study on a historically explored dataset, NOT blind
external evidence for coordinated attack detection or real-world prevention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import cbid_rt2022_controlled_window_v2_1 as legacy

GRAPH_VARIANTS = ('selfloops','uppertriangular','behavioral_knn','shuffled_knn')


def safe_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def prepare_selection(output: Path, classes: list[str], min_val_windows: int):
    """Select labels from TRAIN and VAL supports, never from TEST supports."""
    df = pd.read_csv(output / 'per_class_split_and_window_support.csv')
    if df['class'].tolist() != classes:
        raise AssertionError('Class order mismatch')
    eligible = [j for j, row in df.iterrows()
                if int(row.train_windows) >= 1 and int(row.validation_windows) >= min_val_windows]
    if not eligible:
        raise RuntimeError('No eligible validation classes: lower --min-val-windows')
    safe_json(output / 'validation_selection_labels.json', {
        'min_val_windows': min_val_windows,
        'based_on_train_validation_only': True,
        'selected_labels': [classes[i] for i in eligible],
        'excluded_labels': [classes[i] for i in range(len(classes)) if i not in eligible],
        'method_devised_after_earlier_test_exploration': True,
        'note': 'All 12 labels are still evaluated in test; missing classes are reported as missing.'
    })
    return eligible


def make_window_edges(x_window: np.ndarray, variant: str, k: int, seed: int, window_id: int, split: str):
    """Return edge_index (2,E), edge weights (E,1), diagnostics; no labels used.

    All comparisons use exactly the same feature vectors, same graph size and
    architecture. For kNN, directed arcs have source=query, destination=neighbor
    (GAT messages propagate in that direction). For shuffled control, each query
    receives k independent random non-self neighbors with the same weights.
    """
    if x_window.ndim != 2:
        raise ValueError('Expected two-dimensional window features')
    n = x_window.shape[0]
    if n < 2: raise ValueError('Window must contain at least 2 records')
    if variant not in GRAPH_VARIANTS: raise ValueError(variant)
    if not 1 <= k < n: raise ValueError('Require 1 <= k < window size')
    if variant == 'selfloops':
        return np.empty((2,0), dtype=np.int64), np.empty((0,1), dtype=np.float32), {'n_arcs':0,'mean_weight':None}
    if variant == 'uppertriangular':
        pairs = np.array([(i,j) for i in range(n) for j in range(i+1,n)], dtype=np.int64)
        return pairs.T.copy(), np.ones((len(pairs),1),dtype=np.float32), {'n_arcs':len(pairs),'mean_weight':1.0}

    # L2-normalized cosine: no direct class-label or filename positional signal.
    a = np.nan_to_num(np.asarray(x_window, dtype=np.float64),nan=0.,posinf=0.,neginf=0.)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    a = a / np.maximum(norms, 1e-12)
    cosine = np.clip(a @ a.T,-1.,1.)
    np.fill_diagonal(cosine, -np.inf)
    # Stable sorting ensures reproducible tie breaking by node index.
    ranking = np.argsort(-cosine, axis=1, kind='stable')[:, :k]
    weights = np.clip((np.take_along_axis(cosine, ranking, axis=1) + 1.)/2., 0.,1.)
    if variant == 'shuffled_knn':
        # Randomize receiver topology while matching per-node outdegree and the
        # exact multiset of pairwise cosine weights for this window.
        split_salt = {'train':1,'validation':2,'test':3}.get(split)
        if split_salt is None: raise ValueError(split)
        serial = f'{seed}|{window_id}|{split_salt}|{n}|{k}'.encode()
        shuf_seed = int.from_bytes(hashlib.sha256(serial).digest()[:8], 'little')
        rng = np.random.default_rng(shuf_seed)
        ranking = np.stack([rng.choice(np.arange(n)[np.arange(n)!=i], size=k, replace=False)
                            for i in range(n)], axis=0)
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = ranking.reshape(-1).astype(np.int64)
    edge_index = np.stack([src,dst])
    edge_weight = weights.reshape(-1,1).astype(np.float32)
    assert not (src == dst).any()
    assert np.isfinite(edge_weight).all()
    return edge_index, edge_weight, {'n_arcs':int(len(src)), 'mean_weight':float(edge_weight.mean())}


def graph_diagnostics(X: dict, windows: dict, seed: int, k: int, output: Path,
                      variants: list[str], include_test=False):
    """No PyG dependency; examine edge topology before fitting anything."""
    rows=[]
    for partition in (('train','validation','test') if include_test else ('train','validation')):
        ww=windows[partition][0]
        for variant in variants:
            counts=[]; weights=[]
            # Diagnostics on up to 1000 label-blind, deterministic sample windows.
            subset=np.linspace(0,max(0,len(ww)-1),min(len(ww),1000),dtype=int)
            for wid in subset:
                e,w,d = make_window_edges(X[partition][ww[wid]],variant,k,seed,int(wid),partition)
                counts.append(int(e.shape[1])); weights.extend(w.reshape(-1).tolist())
            rows.append({'partition':partition, 'variant':variant,
                         'sampled_windows':int(len(subset)),
                         'arcs_per_window_mean':float(np.mean(counts)) if counts else 0,
                         'edge_weight_mean':float(np.mean(weights)) if weights else None})
    pd.DataFrame(rows).to_csv(output/'topology_diagnostics.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return rows


def run_gat(args, output, windows, X, classes, eligible):
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
        from torch.utils.data import WeightedRandomSampler
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GATv2Conv, global_mean_pool
    except ImportError as exc:
        raise SystemExit('PyTorch Geometric is required for training: ' + str(exc)) from exc

    if args.deterministic:
        # If the installed scatter kernels do not support deterministic CUDA,
        # fail loudly; never silently suggest that GPU outputs are identical.
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.benchmark=False
    device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print('New architecture: TRUE RESIDUAL GATv2; device=',device)
    print('Pytorch:',torch.__version__,'PyG:',__import__('torch_geometric').__version__)
    safe_json(output/'runtime_versions.json',{'torch':torch.__version__,
              'torch_geometric':__import__('torch_geometric').__version__,
              'cuda':str(torch.version.cuda), 'device':str(device),
              'deterministic_algorithms':bool(args.deterministic)})

    class ResidualNet(nn.Module):
        def __init__(self, n_features, n_classes):
            super().__init__()
            h=args.hidden
            heads=args.heads
            htotal=h*heads
            self.g1=GATv2Conv(n_features,h,heads=heads,concat=True,
                              edge_dim=1,dropout=args.dropout,fill_value=0.)
            self.skip1=nn.Linear(n_features,htotal,bias=False)
            self.norm1=nn.LayerNorm(htotal)
            self.g2=GATv2Conv(htotal,h,heads=heads,concat=True,
                              edge_dim=1,dropout=args.dropout,fill_value=0.)
            self.norm2=nn.LayerNorm(htotal)
            self.g3=GATv2Conv(htotal,h,heads=1,edge_dim=1,
                              dropout=args.dropout,fill_value=0.)
            self.skip3=nn.Linear(htotal,h,bias=False)
            self.norm3=nn.LayerNorm(h)
            self.lin1=nn.Linear(h,max(4,h//2))
            self.lin2=nn.Linear(max(4,h//2),n_classes)

        def forward(self,batch):
            ei,ew=batch.edge_index,batch.edge_attr
            z1=F.leaky_relu(self.g1(batch.x,ei,ew))
            z=self.norm1(z1+self.skip1(batch.x))
            z2=F.leaky_relu(self.g2(z,ei,ew))
            z=self.norm2(z2+z)
            z3=F.leaky_relu(self.g3(z,ei,ew))
            z=self.norm3(z3+self.skip3(z))
            z=global_mean_pool(z,batch.batch)
            z=F.dropout(z,p=args.dropout,training=self.training)
            z=F.leaky_relu(self.lin1(z))
            z=F.dropout(z,p=args.dropout,training=self.training)
            return self.lin2(z)

    def build_data(partition, variant, seed):
        result=[]
        ids,labels,_=windows[partition]
        for j,(window_ids,label) in enumerate(zip(ids,labels)):
            xx=X[partition][window_ids].astype(np.float32,copy=True)
            edge,weights,_=make_window_edges(xx,variant,args.k,seed,j,partition)
            result.append(Data(x=torch.from_numpy(xx),
                edge_index=torch.from_numpy(edge),
                edge_attr=torch.from_numpy(weights),
                y=torch.tensor([int(label)],dtype=torch.long)))
        return result

    def forward_predict(net,loader):
        net.eval()
        correct=[];pred=[];confidence=[]
        with torch.inference_mode():
            for batch in loader:
                batch=batch.to(device)
                logits=net(batch)
                proba=torch.softmax(logits,dim=-1)
                correct.extend(batch.y.cpu().numpy().astype(int).tolist())
                pred.extend(proba.argmax(-1).cpu().numpy().astype(int).tolist())
                confidence.extend(proba.amax(-1).cpu().numpy().tolist())
        return np.asarray(correct,dtype=int),np.asarray(pred,dtype=int),np.asarray(confidence,dtype=float)

    histories=[];rows=[]
    for seed in args.seeds:
        for variant in args.variants:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
            data={p:build_data(p,variant,seed) for p in
                  (('train','validation','test') if args.eval_test else ('train','validation'))}
            net=ResidualNet(X['train'].shape[1],len(classes)).to(device)
            generator=torch.Generator().manual_seed(seed)
            if args.balance_sampler:
                labels=windows['train'][1]
                cnt=np.bincount(labels,minlength=len(classes))
                w=np.reciprocal(np.maximum(cnt[labels],1).astype(float))
                sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),
                          len(labels),replacement=True,generator=generator)
                train_loader=DataLoader(data['train'],batch_size=args.batch_size,sampler=sampler)
            else:
                train_loader=DataLoader(data['train'],batch_size=args.batch_size,
                        shuffle=True,generator=generator)
            val_loader=DataLoader(data['validation'],batch_size=args.batch_size,shuffle=False)
            if args.eval_test:
                test_loader=DataLoader(data['test'],batch_size=args.batch_size,shuffle=False)
            criterion=nn.CrossEntropyLoss()
            optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
            best=-np.inf;best_ep=None;best_state=None;stale=0
            started=time.perf_counter()
            for ep in range(1,args.epochs+1):
                net.train();total=0.;n=0
                for batch in train_loader:
                    batch=batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits=net(batch)
                    loss=criterion(logits,batch.y)
                    if not torch.isfinite(loss):raise RuntimeError('NaN/Inf loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(),2.)
                    optimizer.step()
                    total+=float(loss.detach().cpu())*int(batch.num_graphs)
                    n+=int(batch.num_graphs)
                scheduler.step()
                yy,pp,_=forward_predict(net,val_loader)
                from sklearn.metrics import f1_score
                f=float(f1_score(yy,pp,labels=eligible,average='macro',zero_division=0))
                histories.append({'seed':seed,'variant':variant,'epoch':ep,
                                  'train_loss':total/max(n,1),'val_macroF1_selection':f})
                print(f'{variant} seed={seed} ep={ep} loss={total/max(n,1):.5f} valF1={f:.4f}',flush=True)
                if f>best+1e-9:
                    best=f;best_ep=ep;stale=0
                    best_state={key:value.detach().cpu().clone() for key,value in net.state_dict().items()}
                else:
                    stale+=1
                if stale>=args.patience:break
            if best_state is None:raise RuntimeError('No checkpoint selected')
            net.load_state_dict(best_state)
            tag=f'residual_{variant}_seed{seed}'
            torch.save(best_state,output/f'{tag}_selected.pt')
            safe_json(output/f'{tag}_selection.json',{'seed':seed,'variant':variant,
                'selected_epoch':best_ep,'validation_selection_macroF1':best,
                'training_seconds':time.perf_counter()-started,
                'n_parameters':sum(p.numel() for p in net.parameters()),
                'checkpoints_selected_using_test':False})
            from sklearn.metrics import (accuracy_score,f1_score,matthews_corrcoef,
                 confusion_matrix,precision_recall_fscore_support)
            eval_loaders=[('validation',val_loader)]
            if args.eval_test: eval_loaders.append(('test',test_loader))
            for partition,loader in eval_loaders:
                y,yp,confidence=forward_predict(net,loader)
                metric=legacy.score(y,yp,classes,eligible,partition,tag)
                support=np.bincount(y,minlength=len(classes))
                present_labels=np.where(support>0)[0].tolist()
                metric['macro_F1_test_present_labels_descriptive_only']=float(
                    f1_score(y,yp,labels=present_labels,average='macro',zero_division=0))
                metric['present_classes']=[classes[i] for i in present_labels]
                metric['selection_labels']=[classes[i] for i in eligible]
                safe_json(output/f'{tag}_{partition}.json',metric)
                pd.DataFrame({'true':[classes[i] for i in y],
                              'pred':[classes[i] for i in yp],
                              'confidence':confidence}).to_csv(output/f'{tag}_{partition}_pred.csv',index=False)
                rows.append({'seed':seed,'variant':variant,'partition':partition,
                    'selected_epoch':best_ep,'n_windows':len(y),
                    'accuracy':metric['accuracy'],'macro_F1_all12':metric['macro_F1_all_12_classes'],
                    'macro_F1_selection':metric['macro_F1_eligible_classes'],
                    'macro_F1_present':metric['macro_F1_test_present_labels_descriptive_only'],
                    'weighted_F1':metric['weighted_F1'],'MCC':metric['MCC']})
                print('SELECTED',tag,partition,'acc',round(metric['accuracy'],4),
                    'F1_all12',round(metric['macro_F1_all_12_classes'],4),
                    'F1_selection',round(metric['macro_F1_eligible_classes'],4),flush=True)
            pd.DataFrame(rows).to_csv(output/'metrics_by_variant_seed.csv',index=False)
            pd.DataFrame(histories).to_csv(output/'training_history.csv',index=False)
            del net,data
            if device.type=='cuda':torch.cuda.empty_cache()
    tab=pd.DataFrame(rows)
    compared_partition='test' if args.eval_test else 'validation'
    compared=tab.loc[tab.partition==compared_partition]
    summary=compared.groupby('variant').agg(n_seeds=('seed','size'),
        macroF1_all12_mean=('macro_F1_all12','mean'),macroF1_all12_sd=('macro_F1_all12','std'),
        macroF1_selection_mean=('macro_F1_selection','mean'),macroF1_selection_sd=('macro_F1_selection','std'),
        accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std')).reset_index()
    summary.to_csv(output/f'{compared_partition}_summary.csv',index=False)
    print(f'\n=== DEVELOPMENT {compared_partition.upper()} SUMMARY, NOT EXTERNAL ===')
    print(summary.to_string(index=False))
    print('OUTPUT:',output)


def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--csv',type=Path,default=legacy.DEFAULT_CSV)
    p.add_argument('--out',type=Path,default=Path.home()/'Downloads'/'CBID_RT2022_Residual_Graph_v3')
    p.add_argument('--mode',choices=['audit','preflight','gat'],default='preflight')
    p.add_argument('--window-size',type=int,default=10)
    p.add_argument('--stride',type=int,default=10)
    p.add_argument('--seed',type=int,default=42,help='Seed for topology-only preflight')
    p.add_argument('--seeds',type=int,nargs='+',default=[42],help='Training seeds for comparison')
    p.add_argument('--variants',nargs='+',choices=GRAPH_VARIANTS,
                   default=['selfloops','behavioral_knn','shuffled_knn'])
    p.add_argument('--k',type=int,default=2)
    p.add_argument('--min-val-windows',type=int,default=5,
                   help='Checkpoint selection uses classes with this many validation windows')
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--epochs',type=int,default=18)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--dropout',type=float,default=0.3)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--balance-sampler',action='store_true')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--deterministic',action='store_true',help='Try PyTorch deterministic algorithms (may fail on CUDA scatter)')
    p.add_argument('--eval-test',action='store_true',help='Explicitly enable previously examined test evaluation; DEFAULT: validation only')
    args=p.parse_args()
    if not args.csv.is_file():p.error(f'Dataset missing: {args.csv}')
    if args.window_size<2 or args.stride<1 or args.stride<args.window_size:
        p.error('Use nonoverlapping windows with window-size>=2 and stride>=window-size')
    if args.k<1 or args.k>=args.window_size:p.error('k must be in [1, window-size-1]')
    if args.epochs<1 or args.patience<1 or args.lr<=0 or args.batch_size<1 or args.hidden<4:
        p.error('Invalid training hyperparameter')
    if args.min_val_windows<1:p.error('--min-val-windows must be >=1')
    if len(args.seeds)!=len(set(args.seeds)):p.error('Duplicate seeds')
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=args.out.expanduser()/f'{args.mode}_{timestamp}'
    output.mkdir(parents=True,exist_ok=False)
    parts,windows,audit,features,classes,_=legacy.inspect_clean_split(args.csv,output,args.window_size,args.stride)
    eligible=prepare_selection(output,classes,args.min_val_windows)
    safe_json(output/'run_configuration.json',{
        'utc':timestamp,'python':platform.python_version(), 'input_csv_sha256':legacy.digest(args.csv),
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'genuine_interdevice_graph':False,
        'previous_test_already_seen':True,
        'validation_selection_posthoc_after_previous_exploration':True,
        'architecture':'New true-residual 3-layer edge-aware GATv2, never call this the original unchanged architecture',
        'graph_mode':'behavioral_knn defined by per-window standardized-feature cosine, not source/destination communications',
        'edge_counterfactual':'shuffled_knn has same weights/degree as real behavioral_knn but random topology',
        'no_claims':['host collaboration','temporal prediction','IPS actions','new independent test','significant improvement without evidence']})
    if args.mode=='audit':
        print('AUDIT COMPLETE:',output,'NO TRAINING');return
    X,prep=legacy.preprocessing(parts,features)
    safe_json(output/'preprocessing_description.json',prep)
    graph_diagnostics(X,windows,args.seed,args.k,output,args.variants, include_test=args.eval_test)
    if args.mode=='preflight':
        print('PREFLIGHT COMPLETE:',output,'NO TRAINING');return
    run_gat(args,output,windows,X,classes,eligible)

if __name__=='__main__':
    main()
