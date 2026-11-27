#!/usr/bin/env python3
"""IJ-AI #31617: fixed DeepSets/MLP + mean aggregation control for RT-IoT2022.

Uses the *unmodified* v2.1 dataset-cleaning and train-only preprocessing code.
Compared to the previously trained five-seed GATv2 experiments, this is an
EXPLORATORY architecture control, not a new independent or blind evaluation.
Dataset windows are artificial sequences of class-sorted CSV records and do
not represent contemporaneous devices or temporal network events.

The test set was already examined in earlier studies. Common10 is a post-hoc
reporting set; validation checkpoint selection uses the original train/val
eligible11 classes. Never use test results to change this architecture.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cbid_rt2022_controlled_window_v2_1 import (
    DEFAULT_CSV, DEFAULT_OUT, inspect_clean_split, preprocessing, safe_write_json,
    score, digest,
)

SEEDS = (11, 22, 33, 42, 44)
EXPERIMENT_VARIANTS = ("DeepSets_mean", "GATv2_selfloops", "GATv2_uppertriangular",
                       "XGB_window_mean", "RF_window_mean", "XGB_vote10", "RF_vote10")


def get_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("PyTorch missing from selected Python environment. Activate ~/cbid-env.") from exc
    return torch, nn, F


def define_model(input_dim, n_classes, width, hidden, dropout):
    torch, nn, F = get_torch()

    class DeepSetsMean(nn.Module):
        """phi(x_i) then permutation-invariant mean, then rho."""
        def __init__(self):
            super().__init__()
            # Three per-record mappings, matching the three GATv2 stages in depth.
            # width=hidden*heads defaults to 1024; actual parameter counts logged.
            self.e1 = nn.Linear(input_dim, width)
            self.n1 = nn.LayerNorm(width)
            self.e2 = nn.Linear(width, width)
            self.n2 = nn.LayerNorm(width)
            self.e3 = nn.Linear(width, hidden)
            self.f1 = nn.Linear(hidden, hidden // 2)
            self.f2 = nn.Linear(hidden // 2, n_classes)
            self.drop = nn.Dropout(dropout)

        def forward(self, window):
            if window.ndim != 3:
                raise ValueError(f"Expected B x W x F, got {tuple(window.shape)}")
            z = F.leaky_relu(self.e1(window))
            z = self.n1(z + z)  # match doubled activations + norm used in GAT v2.1
            z = F.leaky_relu(self.e2(z))
            z = self.n2(z + z)
            z = F.leaky_relu(self.e3(z))
            # Mean pooling is invariant to record order; NO attention/edges used.
            pooled = z.mean(dim=1)
            return self.f2(self.drop(F.leaky_relu(self.f1(self.drop(pooled)))))
    return DeepSetsMean()


def smoke_test():
    torch, nn, F = get_torch()
    torch.manual_seed(43)
    net = define_model(93, 12, 1536, 256, .3)
    x = torch.randn(4, 10, 93)
    net.eval()
    with torch.no_grad():
        y1 = net(x)
        y2 = net(x[:, list(reversed(range(10))), :])
    if y1.shape != (4, 12) or not torch.allclose(y1, y2, atol=1e-5, rtol=1e-5):
        raise AssertionError("DeepSets permutation-invariance or dimension test failed")
    net.train()
    loss = nn.CrossEntropyLoss()(net(x), torch.tensor([0,1,2,3]))
    loss.backward()
    if not np.isfinite(float(loss.item())):
        raise AssertionError("Invalid loss")
    print("SMOKE TEST PASSED: dimensions, mean-pooling permutation invariance, gradients.")
    print("Parameter count:", sum(x.numel() for x in net.parameters()))


def find_existing_runs(root, mode, target_seeds):
    """Latest complete run per seed. No lookups outside local RT2022 output root."""
    candidates = {}
    for run in sorted(root.glob(f"{mode}_*")):
        meta = run / 'full_run_metadata.json'
        if not meta.exists():
            continue
        try:
            blob = json.loads(meta.read_text(encoding='utf-8'))
            seed = int(blob['command_arguments']['seed'])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if seed in target_seeds:
            candidates.setdefault(seed, []).append((run, blob))
    return candidates


def verify_existing_gat(root, source_sha, test_true, classes, desired_args):
    """Never compare to a different dataset, label order, or parameter protocol."""
    known = find_existing_runs(root, 'gat', SEEDS)
    found = {}
    for seed in SEEDS:
        for run, meta in reversed(known.get(seed, [])):
            path = run / 'GATv2_selfloops_test_pred.csv'
            if not path.exists() or meta.get('protocol', {}).get('input_sha256') != source_sha:
                continue
            old = meta['command_arguments']
            check = ['hidden','heads','window_size','stride','batch_size','epochs','patience','balance_sampler']
            bad = {k: (old.get(k), desired_args.get(k)) for k in check if old.get(k)!=desired_args.get(k)}
            if bad:
                print(f"SKIP incompatible GAT seed {seed} folder {run}: {bad}")
                continue
            if abs(float(old.get('lr', -1))-float(desired_args['lr'])) > 1e-12:
                continue
            data = pd.read_csv(path).astype(str)
            if list(data.columns) != ['true','pred'] or not np.array_equal(data['true'].to_numpy(), test_true):
                raise RuntimeError(f"Truth mismatch: GAT saved predictions, seed {seed}: {path}")
            found[seed] = run
            break
    print('Verified GAT seeds with matching source SHA256, run args and ordered ground truth:', sorted(found))
    return found


def train_seed(seed, args, torch, Xwindows, labels, classes, eligible, common10, output):
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from sklearn.metrics import f1_score

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    model = define_model(Xwindows['train'].shape[-1], len(classes), args.mlp_width,
                         args.hidden, args.dropout).to(device)
    count = int(sum(x.numel() for x in model.parameters()))
    datasets = {p: TensorDataset(torch.from_numpy(Xwindows[p]),
                                 torch.from_numpy(labels[p].astype(np.int64)))
                for p in ('train','validation','test')}
    if args.balance_sampler:
        class_count=np.bincount(labels['train'], minlength=len(classes))
        w=1.0 / np.maximum(class_count[labels['train']],1)
        sampler=WeightedRandomSampler(torch.tensor(w,dtype=torch.float64),len(w),replacement=True,generator=generator)
        train_loader=DataLoader(datasets['train'],batch_size=args.batch_size,sampler=sampler)
    else:
        train_loader=DataLoader(datasets['train'],batch_size=args.batch_size,shuffle=True,generator=generator)
    val_loader=DataLoader(datasets['validation'],batch_size=args.batch_size,shuffle=False)
    test_loader=DataLoader(datasets['test'],batch_size=args.batch_size,shuffle=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr, weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
    criterion=nn.CrossEntropyLoss()
    best_score=-1.0; best_state=None; best_epoch=-1; since=0
    history=[]; start=time.perf_counter()

    def evaluate(loader):
        model.eval(); truth=[]; pred=[]
        with torch.inference_mode():
            for feats, y in loader:
                out=model(feats.to(device))
                truth.extend(y.cpu().numpy().tolist())
                pred.extend(out.argmax(-1).cpu().numpy().tolist())
        return np.asarray(truth,dtype=np.int64),np.asarray(pred,dtype=np.int64)

    for epoch in range(1,args.epochs+1):
        model.train(); loss_total=0.; n_obs=0
        for bx, by in train_loader:
            bx=bx.to(device); by=by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(bx)
            loss=criterion(logits,by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.0)
            optimizer.step()
            loss_total+=float(loss.item())*int(len(bx));n_obs+=int(len(bx))
        schedule.step()
        yv,pv=evaluate(val_loader)
        val_f1=float(f1_score(yv,pv,labels=eligible,average='macro',zero_division=0))
        history.append({'seed':seed,'epoch':epoch,'train_loss':loss_total/max(1,n_obs),
                        'validation_macroF1_eligible11':val_f1})
        print(f"  seed {seed:2d} epoch {epoch:2d} train_loss {loss_total/max(1,n_obs):.4f} val_macroF1_eligible11 {val_f1:.5f}",flush=True)
        if val_f1>best_score+1e-8:
            best_score=val_f1; best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            since=0
        else:
            since+=1
            if since>=args.patience:break
    if best_state is None:
        raise RuntimeError('No validation checkpoint selected')
    elapsed=time.perf_counter()-start
    model.load_state_dict(best_state)
    folder=output / f'seed_{seed}'
    folder.mkdir(exist_ok=False,parents=True)
    torch.save(best_state,folder/'DeepSets_mean_selected_state_dict.pt')
    safe_write_json(folder/'selection.json',{
        'seed':seed, 'best_epoch':best_epoch, 'best_validation_macroF1_eligible11':best_score,
        'train_seconds':elapsed, 'n_parameters':count,
        'optimizer':'AdamW', 'lr':args.lr, 'weight_decay':1e-4,'gradient_clip_norm':2.0,
        'schedule':'CosineAnnealingLR(T_max=10)', 'batch_size':args.batch_size,
        'epochs_max':args.epochs,'patience':args.patience, 'dropout':args.dropout,
        'balance_sampler':args.balance_sampler, 'architecture':f'phi: Linear(D,{args.mlp_width})-LN-LeakyReLU; Linear({args.mlp_width},{args.mlp_width})-LN-LeakyReLU; Linear({args.mlp_width},256)-LeakyReLU; mean over 10 records; rho: 256-128-12',
        'no_graph_attention':True,'no_graph_edges':True,
        'torch':torch.__version__, 'device':str(device)})
    pd.DataFrame(history).to_csv(folder/'training_history.csv', index=False)
    result_rows=[]
    for part, loader in [('validation',val_loader),('test',test_loader)]:
        yy,pp=evaluate(loader)
        if not np.array_equal(yy,labels[part]):
            raise AssertionError('DataLoader changed label order')
        metrics=score(yy,pp,classes,eligible,part,'DeepSets_mean',elapsed)
        metrics['macroF1_posthoc_common10']=float(f1_score(yy,pp,labels=common10,average='macro',zero_division=0))
        metrics['common10_class_labels']=[classes[i] for i in common10]
        metrics['seed']=seed
        safe_write_json(folder/f'DeepSets_mean_{part}.json',metrics)
        pd.DataFrame({'true':[classes[i] for i in yy], 'pred':[classes[i] for i in pp]}).to_csv(
            folder/f'DeepSets_mean_{part}_pred.csv',index=False)
        result_rows.append({'model':'DeepSets_mean','seed':seed,'partition':part,
                            'macroF1_common10':metrics['macroF1_posthoc_common10'],
                            'macroF1_eligible11':metrics['macro_F1_eligible_classes'],
                            'macroF1_all12':metrics['macro_F1_all_12_classes'],
                            'accuracy':metrics['accuracy'], 'n_parameters':count,'best_epoch':best_epoch,
                            'train_seconds':elapsed})
    print(f'FINAL seed={seed} epoch={best_epoch} TEST common10_macroF1={result_rows[-1]["macroF1_common10"]:.6f} acc={result_rows[-1]["accuracy"]:.6f}',flush=True)
    del model
    if device.type == 'cuda': torch.cuda.empty_cache()
    return result_rows


def make_comparison(out, root, gat_runs, audit_sha, test_true, classes, common10):
    from sklearn.metrics import f1_score, accuracy_score
    rows=[]
    for seed in SEEDS:
        for model in EXPERIMENT_VARIANTS:
            if model=='DeepSets_mean':
                path=out/f'seed_{seed}'/'DeepSets_mean_test_pred.csv'
            elif model.startswith('GATv2_'):
                path=gat_runs.get(seed,Path('/NONEXISTENT'))/f'{model}_test_pred.csv'
            else:
                runs=find_existing_runs(root,'baselines',SEEDS)
                path=Path('/NONEXISTENT')
                for run,meta in reversed(runs.get(seed,[])):
                    p=run/f'{model}_test_pred.csv'
                    if p.is_file() and meta.get('protocol',{}).get('input_sha256')==audit_sha:
                        path=p;break
            if not path.is_file(): continue
            d=pd.read_csv(path).astype(str)
            if not np.array_equal(test_true,d['true'].to_numpy()):
                raise RuntimeError(f'Ground-truth mismatch for {path}')
            y=d['true'].to_numpy();p=d['pred'].to_numpy();selected=[classes[i] for i in common10]
            rows.append({'seed':seed,'model':model, 'macroF1_common10':float(f1_score(y,p,labels=selected,average='macro',zero_division=0)),
                         'accuracy':float(accuracy_score(y,p)), 'prediction_file':str(path)})
    if not rows:raise RuntimeError('No saved predictions')
    raw=pd.DataFrame(rows);raw.to_csv(out/'matched_five_seed_comparison.csv',index=False)
    agg=raw.groupby('model').agg(seeds=('seed','nunique'),macroF1_10_mean=('macroF1_common10','mean'),
        macroF1_10_sd=('macroF1_common10','std'),accuracy_mean=('accuracy','mean'),
        accuracy_sd=('accuracy','std')).reset_index()
    agg.to_csv(out/'matched_five_seed_summary.csv',index=False)
    pair=[]
    wide=raw.pivot(index='seed',columns='model',values='macroF1_common10')
    for a in ['GATv2_selfloops','GATv2_uppertriangular','XGB_window_mean','RF_window_mean']:
        if a not in wide.columns:continue
        for seed in SEEDS:
            if seed in wide.index and pd.notna(wide.loc[seed,'DeepSets_mean']) and pd.notna(wide.loc[seed,a]):
                pair.append({'seed':seed,'comparison':'DeepSets_mean minus '+a,
                    'macroF1_common10_difference':float(wide.loc[seed,'DeepSets_mean']-wide.loc[seed,a])})
    if pair:
        pp=pd.DataFrame(pair);pp.to_csv(out/'paired_differences_common10.csv',index=False)
        print('\nPAIRED SEED DIFFERENCES (descriptive, not statistical significance):')
        print(pp.groupby('comparison').macroF1_common10_difference.agg(['mean','std','min','max']).to_string())
    print('\nFIVE-SEED MODEL COMPARISON (missing models/seeds clearly shown):')
    print(agg.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    return agg


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv',type=Path,default=DEFAULT_CSV)
    p.add_argument('--root',type=Path,default=DEFAULT_OUT)
    p.add_argument('--out',type=Path,default=Path.home()/'Downloads'/'CBID_31617_DeepSets_5Seeds')
    p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--mlp-width',type=int,default=1536, help='Frozen ~GATv2 parameter-budget control, not a tuned hyperparameter')
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--window-size',type=int,default=10)
    p.add_argument('--stride',type=int,default=10)
    p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--dropout',type=float,default=.3)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--epochs',type=int,default=18)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--balance-sampler',action='store_true')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--smoke-test',action='store_true')
    args=p.parse_args()
    if args.smoke_test:smoke_test();return
    if args.hidden!=256 or args.mlp_width!=1536 or args.heads!=4 or args.window_size!=10 or args.stride!=10 or abs(args.lr-.0003)>1e-12 or args.batch_size!=32 or args.epochs!=18 or args.patience!=5 or args.balance_sampler or args.dropout!=.3:
        p.error('This control script requires fixed design: MLP width1536, hidden256, GAT heads4 reference, W10, stride10, lr.0003, batch32, epochs18, patience5, dropout.3, no sampler. Do not tune on test.')
    if not args.csv.is_file():p.error(f'Dataset not found: {args.csv}')
    if args.out.exists():p.error(f'Output folder already exists: {args.out}. Use a NEW --out; do not overwrite experiment evidence.')
    args.out.mkdir(parents=True,exist_ok=False)
    print('=== DeepSets MLP+mean control: fixed protocol, five seeds ===',flush=True)
    print('Source CSV:',args.csv)
    torch,nn,F=get_torch()
    print('PyTorch',torch.__version__,'CUDA available',torch.cuda.is_available(),flush=True)
    # The dataset is prepared once. All seeds reuse SAME splits, features, windows.
    parts, windows, protocol, features, classes, eligible = inspect_clean_split(
        args.csv,args.out,args.window_size,args.stride)
    source_sha=protocol['input_sha256']
    ppX, prepdesc=preprocessing(parts,features)
    safe_write_json(args.out/'preprocessing_description.json',prepdesc)
    labels={part:windows[part][1] for part in ('train','validation','test')}
    Xwindows={part:np.ascontiguousarray(ppX[part][windows[part][0]])
             for part in ('train','validation','test')}
    support=pd.read_csv(args.out/'per_class_split_and_window_support.csv')
    common_names=support.loc[(support[['train_windows','validation_windows','test_windows']]>0).all(axis=1),'class'].tolist()
    common10=[classes.index(c) for c in common_names]
    if len(common10)!=10:
        raise ValueError(f'Expected 10 common window-supported classes, got {len(common10)}; STOP: protocol inconsistent.')
    # Cross-check against prior GAT predictions; prevents accidental mismatched dataset runs.
    truth=np.array([classes[i] for i in labels['test']],dtype=str)
    desired_args=vars(args)
    gat_runs=verify_existing_gat(args.root,source_sha,truth,classes,desired_args)
    if len(gat_runs)!=5:
        raise RuntimeError(f'All five GAT runs with matching dataset SHA, labels and parameters required; found {sorted(gat_runs)}. No training was started.')
    # Parameter-budget cross-check against the saved (unmodified) PyG GATv2 run.
    selfloops_selection = json.loads((gat_runs[SEEDS[0]]/'GATv2_selfloops_selection.json').read_text(encoding='utf-8'))
    gat_parameter_count = int(selfloops_selection['n_parameters'])
    probe_model = define_model(ppX['train'].shape[1], len(classes), args.mlp_width, args.hidden, args.dropout)
    deepsets_parameter_count = sum(t.numel() for t in probe_model.parameters())
    del probe_model
    print(f'Parameter budget: DeepSets={deepsets_parameter_count:,}, GATv2_selfloops={gat_parameter_count:,}, ratio={deepsets_parameter_count/gat_parameter_count:.3f}', flush=True)
    safe_write_json(args.out/'experiment_metadata.json',{
        'name':'DeepSets/MLP+mean architectural control, IJ-AI 31617',
        'utc_start':datetime.now(timezone.utc).isoformat(),
        'python':platform.python_version(),'torch':torch.__version__,
        'source_csv_sha256':source_sha,'csv_path':str(args.csv),
        'deepsets_parameter_count':deepsets_parameter_count,'gatv2_selfloops_parameter_count':gat_parameter_count,
        'parameter_count_ratio_deepsets_over_gat':deepsets_parameter_count/gat_parameter_count,
        'seeds':SEEDS,'original_gat_folders':{str(k):str(v) for k,v in gat_runs.items()},
        'arguments':{k:str(v) for k,v in vars(args).items()},
        'validation_checkpoint_selection_uses_existing_eligible11':[classes[i] for i in eligible],
        'posthoc_common10_reporting_labels':common_names,
        'neither_new_blind_test_nor_temporal_or_interdevice_data':True,
        'class_sorted_window_bias_unchanged':True,
        'difference_from_gat':'Replace graph convolution with per-record MLP and permutation-invariant mean pooling; training schedule same.',
        'architecture_frozen_before_this_run':True,
        'test_already_inspected_before_this_experiment':True,
        'not_evidence_of_graph_advantage_or_deployed_prevention':True,
    })
    runrows=[]
    for seed in SEEDS:
        print(f'\n=== TRAIN DeepSets seed={seed} ===',flush=True)
        runrows += train_seed(seed,args,torch,Xwindows,labels,classes,eligible,common10,args.out)
    pd.DataFrame(runrows).to_csv(args.out/'deepsets_five_seed_metrics.csv',index=False)
    make_comparison(args.out,args.root,gat_runs,source_sha,truth,classes,common10)
    from shutil import make_archive
    archive=make_archive(str(args.out),'zip',root_dir=args.out)
    print('\nFINISHED:',args.out,'\nUPLOAD:',archive,flush=True)
    print('WARNING: Ten-class metric is post-hoc, test was previously inspected. No evidence of actual device coordination.',flush=True)

if __name__=='__main__':
    main()
