#!/usr/bin/env python3
"""CBID-IPS IJ-AI 31617 — V8 EDGE-AWARE, MULTI-TARGET GRAPH DEV STUDY.

Uses the existing source/destination/10-s aggregated V4 CSV and V6's exact
23-Apr training/validation boundaries and train-only preprocessing.

MAJOR CHANGE FROM V7:
  one graph per GLOBAL 10-second bin, with actual source and destination IPs as
  temporary node identifiers; observed pair-flows are directed EDGES. The
  nine flow-aggregate attributes are attached to edges and used by GATv2.
  Output: per source->destination flow-group class, not per artificial hub.

NO EXTERNAL TEST: the 24-Apr labels are filtered by time BEFORE they are read.
The 11 targets were label-selected in earlier exploration. All results here
are developmental and cannot be represented as blinded evidence.

Requires in same folder: cbid_gatv2_validation_v6.py.
Run:
  python cbid_v8_edge_aware_development.py --check-only
  python cbid_v8_edge_aware_development.py --seeds 11 22 33 --epochs 25

For a stronger contribution, the extraction should later be expanded to all
hosts, more NetFlow edge attributes, and an independent evaluation. This file
is a TESTABLE EXPERIMENTAL PROTOTYPE, not the complete final CBID-IPS product.
"""
from __future__ import annotations
import argparse
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

import cbid_gatv2_validation_v6 as v6


def build_global_graphs(part, local_array, torch, Data, *, variant='real_edges'):
    """Map IP identities to per-window node IDs. NEVER include raw IPs in x.

    All 9 features come from flows completed within their bin; labels NEVER
    enter construction of x / edge_attr / edge_index. The message graph has
    reverse arcs for representation learning (not additional observed flows).
    The edge-level classification head uses ORIGINAL observed direction only.
    """
    assert variant in ('real_edges', 'no_messages', 'shuffled_edges')
    output = []
    for tm, sub in part.groupby('window_start_ms_META', sort=True):
        rowids = sub.index.to_numpy(dtype='int64')
        attr = local_array[rowids].astype('float32')
        src_ips = sub['source_ip_META'].astype(str).to_numpy()
        dst_ips = sub['destination_ip_META'].astype(str).to_numpy()
        names = sorted(set(src_ips).union(dst_ips))
        lookup = {ip: idx for idx, ip in enumerate(names)}
        src = np.fromiter((lookup[p] for p in src_ips), dtype=np.int64)
        dst = np.fromiter((lookup[p] for p in dst_ips), dtype=np.int64)
        n = len(names)
        d = attr.shape[1]
        # Aggregate OUTGOING/INCOMING observed features per host. All features
        # are computed without labels and are available at the bin's end.
        sout, sin = np.zeros((n, d), np.float32), np.zeros((n, d), np.float32)
        cout, cin = np.zeros(n, np.float32), np.zeros(n, np.float32)
        np.add.at(sout, src, attr); np.add.at(sin, dst, attr)
        np.add.at(cout, src, 1.); np.add.at(cin, dst, 1.)
        x = np.concatenate((sout / np.maximum(cout[:, None], 1.),
                            sin / np.maximum(cin[:, None], 1.),
                            np.log1p(cout)[:, None], np.log1p(cin)[:, None]), axis=1)
        label_index = np.stack((src, dst), axis=0)
        y = sub[v6.LABEL].to_numpy(dtype=np.int64)
        # Graph topology is recomputed per 10-s global window. It is not a
        # transductive all-days graph and no IP integer/string is a feature.
        if variant == 'shuffled_edges':
            # Simple descriptive topology-perturbation control: permute
            # destination endpoints of MESSAGE arcs only, not true edge labels.
            rng = np.random.default_rng(20261123 + (int(tm) // 10_000) % 1_000_003)
            msg_dst = dst[rng.permutation(len(dst))]
        else:
            msg_dst = dst
        loops = np.arange(n, dtype=np.int64)
        if variant == 'no_messages':
            msg_index = np.stack((loops, loops))
            msg_attr = np.zeros((n, d + 1), np.float32)
        else:
            msg_index = np.stack((np.concatenate((src, msg_dst, loops)),
                                  np.concatenate((msg_dst, src, loops))))
            fwd = np.column_stack((attr, np.ones(len(attr), np.float32)))
            rev = np.column_stack((attr, -np.ones(len(attr), np.float32)))
            msg_attr = np.concatenate((fwd, rev, np.zeros((n, d+1), np.float32)))
        output.append(Data(
            x=torch.from_numpy(x.copy()),
            edge_index=torch.from_numpy(msg_index.copy()).long(),
            edge_attr=torch.from_numpy(msg_attr.copy()),
            edge_label_index=torch.from_numpy(label_index.copy()).long(),
            edge_label_attr=torch.from_numpy(attr.copy()),
            y=torch.from_numpy(y.copy()).long(),
            edge_row_ids=torch.from_numpy(rowids.copy()).long(),
        ))
    return output


def summary_graphs(part):
    sizes = part.groupby('window_start_ms_META', sort=True).size()
    if sizes.empty:
        raise RuntimeError('No graph windows')
    print(f"GLOBAL 10-s graphs={len(sizes):,}; edge records={len(part):,}; "
          f"pair records per graph median={sizes.median():.1f}, max={sizes.max():,}")
    print('  Unique source IPs in this split:', part.source_ip_META.nunique())
    print('  Unique destination IPs in this split:', part.destination_ip_META.nunique())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input', type=Path, default=v6.DEFAULT_INPUT)
    ap.add_argument('--output-root', type=Path,
                    default=Path.home() / 'Downloads' / 'CBID_NFToN_EdgeAware_v8')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--skip-latency', action='store_true', help='Skip fixed-batch inference latency')
    ap.add_argument('--skip-baselines', action='store_true',
                    help='Skip local and fair source/destination-context RF/XGB validation baselines')
    ap.add_argument('--seeds', nargs='+', type=int, default=[11,22,33,44,55])
    ap.add_argument('--variants', nargs='+', choices=['no_messages','real_edges','shuffled_edges'],
                    default=['no_messages','real_edges'])
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--weight-decay', type=float, default=2e-4)
    ap.add_argument('--embargo-seconds', type=int, default=120)
    ap.add_argument('--validation-start', default='2019-04-23T16:00:00+00:00')
    ap.add_argument('--holdout-start', default='2019-04-24T00:00:00+00:00')
    args = ap.parse_args()
    if not args.input.is_file():
        ap.error('Input CSV missing: ' + str(args.input))
    if any(s < 0 for s in args.seeds) or len(args.seeds) != len(set(args.seeds)):
        ap.error('Seeds must be distinct and nonnegative')
    if not all([args.lr>0, args.epochs>1, args.patience>0, args.batch_size>0,
                args.hidden>1, args.heads>0, 0<=args.dropout<1, args.embargo_seconds>=0]):
        ap.error('Invalid hyperparameters')
    t1, t2 = v6.utc_ms(args.validation_start), v6.utc_ms(args.holdout_start)
    print('CBID-IPS V8 — global host graph + edge-aware GATv2, DEVELOPMENT ONLY')
    print('Input:', args.input)
    print('24-Apr records excluded by time before labels were read.')
    parts = v6.read_train_validation(args.input, t1, t2, args.embargo_seconds*1000)
    v6.summarize_split(parts)
    for split in ('train','validation'):
        print(split.upper()+':',end=' '); summary_graphs(parts[split])
    if args.check_only:
        print('CHECK COMPLETE: no preprocessing or training; original data untouched.')
        return
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GATv2Conv
    except ImportError as err:
        raise SystemExit('Torch and torch_geometric needed in cbid-env: '+str(err))
    torch.set_num_threads(min(8, torch.get_num_threads()))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device',dev, 'torch',torch.__version__)
    loc, _, _ = v6.preprocess(parts)
    feature_dim = len(v6.FEATURES)
    node_dim = 2*feature_dim+2
    msg_dim = feature_dim+1
    yt = parts['train'][v6.LABEL].to_numpy(dtype=np.int64)
    yv = parts['validation'][v6.LABEL].to_numpy(dtype=np.int64)
    class_weights = len(yt) / (2*np.maximum(np.bincount(yt, minlength=2),1))
    # Keep loss / seed / parameters matched between real and control.
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights,device=dev,dtype=torch.float32))

    class EdgeAwareGAT(nn.Module):
        def __init__(self):
            super().__init__()
            h=args.hidden
            self.project = nn.Sequential(nn.Linear(node_dim,h),nn.GELU(),nn.Dropout(args.dropout))
            self.g1 = GATv2Conv(h,h,heads=args.heads,concat=False,edge_dim=msg_dim,
                                dropout=args.dropout,add_self_loops=False)
            self.g2 = GATv2Conv(h,h,heads=args.heads,concat=False,edge_dim=msg_dim,
                                dropout=args.dropout,add_self_loops=False)
            self.norm1=nn.LayerNorm(h)
            self.norm2=nn.LayerNorm(h)
            self.local = nn.Sequential(nn.Linear(feature_dim,h),nn.GELU(),nn.Linear(h,2))
            self.relation = nn.Sequential(nn.Linear(2*h+feature_dim,h),nn.GELU(),
                                          nn.Dropout(args.dropout),nn.Linear(h,2))
            self.gate_logit=nn.Parameter(torch.tensor(-1.5, dtype=torch.float32))

        def forward(self, batch):
            z=self.project(batch.x)
            z=self.norm1(z+0.5*F.dropout(F.elu(self.g1(z,batch.edge_index,batch.edge_attr)),
                                           p=args.dropout,training=self.training))
            z=self.norm2(z+0.5*F.dropout(F.elu(self.g2(z,batch.edge_index,batch.edge_attr)),
                                           p=args.dropout,training=self.training))
            src,dst=batch.edge_label_index
            e=batch.edge_label_attr
            return self.local(e)+torch.sigmoid(self.gate_logit)*self.relation(
                torch.cat((z[src],z[dst],e),dim=-1))

    out=args.output_root/('development_v8_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    config={
       'evaluation':'DEVELOPMENT ONLY; known 23-Apr validation; previously examined 24-Apr test NOT touched',
       'dataset_selection':'11 label-selected destinations; not an independent test',
       'architecture':'10-sec global multi-target host graph, pair-edge labels, 9 standardized edge features; '
                      'two edge-aware GATv2 layers with local-logit residual and learned gate',
       'graph_ablations':list(args.variants), 'classification_target':'source->destination 10s group scanning label',
       'holdout':'All rows >= 24-Apr filtered by timestamp BEFORE labels are accessed',
       'limitations':['Only nine coarse features, not all NetFlow features',
                     'Source/destination scope was selected after labels were examined',
                     'No independent test; cannot substantiate inter-device gain if validation is improved',
                     'Source IPs are network endpoints, not confirmed compromised IoT hardware',
                     'Future communication fields and prevention claims not validated'],
       'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
    }
    (out/'configuration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    pd.DataFrame({'y_true':yv}).to_csv(out/'validation_labels.csv',index=False)
    (out/'predictions').mkdir()
    (out/'checkpoints').mkdir()
    results=[]; histories=[]

    # Match the GNN's information budget as far as possible: a conventional
    # baseline sees the same source/destination *aggregates* and exact same
    # edge attributes. IPs themselves are keys only, not predictors.
    if not args.skip_baselines:
        from sklearn.ensemble import RandomForestClassifier
        try:
            from xgboost import XGBClassifier
        except ImportError:
            XGBClassifier = None
            print('WARNING: xgboost not installed; XGB baselines skipped')

        def global_context(p, local):
            z=pd.DataFrame(local,columns=v6.FEATURES)
            tm=p['window_start_ms_META']
            s=z.groupby([tm,p['source_ip_META']],sort=False).transform('mean').to_numpy()
            t=z.groupby([tm,p['destination_ip_META']],sort=False).transform('mean').to_numpy()
            sc=p.groupby(['window_start_ms_META','source_ip_META'],sort=False)[
                'source_ip_META'].transform('size').to_numpy(dtype='float32')
            dc=p.groupby(['window_start_ms_META','destination_ip_META'],sort=False)[
                'destination_ip_META'].transform('size').to_numpy(dtype='float32')
            return np.column_stack((local,s,t,np.log1p(sc),np.log1p(dc))).astype('float32')

        ctx_train=global_context(parts['train'],loc['train'])
        ctx_val=global_context(parts['validation'],loc['validation'])
        baseline_scores=[]
        algos=['RF'] + (['XGB'] if XGBClassifier is not None else [])
        print('\nVALIDATION-ONLY REFERENCE BASELINES (same edge labels/splits)')
        for name in algos:
            for feature_mode in ('local','context'):
                xx=loc['train'] if feature_mode=='local' else ctx_train
                vv=loc['validation'] if feature_mode=='local' else ctx_val
                if name=='RF':
                    clf=RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1,
                                               min_samples_leaf=2,class_weight='balanced_subsample')
                    clf.fit(xx,yt)
                else:
                    clf=XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,
                                      subsample=.85,colsample_bytree=.85,
                                      objective='binary:logistic',tree_method='hist',
                                      n_jobs=8,random_state=42)
                    clf.fit(xx,yt,sample_weight=class_weights[yt])
                p=clf.predict_proba(vv)[:,1]
                m=v6.metric(yv,p)
                row={'model':name+'_'+feature_mode,'n_features':xx.shape[1],
                     'AP':m['average_precision'],'macroF1_fixed05':m['macro_f1'],
                     'scan_recall_fixed05':m['scanning_recall'],
                     'MCC_fixed05':m['mcc']}
                baseline_scores.append(row)
                pd.DataFrame({'y_true':yv,'probability':p}).to_csv(
                    out/'predictions'/f'{name}_{feature_mode}.csv',index=False)
                print(row['model'], 'AP',round(row['AP'],4),
                      'macroF1',round(row['macroF1_fixed05'],4),flush=True)
        pd.DataFrame(baseline_scores).to_csv(out/'baseline_validation_metrics.csv',index=False)

    def infer(model,loader):
        model.eval()
        p=np.full(len(yv),np.nan,dtype='float64')
        with torch.inference_mode():
            for g in loader:
                g=g.to(dev)
                logits=model(g)
                pr=torch.softmax(logits,dim=-1)[:,1].detach().cpu().numpy()
                idx=g.edge_row_ids.detach().cpu().numpy()
                p[idx]=pr
        if not np.isfinite(p).all():
            raise RuntimeError('Invalid or unmapped validation rows')
        return p

    def latency(model,loader,device,repeats=4,max_batches=12):
        model=model.to(device)
        model.eval()
        spans=[]; edge_count=0; graph_count=0
        with torch.inference_mode():
            for pass_id in range(repeats+1):
                if device.type=='cuda':torch.cuda.synchronize()
                t0=time.perf_counter(); graphs=edges=0
                for j,g in enumerate(loader):
                    if j>=max_batches:break
                    graphs+=int(g.num_graphs);edges+=len(g.y)
                    z=model(g.to(device))
                    _=torch.softmax(z,dim=-1)[:,1].cpu()
                if device.type=='cuda':torch.cuda.synchronize()
                if pass_id:spans.append(time.perf_counter()-t0)
                edge_count,graph_count=edges,graphs
        return {'device':str(device),'graphs_timed':graph_count,
                'pair_records_timed':edge_count,
                'median_ms_per_graph':1000*float(np.median(spans))/max(graph_count,1),
                'median_ms_per_pair_record':1000*float(np.median(spans))/max(edge_count,1),
                'scope':'prebuilt graphs + PyG batching + transfers + GATv2 + softmax; '
                        'excludes NetFlow collection and graph feature construction'}

    latency_rows=[]
    for variant in args.variants:
        print('\nPreparing:',variant,flush=True)
        train_graphs=build_global_graphs(parts['train'],loc['train'],torch,Data,variant=variant)
        val_graphs=build_global_graphs(parts['validation'],loc['validation'],torch,Data,variant=variant)
        val_loader=DataLoader(val_graphs,batch_size=args.batch_size,shuffle=False)
        for seed in args.seeds:
            v6.seed_all(seed)
            train_loader=DataLoader(train_graphs,batch_size=args.batch_size,shuffle=True,
                                    generator=torch.Generator().manual_seed(seed))
            model=EdgeAwareGAT().to(dev)
            optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
            best=-1; best_epoch=None; best_state=None; last_better=0
            t0=time.perf_counter()
            for epoch in range(1,args.epochs+1):
                model.train(); losses=[]
                for g in train_loader:
                    g=g.to(dev)
                    optimizer.zero_grad(set_to_none=True)
                    logits=model(g)
                    loss=criterion(logits,g.y)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                probs=infer(model,val_loader)
                metrics=v6.metric(yv,probs)
                ap_score=metrics['average_precision']
                histories.append({'variant':variant,'seed':seed,'epoch':epoch,
                                  'train_loss':np.mean(losses),
                                  'validation_AP':ap_score,
                                  'validation_macroF1_fixed05':metrics['macro_f1'],
                                  'gate':float(torch.sigmoid(model.gate_logit).detach().cpu())})
                if ap_score>best+1e-8:
                    best=ap_score;best_epoch=epoch;last_better=epoch
                    best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
                if epoch in (1,5,10) or epoch-last_better>=args.patience:
                    print(f'{variant} seed={seed} epoch={epoch} '
                          f'loss={np.mean(losses):.4f} valAP={ap_score:.4f} best={best:.4f} '
                          f'macroF1={metrics["macro_f1"]:.4f}',flush=True)
                if epoch-last_better>=args.patience:
                    break
            model.load_state_dict(best_state)
            probs=infer(model,val_loader)
            metrics=v6.metric(yv,probs)
            row={'variant':variant,'seed':seed,'selected_epoch':best_epoch,
                 'training_seconds':time.perf_counter()-t0,
                 'parameters':sum(p.numel() for p in model.parameters()),
                 'gate':float(torch.sigmoid(model.gate_logit).detach().cpu()),
                 'validation_AP':metrics['average_precision'],
                 'validation_macroF1_fixed05':metrics['macro_f1'],
                 'validation_scan_recall_fixed05':metrics['scanning_recall'],
                 'validation_scan_precision_fixed05':metrics['scanning_precision'],
                 'validation_MCC_fixed05':metrics['mcc']}
            results.append(row)
            tag=f'{variant}_seed{seed}'
            pd.DataFrame({'y_true':yv,'probability':probs}).to_csv(out/'predictions'/f'{tag}.csv',index=False)
            torch.save({'state_dict':best_state,'tag':tag,'best_epoch':best_epoch,
                        'config':config},out/'checkpoints'/f'{tag}.pt')
            pd.DataFrame(results).to_csv(out/'validation_metrics.csv',index=False)
            pd.DataFrame(histories).to_csv(out/'training_history.csv',index=False)
            print('SELECTED',tag,'AP',round(row['validation_AP'],4),
                  'macroF1',round(row['validation_macroF1_fixed05'],4),'epoch',best_epoch,flush=True)
            if not args.skip_latency and seed==args.seeds[0]:
                for d in [torch.device('cpu')] + ([torch.device('cuda')] if torch.cuda.is_available() else []):
                    rec=latency(model,val_loader,d)
                    rec.update({'variant':variant,'seed':seed})
                    latency_rows.append(rec)
                    print('LATENCY',variant,rec['device'],
                          'ms/graph',round(rec['median_ms_per_graph'],4),flush=True)
                pd.DataFrame(latency_rows).to_csv(out/'inference_latency.csv',index=False)
                model.to(dev)
        del train_graphs,val_graphs,val_loader

    df=pd.DataFrame(results)
    summary=df.groupby('variant').agg(
      seeds=('seed','size'),AP_mean=('validation_AP','mean'),AP_sd=('validation_AP','std'),
      macroF1_mean=('validation_macroF1_fixed05','mean'),
      macroF1_sd=('validation_macroF1_fixed05','std'),
      MCC_mean=('validation_MCC_fixed05','mean'),
      training_seconds_mean=('training_seconds','mean')).reset_index()
    summary.to_csv(out/'summary_by_variant.csv',index=False)
    if {'no_messages','real_edges'}.issubset(set(args.variants)):
        pt=df.pivot(index='seed',columns='variant',values='validation_AP')
        paired=pd.DataFrame({'seed':pt.index,
            'AP_difference_real_minus_no_messages':pt.real_edges-pt.no_messages})
        paired.to_csv(out/'paired_edge_differences.csv',index=False)
    print('\n=== V8 DEVELOPMENT SUMMARY — NOT INDEPENDENT VALIDATION ===')
    print(summary.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print('Outputs:',out)
    print('STOP: Do not use old 24-Apr test as a new blinded test; results need independent verification.')

if __name__=='__main__':
    main()
