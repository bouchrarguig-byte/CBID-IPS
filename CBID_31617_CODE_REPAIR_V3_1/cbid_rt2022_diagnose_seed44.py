#!/usr/bin/env python3
"""No-training diagnosis of IJ-AI 31617 seed 44 and class-bag artifacts.
Reads the previously saved GAT selected-run predictions and the per-window
manifest. Does not read the source RT-IoT2022 CSV or train any model.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def selected_run(root,seed):
    valid=[]
    for folder in sorted(root.glob('gat_*')):
        meta=folder/'full_run_metadata.json'
        if not meta.exists():continue
        try:
            obj=json.loads(meta.read_text(encoding='utf-8'))
            if int(obj['command_arguments']['seed'])!=seed:continue
        except (ValueError,KeyError):continue
        needed=[folder/f'{v}_{p}_pred.csv' for v in ('GATv2_uppertriangular','GATv2_selfloops') for p in ('validation','test')]
        if all(f.exists() for f in needed):valid.append(folder)
    if not valid:raise SystemExit(f'No complete run for seed {seed} found under {root}')
    return valid[-1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.home()/'Downloads'/'CBID_RT2022_Controlled_Window_v2')
    ap.add_argument('--seed',type=int,default=44)
    ap.add_argument('--out',type=Path,default=Path.home()/'Downloads'/'CBID_31617_Diagnostic_Seed44')
    args=ap.parse_args()
    run=selected_run(args.root,args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    print('Selected complete run:',run)
    metadata=json.loads((run/'full_run_metadata.json').read_text(encoding='utf-8'))
    cls=metadata.get('protocol',{}).get('classes')
    if not cls:
        metrics_file=run/'GATv2_uppertriangular_test.json'
        if not metrics_file.exists():
            raise SystemExit('Neither class list in metadata nor test JSON found')
        obj=json.loads(metrics_file.read_text(encoding='utf-8'))
        cls=[k for k in obj['per_class'] if k not in ('accuracy','macro avg','weighted avg','micro avg','samples avg')]
    manifest=run/'test_window_manifest.csv'
    manifest_df=pd.read_csv(manifest) if manifest.exists() else None
    variant_results=[]; all_predictions={}
    for variant in ('GATv2_uppertriangular','GATv2_selfloops'):
        p=pd.read_csv(run/f'{variant}_test_pred.csv')
        if not {'true','pred'}.issubset(p):raise SystemExit('Expected true,pred columns')
        true=p['true'].astype(str).to_numpy()
        pred=p['pred'].astype(str).to_numpy()
        wrong=(true!=pred)
        if manifest_df is not None and len(manifest_df)!=len(p):
            raise SystemExit('Window manifest and predictions lengths differ')
        p['window_id']=np.arange(len(p))
        p['wrong']=wrong
        if manifest_df is not None:
            p=p.merge(manifest_df,on='window_id',how='left',validate='one_to_one')
        p.to_csv(args.out/f'{variant}_test_diagnostic.csv',index=False)
        per=[]
        for label in cls:
            support=int((true==label).sum())
            tp=int(((true==label)&(pred==label)).sum())
            fp=int(((true!=label)&(pred==label)).sum())
            fn=int(((true==label)&(pred!=label)).sum())
            per.append({'class':label,'support':support,'correct':tp,
                        'errors_of_class':fn,'false_positives':fp,
                        'recall':float(tp/support) if support else None})
        pd.DataFrame(per).to_csv(args.out/f'{variant}_per_class_error.csv',index=False)
        cm=pd.DataFrame(confusion_matrix(true,pred,labels=cls),index=cls,columns=cls)
        cm.to_csv(args.out/f'{variant}_confusion.csv')
        mistakes=p[p.wrong].groupby(['true','pred']).size().sort_values(ascending=False)
        mistakes.rename('count').reset_index().to_csv(args.out/f'{variant}_top_confusions.csv',index=False)
        n_wrong=int(wrong.sum())
        variant_results.append({'variant':variant,'accuracy':float(accuracy_score(true,pred)),
                                'n_windows':len(p),'n_errors':n_wrong,
                                'macroF1_all12':float(f1_score(true,pred,labels=cls,average='macro',zero_division=0))})
        all_predictions[variant]={'true':true,'pred':pred}
        print(f'\n{variant}: incorrect windows={n_wrong}/{len(true)}; accuracy={1-n_wrong/len(true):.5f}')
        if len(mistakes):print(mistakes.head(10).to_string())
        if manifest_df is not None:
            mixed=(p['n_distinct_classes']>1).to_numpy()
            print('wrong pure-label bags:',int((wrong&~mixed).sum()),
                  '| wrong mixed-label bags:',int((wrong&mixed).sum()))
    pd.DataFrame(variant_results).to_csv(args.out/'variant_summary.csv',index=False)
    e=all_predictions['GATv2_uppertriangular']
    n=all_predictions['GATv2_selfloops']
    assert np.array_equal(e['true'],n['true']), 'Models evaluated on different test labels/order'
    ef=e['pred']!=e['true']; nf=n['pred']!=n['true']
    comp={'wrong_both':int(np.logical_and(ef,nf).sum()),
          'wrong_only_edges':int(np.logical_and(ef,~nf).sum()),
          'wrong_only_selfloops':int(np.logical_and(~ef,nf).sum()),
          'correct_both':int(np.logical_and(~ef,~nf).sum())}
    (args.out/'paired_error_breakdown.json').write_text(json.dumps(comp,indent=2),encoding='utf-8')
    print('\nPaired error breakdown:',comp)
    print('These are diagnostics, not a new independent evaluation or justification for excluding any seed.')
    shutil.make_archive(str(args.out),'zip',root_dir=args.out)
    print('OUTPUT:',args.out)
    print('UPLOAD:',str(args.out)+'.zip')

if __name__=='__main__':main()
