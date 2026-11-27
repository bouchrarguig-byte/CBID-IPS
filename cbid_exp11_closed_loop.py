#!/usr/bin/env python3
"""EXP-11.1: exploratory causal pair-window replay, not packet-level IPS.
Only admitted observations feed history and model. Labels used solely in accounting.
A block at window close w suppresses the next clock window w+1 for that pair.
"""
import argparse, collections, hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from cbid_tcag_v9_06_graph_prototype import Model
from cbid_tcag_v9_02_temporal_features import NUM, features

MODEL_HASH='92198e78d35047750afc34edf96c9f482a81a746c302b7170fc8a857cbc2952e'
POLICY_HASH='d148b49e190382565810ca580d598a853fe769ad50a841250f306becc2aaf023'
MODELS=['M0_local','G1_real_peer']
SEEDS=[42,43,44]
DAYS={'C':('2019-04-26',91893,4827456),'D':('2019-04-27',84086,3750975)}
KEY=['src','dst','w']; BASE=KEY+['flows']+NUM+['distinct_dst_ports']
POLICIES=['allow','fixed_B','apre_C','apre_C_cic']

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def check(path, expected):
    if sha(path)!=expected: raise ValueError('Hash mismatch: '+str(path))

def load(root):
    f=root/'CBID_TCAG_V10/04_frozen_models'; p=root/'CBID_APRE_V1/04_frozen_policy'
    check(f/'freeze.json',MODEL_HASH); check(p/'policy_freeze.json',POLICY_HASH)
    m=json.loads((f/'freeze.json').read_text()); pol=json.loads((p/'policy_freeze.json').read_text())
    for name,h in m['files'].items(): check(f/name,h)
    check(p/'thresholds.csv',pol['thresholds_sha256'])
    # The original Model definition must be byte-identical to its frozen source.
    import cbid_tcag_v9_06_graph_prototype as gp
    check(gp.__file__,m['sources']['cbid_tcag_v9_06_graph_prototype.py'])
    import cbid_tcag_v9_02_temporal_features as fe
    check(fe.__file__,m['sources']['cbid_tcag_v9_02_temporal_features.py'])
    cal=pd.read_csv(f/'calibration.csv',float_precision='round_trip').set_index(['seed','model'])
    th=pd.read_csv(p/'thresholds.csv',float_precision='round_trip').set_index(['seed','model','horizon_s'])
    if not cal.index.is_unique or not th.index.is_unique: raise ValueError('Duplicate thresholds')
    return f,m['feature_groups'],cal,th

class History:
    def __init__(self):
        self.pair={}; self.source={}; self.last=None
    @staticmethod
    def previous(store,key,w):
        h=store.get(key)
        if h is None:return np.zeros(4)
        while h and h[0][0]<w-6: h.popleft()
        if not h:return np.zeros(4)
        return np.r_[np.sum([v for _,v in h],axis=0),len(h)]
    def build(self,d,w):
        if self.last is not None and w<=self.last: raise ValueError('Nonmonotone windows')
        self.last=w
        # This routine receives no labels or unavailable rows.
        if not set(d.columns)==set(BASE): raise ValueError('Unexpected input columns')
        d=d.copy().reset_index(drop=True)
        if d.empty:return d
        d['total_bytes']=d.IN_BYTES+d.OUT_BYTES; d['total_pkts']=d.IN_PKTS+d.OUT_PKTS
        d['mean_duration']=d.FLOW_DURATION_MILLISECONDS/d.flows
        s=d.groupby('src',sort=False).agg(src_flows=('flows','sum'),src_bytes=('total_bytes','sum'),src_destinations=('dst','nunique'))
        for c in s.columns:d[c]=d.src.map(s[c])
        ph=np.array([self.previous(self.pair,(a,b),w) for a,b in zip(d.src,d.dst)])
        sh={a:self.previous(self.source,a,w) for a in s.index}
        d[['pair_flows_prev60','pair_total_bytes_prev60','pair_total_pkts_prev60','pair_active_windows_prev60']]=ph
        d[['source_src_flows_prev60','source_src_bytes_prev60','source_src_destinations_prev60','source_active_windows_prev60']]=np.array([sh[a] for a in d.src])
        grp=d.groupby('dst',sort=False); d['peer_count']=grp.src.transform('size')-1
        for c in ['flows','total_bytes','total_pkts','distinct_dst_ports','pair_flows_prev60','pair_active_windows_prev60']:
            d['peer_mean_'+c]=(grp[c].transform('sum')-d[c])/d.peer_count.clip(lower=1)
        for a,b,v in zip(d.src,d.dst,d[['flows','total_bytes','total_pkts']].to_numpy()):self.pair.setdefault((a,b),collections.deque()).append((w,v))
        for a,v in zip(s.index,s.to_numpy()):self.source.setdefault(a,collections.deque()).append((w,v))
        return d

class Scorer:
    def __init__(self,f,g,seed,name):
        self.name=name;self.cols=g['local']+(g['temporal'] if name=='G1_real_peer' else [])
        if name=='M0_local':
            self.model=xgb.Booster();self.model.load_model(f/f'{seed}_{name}.json');self.model.set_param({'device':'cpu','nthread':1})
        else:
            norm=np.load(f/f'{seed}_{name}.npz');self.mean=norm['mean'];self.std=norm['std']
            self.model=Model(len(self.cols));self.model.load_state_dict(torch.load(f/f'{seed}_{name}.pt',map_location='cpu',weights_only=True));self.model.eval()
    def __call__(self,d):
        z=d[self.cols].to_numpy(dtype=np.float64)
        if self.name=='M0_local':p=self.model.inplace_predict(z.astype('float32'),validate_features=False)
        else:
            v=((np.log1p(z)-self.mean)/self.std).clip(-12,12).astype('float32');g=pd.factorize(d.dst,sort=False)[0].astype('int64')
            with torch.no_grad():p=torch.sigmoid(self.model(torch.from_numpy(v),torch.from_numpy(g),int(g.max()+1),'real')).numpy()
        if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('Invalid scores')
        return p.astype(float)

def replay(d,score,theta,policy,reference=None,groups=None):
    hist=History(); scheduled=set(); prev=None; credit=0.; blocked_a=blocked_b=actions=0; observed=0; scored=0; history_error=0.; score_changes=0
    trace=[]; action_digest=hashlib.sha256(); blocked_windows=0
    # Input ordered by clock; the next window may be absent (then block expires).
    for w,raw in d.groupby('w',sort=True):
        w=int(w); pairs=list(zip(raw.src,raw.dst)); scheduled=scheduled if prev==w-1 else set()
        blocked=np.array([k in scheduled for k in pairs],dtype=bool)
        # Ground truth used only for offline evaluation; excluded from decision input.
        blocked_windows+=int(blocked.sum())
        blocked_a+=int(raw.attack_flows.to_numpy()[blocked].sum());blocked_b+=int((raw.flows-raw.attack_flows).to_numpy()[blocked].sum())
        admitted=raw.loc[~blocked,BASE]; observed+=len(admitted); obs=hist.build(admitted,w); scheduled=set();prev=w
        if obs.empty:continue
        if policy=='allow' and groups is not None:
            cols=sum([groups[k] for k in ('local','temporal','relational')],[])
            v=raw.loc[~blocked,cols].to_numpy(dtype=float); actual=obs[cols].to_numpy(dtype=float)
            if not np.allclose(actual,v,rtol=1e-8,atol=1e-7):raise ValueError(f'Feature parity failed at {w}: {np.max(np.abs(actual-v))}')
            history_error=max(history_error,float(np.max(np.abs(actual-v))))
        if score is None:continue
        p=score(obs);scored+=len(p)
        if reference is not None:
            baseline=np.array([reference[(w,a,b)] for a,b in zip(obs.src,obs.dst)])
            score_changes+=int((np.abs(baseline-p)>1e-6).sum())
        if policy=='allow':
            trace.extend((w,a,b,float(s)) for a,b,s in zip(obs.src,obs.dst,p));continue
        candidates=np.flatnonzero(p>=theta)
        # Deterministic score descending, then src/dst tie-break; no future data.
        order=sorted(candidates,key=lambda i:(-p[i],str(obs.src.iloc[i]),str(obs.dst.iloc[i])))
        if policy=='apre_C_cic':
            credit=min(10.,credit+.01*len(obs));order=order[:math.floor(credit+1e-12)];credit-=len(order)
        scheduled={(obs.src.iloc[i],obs.dst.iloc[i]) for i in order};actions+=len(scheduled)
        for pair in sorted(scheduled):action_digest.update(json.dumps([w+1,*pair],separators=(',',':')).encode()+b'\n')
    A=int(d.attack_flows.sum());B=int((d.flows-d.attack_flows).sum())
    if not (0<=blocked_a<=A and 0<=blocked_b<=B):raise AssertionError('Conservation')
    return dict(total_attack=A,total_benign=B,blocked_attack=blocked_a,blocked_benign=blocked_b,attack_suppression=blocked_a/A if A else None,benign_disruption=blocked_b/B if B else None,budget_pass=blocked_b/B<=.01 if B else None,actions=actions,blocked_pair_windows=blocked_windows,action_schedule_sha256=action_digest.hexdigest(),admitted_pair_windows=observed,scored_pair_windows=scored,scores_changed_vs_allow=score_changes,max_feature_parity_abs_error=history_error),trace

def selftest():
    rows=[]
    for w in [100,101,102,108]:
        for a,b in [('a','x'),('b','x'),('a','y')]:
            rows.append(dict(src=a,dst=b,w=w,flows=2,attack_flows=int(a=='a'),**{k:2 for k in NUM}))
    raw=pd.DataFrame(rows);ports=raw[KEY].assign(L4_DST_PORT=80)
    d,g=features(raw,ports);d=d.sort_values(['w','src','dst']);d['y']=(d.attack_flows>0).astype(int)
    def scorer(x):return np.ones(len(x))
    m,trace=replay(d,scorer,1,'allow',groups=g);assert m['blocked_attack']==m['blocked_benign']==0
    ref={(w,a,b):s for w,a,b,s in trace}
    m,_=replay(d,scorer,.5,'fixed_B',reference=ref);assert m['blocked_attack']==2 and m['blocked_benign']==4,m
    # Verify history and peer context change when one pair is removed, with no labels.
    h=History();o=h.build(d[d.w==100][BASE],100);o=h.build(d[(d.w==101)&(d.src=='b')][BASE],101);assert o.peer_count.iloc[0]==0
    o=h.build(d[d.w==102][BASE],102);assert o.loc[(o.src=='a')&(o.dst=='x'),'pair_active_windows_prev60'].iloc[0]==1
    # Label permutation must not change interventions/actions or scores.
    q=d.copy();q['attack_flows']=q.flows-q.attack_flows
    n,_=replay(q,scorer,.5,'fixed_B',reference=ref);assert n['action_schedule_sha256']==m['action_schedule_sha256'] and n['blocked_attack']==m['blocked_benign']
    print('SELFTEST PASS: parity, causal timing, expiry, feedback, label independence, accounting')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');p.add_argument('--root',type=Path,default=Path.home()/'Downloads');p.add_argument('--input',type=Path);p.add_argument('--day',choices=DAYS);p.add_argument('--out',type=Path);p.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=SEEDS);a=p.parse_args()
    torch.set_num_threads(1);selftest()
    if a.self_test:return
    if not a.input or not a.day or not a.out:p.error('--input --day --out required')
    f,g,cal,th=load(a.root); day,n,total=DAYS[a.day]
    d=pd.read_csv(a.input,dtype={'src':str,'dst':str}).sort_values(['w','src','dst']).reset_index(drop=True);s=int(pd.Timestamp(day,tz='UTC').timestamp())//10
    if len(d)!=n or int(d.flows.sum())!=total or not d.w.between(s,s+8639).all() or d.duplicated(KEY).any() or d[KEY].isna().any().any():raise ValueError('Unexpected input day/keys')
    if not np.isfinite(d[['flows','attack_flows']+NUM+['distinct_dst_ports']]).all().all() or not (d.flows>0).all() or not d.attack_flows.between(0,d.flows).all():raise ValueError('Invalid counts')
    if not (d[['flows','attack_flows']]%1==0).all().all() or not (d[NUM+['distinct_dst_ports']]>=0).all().all():raise ValueError('Invalid feature values')
    expected={'C':(2686781,2140675),'D':(3396651,354324)}[a.day]
    # Correct C attack count from manifest (DDoS 1,654,220 + password 1,032,561).
    if (int(d.attack_flows.sum()),int((d.flows-d.attack_flows).sum()))!=expected:raise ValueError('Unexpected attack/benign counts')
    a.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(stage='exploratory',day=day,input_sha256=sha(a.input),model_freeze_sha256=MODEL_HASH,policy_freeze_sha256=POLICY_HASH,script_sha256=sha(__file__),seeds=a.seeds,models=MODELS,policies=POLICIES,horizon_s=10,credit_rate=.01,credit_cap=10,python=platform.python_version(),torch=torch.__version__,xgboost=xgb.__version__,numpy=np.__version__,pandas=pd.__version__)
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    for seed in a.seeds:
        for name in MODELS:
            scorer=Scorer(f,g,seed,name); ref=None
            for policy in POLICIES:
                theta=None if policy=='allow' else float(cal.loc[(seed,name),'threshold'] if policy=='fixed_B' else th.loc[(seed,name,10),'oracle_threshold'])
                t=time.monotonic();r,trace=replay(d,scorer,theta,policy,reference=ref,groups=g)
                if policy=='allow':ref={(w,a,b):s for w,a,b,s in trace}
                r.update(day=day,seed=seed,model=name,policy=policy,threshold=theta,wall_seconds=time.monotonic()-t);rows.append(r);pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False);print(json.dumps(r),flush=True)
    summary=pd.DataFrame(rows).groupby(['model','policy']).agg(attack_mean=('attack_suppression','mean'),benign_mean=('benign_disruption','mean'),budget_passes=('budget_pass','sum'),n=('seed','size'),mean_actions=('actions','mean'),mean_changed_scores=('scores_changed_vs_allow','mean')).reset_index();summary.to_csv(a.out/'summary.csv',index=False);print(summary.to_string(index=False))
if __name__=='__main__':main()
