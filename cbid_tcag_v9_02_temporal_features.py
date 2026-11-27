#!/usr/bin/env python3
"""TCAG V9 development feature extraction. A/B only; no confirmation metrics.
One observation = directed source/destination pair in a 10 s UTC completion window.
Features available at window close. Labels are targets only, never features.
"""
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
DAYS=('2019-04-23','2019-04-25')
EXPECTED='b3b6256e970a8986d87716edea4bfd436d68f04058bf5c26cf67e6d839c83698'
KEY=['src','dst','w']
NUM=['IN_BYTES','OUT_BYTES','IN_PKTS','OUT_PKTS','FLOW_DURATION_MILLISECONDS','RETRANSMITTED_IN_PKTS','RETRANSMITTED_OUT_PKTS']
COLS=['FLOW_END_MILLISECONDS','IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_DST_PORT','Label','Attack']+NUM

def history(df, keys, cols, prefix):
    # Only completed previous windows, within this day. No row-order dependence.
    s=df.sort_values(keys+['w']).reset_index(drop=True)
    times=s.w.to_numpy(); values=s[cols].to_numpy(dtype=float)
    out=np.zeros((len(s),len(cols)+1))
    for ix in s.groupby(keys,sort=False).indices.values():
        t=times[ix]; v=values[ix]; end=np.searchsorted(t,t,side='left'); start=np.searchsorted(t,t-6,side='left')
        cs=np.vstack([np.zeros((1,len(cols))),np.cumsum(v,axis=0)])
        out[ix,:-1]=cs[end]-cs[start]; out[ix,-1]=end-start
    names=[prefix+c+'_prev60' for c in cols]+[prefix+'active_windows_prev60']
    s[names]=out
    return s[keys+['w']+names]

def features(g, ports):
    g=g.sort_values(KEY).reset_index(drop=True)
    g=g.merge(ports.groupby(KEY).size().rename('distinct_dst_ports').reset_index(),on=KEY,how='left',validate='one_to_one')
    g['distinct_dst_ports']=g.distinct_dst_ports.fillna(0)
    g['total_bytes']=g.IN_BYTES+g.OUT_BYTES; g['total_pkts']=g.IN_PKTS+g.OUT_PKTS
    g['mean_duration']=g.FLOW_DURATION_MILLISECONDS/g.flows
    local=['flows']+NUM+['distinct_dst_ports','total_bytes','total_pkts','mean_duration']
    src=g.groupby(['src','w'],as_index=False).agg(src_flows=('flows','sum'),src_bytes=('total_bytes','sum'),src_destinations=('dst','nunique'))
    g=g.merge(src,on=['src','w'],validate='many_to_one')
    g=g.merge(history(g,['src','dst'],['flows','total_bytes','total_pkts'],'pair_'),on=KEY,validate='one_to_one')
    g=g.merge(history(src,['src'],['src_flows','src_bytes','src_destinations'],'source_'),on=['src','w'],validate='many_to_one')
    peercols=['flows','total_bytes','total_pkts','distinct_dst_ports','pair_flows_prev60','pair_active_windows_prev60']
    grp=g.groupby(['dst','w'],sort=False)
    g['peer_count']=grp.src.transform('size')-1
    for c in peercols:
        g['peer_mean_'+c]=(grp[c].transform('sum')-g[c])/g.peer_count.clip(lower=1)
    local+=['src_flows','src_bytes','src_destinations']
    temporal=[c for c in g if c.startswith(('pair_','source_'))]
    relational=[c for c in g if c.startswith('peer_')]
    return g,{'local':local,'temporal':temporal,'relational':relational}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--archive',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--chunksize',type=int,default=250000); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256()
    with a.archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    if h.hexdigest()!=EXPECTED: raise SystemExit('Archive SHA-256 mismatch; stop.')
    bounds={d:(int(pd.Timestamp(d,tz='UTC').timestamp()*1000),int((pd.Timestamp(d,tz='UTC')+pd.Timedelta(days=1)).timestamp()*1000)) for d in DAYS}
    pieces={d:[] for d in DAYS}; ports={d:[] for d in DAYS}; counts={d:{} for d in DAYS}
    with zipfile.ZipFile(a.archive) as z:
        members=[m for m in z.infolist() if Path(m.filename).name.lower()=='nf-ton-iot-v3.csv']
        if len(members)!=1: raise SystemExit('Expected unique CSV')
        with z.open(members[0]) as f:
            for i,c in enumerate(pd.read_csv(f,usecols=COLS,chunksize=a.chunksize,dtype={'IPV4_SRC_ADDR':'string','IPV4_DST_ADDR':'string'}),1):
                t=pd.to_numeric(c.FLOW_END_MILLISECONDS,errors='coerce')
                for d,(lo,hi) in bounds.items():
                    mask=t.ge(lo)&t.lt(hi); s=c.loc[mask].copy()
                    if s.empty: continue
                    s=s.rename(columns={'IPV4_SRC_ADDR':'src','IPV4_DST_ADDR':'dst'}); s['w']=(t.loc[mask]//10000).astype('int64')
                    if s[['src','dst']].isna().any().any() or s.src.str.strip().eq('').any() or s.dst.str.strip().eq('').any(): raise ValueError('Missing endpoint')
                    lab=pd.to_numeric(s.Label,errors='raise')
                    if not lab.isin([0,1]).all(): raise ValueError('Nonbinary label')
                    s['attack_flows']=lab.astype('int64'); s['flows']=1
                    for k,v in s.Attack.fillna('<missing>').value_counts().items(): counts[d][str(k)]=counts[d].get(str(k),0)+int(v)
                    for k in NUM:
                        s[k]=pd.to_numeric(s[k],errors='raise')
                        if not np.isfinite(s[k]).all() or (s[k]<0).any(): raise ValueError('Invalid numeric '+k)
                    pieces[d].append(s.groupby(KEY,sort=False)[['flows','attack_flows']+NUM].sum().reset_index())
                    ports[d].append(s[KEY+['L4_DST_PORT']].dropna().drop_duplicates())
                if i%20==0: print('Chunks processed:',i,flush=True)
    report={'sha256':h.hexdigest(),'window_seconds':10,'history_seconds':60,'target':'any malicious completed flow in pair-window','days':{}}
    for d in DAYS:
        g=pd.concat(pieces[d],ignore_index=True).groupby(KEY,as_index=False)[['flows','attack_flows']+NUM].sum(); pieces[d].clear()
        port=pd.concat(ports[d],ignore_index=True).drop_duplicates(); ports[d].clear()
        g,groups=features(g,port); g['y']=(g.attack_flows>0).astype('int8')
        g.to_csv(a.out/f'pair_windows_{d}.csv',index=False)
        report['feature_groups']=groups
        report['days'][d]={'flows':int(g.flows.sum()),'windows':len(g),'positive_windows':int(g.y.sum()),'negative_windows':int((g.y==0).sum()),'mixed_windows':int(((g.attack_flows>0)&(g.attack_flows<g.flows)).sum()),'flow_attack_counts':counts[d]}
    (a.out/'feature_manifest.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
