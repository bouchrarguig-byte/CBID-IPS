#!/usr/bin/env python3
"""TCAG V9: label-blind structural feasibility on development dates only.
No training, no labels, no confirmation-period statistics.
"""
import argparse, hashlib, json, zipfile
from pathlib import Path
import pandas as pd
import numpy as np

DAYS = ('2019-04-23','2019-04-25')
TS = 'FLOW_END_MILLISECONDS'
SRC = 'IPV4_SRC_ADDR'
DST = 'IPV4_DST_ADDR'
COLS = [TS,SRC,DST]

def summarize(g):
    g = g.groupby(['src','dst','w'],sort=False,as_index=False)['flows'].sum()
    g = g.sort_values(['src','dst','w'])
    previous = g.groupby(['src','dst'],sort=False)['w'].diff()
    peers = g.groupby(['dst','w'],sort=False)['src'].transform('size')
    multi = peers >= 2
    def ratio(mask):
        return float(mask.mean()) if len(g) else None
    return {'flows':int(g.flows.sum()),'source_destination_windows':len(g),
            'distinct_sources':int(g.src.nunique()),'distinct_destinations':int(g.dst.nunique()),
            'destination_windows':int(g[['dst','w']].drop_duplicates().shape[0]),
            'source_windows_with_concurrent_peer':int(multi.sum()),
            'fraction_with_concurrent_peer':ratio(multi),
            'fraction_with_same_pair_in_previous_10s':ratio(previous.eq(1)),
            'fraction_with_same_pair_in_previous_60s':ratio(previous.between(1,6)),
            'fraction_with_peer_and_pair_history_60s':ratio(multi & previous.between(1,6))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--chunksize',type=int,default=250000)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256()
    with a.archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    results={'archive':str(a.archive),'archive_sha256':h.hexdigest(), 'days':list(DAYS),'window_seconds':10,'labels_read':False}
    bounds={d:(int(pd.Timestamp(d,tz='UTC').timestamp()*1000),int((pd.Timestamp(d,tz='UTC')+pd.Timedelta(days=1)).timestamp()*1000)) for d in DAYS}
    pieces={d:[] for d in DAYS}; invalid={d:0 for d in DAYS}
    with zipfile.ZipFile(a.archive) as z:
        members=[m for m in z.infolist() if Path(m.filename).name.lower()=='nf-ton-iot-v3.csv']
        if len(members)!=1: raise SystemExit(f'Expected one NF-ToN-IoT-v3.csv, found {len(members)}')
        m=members[0]; results['member']=m.filename; results['uncompressed_bytes']=m.file_size
        with z.open(m) as f: columns=pd.read_csv(f,nrows=0).columns.tolist()
        results['schema']=columns
        missing=set(COLS)-set(columns)
        if missing: raise SystemExit(f'Missing required columns: {sorted(missing)}')
        with z.open(m) as f:
            for i,c in enumerate(pd.read_csv(f,usecols=COLS,dtype={SRC:'string',DST:'string'},chunksize=a.chunksize),1):
                t=pd.to_numeric(c[TS],errors='coerce')
                for d,(lo,hi) in bounds.items():
                    mask=t.ge(lo)&t.lt(hi)
                    s=c.loc[mask,[SRC,DST]].copy(); s['w']=(t.loc[mask]//10000).astype('int64')
                    valid=s[SRC].notna()&s[DST].notna()&s[SRC].str.strip().ne('')&s[DST].str.strip().ne('')
                    invalid[d]+=int((~valid).sum()); s=s.loc[valid].rename(columns={SRC:'src',DST:'dst'})
                    if len(s): pieces[d].append(s.groupby(['src','dst','w'],sort=False).size().rename('flows').reset_index())
                if i%20==0: print(f'Chunks processed: {i}',flush=True)
    for d in DAYS:
        g=pd.concat(pieces[d],ignore_index=True) if pieces[d] else pd.DataFrame({'src':pd.Series(dtype='string'),'dst':pd.Series(dtype='string'),'w':pd.Series(dtype='int64'),'flows':pd.Series(dtype='int64')})
        results[d]=summarize(g); results[d]['rows_missing_endpoint']=invalid[d]
        del g; pieces[d].clear()
    output=a.out/'tcag_v9_01_audit.json'; output.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2)); print(f'Saved: {output}')
if __name__=='__main__': main()
