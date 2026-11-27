#!/usr/bin/env python3
"""Verify archived artifact identity and recorded replay arithmetic, without loading models."""
import csv, hashlib, json, math, pathlib, ast
ROOT=pathlib.Path(__file__).resolve().parent

def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def check(rel,h):
    p=ROOT/rel
    if not p.is_file() or digest(p)!=h: raise ValueError(f'Hash mismatch or missing: {rel}')
def read(rel): return json.loads((ROOT/rel).read_text())

def main():
    f=pathlib.Path('CBID_TCAG_V10/04_frozen_models'); p=pathlib.Path('CBID_APRE_V1/04_frozen_policy'); a=pathlib.Path('CBID_FINAL_AUDIT/09_final_protocol/final_protocol.json')
    check(f/'freeze.json','92198e78d35047750afc34edf96c9f482a81a746c302b7170fc8a857cbc2952e')
    check(p/'policy_freeze.json','d148b49e190382565810ca580d598a853fe769ad50a841250f306becc2aaf023')
    check(a,'28ddf8fb93986db0fa99ee5f79378e3c4ec777b8358c6e9033f78a47ea0b8e2e')
    m=read(f/'freeze.json'); pol=read(p/'policy_freeze.json')
    for name,h in m['files'].items(): check(f/name,h)
    for name,h in m['sources'].items(): check(name,h)
    check(p/'thresholds.csv',pol['thresholds_sha256']); check('cbid_apre_v1_04_freeze_policy.py',pol['script_sha256']); check('cbid_apre_v1_03_relation_gate.py',pol['audit_script_sha256'])
    check('exp11_4_freeze.json','c48b1feca8df7a3d5bc032f092a773df873563c6586fa7fd26cfee6be6545d0c')
    check('exp12_3_freeze.json','9b66feea71f8096b6101455e3f63a388a92052fe99a9e71d54e3c55cf3d90964')
    for fn in ['exp11_4_freeze.json','exp12_3_freeze.json']:
        for name,h in read(fn)['source_hashes'].items(): check(name,h)
    risk=read('exp11_4_freeze.json')['risk_artifacts']
    for key,h in risk.items():
        matches=list((ROOT/'CBID_EXP11_2').glob('*/risk_'+key+'.joblib'))
        if len(matches)!=1: raise ValueError(f'Risk file missing or duplicated: {key}')
        check(matches[0].relative_to(ROOT),h)
    stages={'11_1':48,'11_2':24,'11_3':80,'11_4':36,'12_2':18,'12_3':12}
    for stage,n in stages.items():
        rows=[]
        for fp in sorted((ROOT/('CBID_EXP'+stage)).glob('*/metrics.csv')):
            with fp.open(newline='') as fh: rows.extend(csv.DictReader(fh))
        if len(rows)!=n: raise ValueError(f'EXP{stage}: expected {n}, got {len(rows)}')
        seen=set()
        for r in rows:
            key=tuple(r.get(x,'') for x in ['day','seed','model','policy','rate','penalty'])
            if key in seen: raise ValueError(f'Duplicate {stage}: {key}')
            seen.add(key)
            A,B,a,b=map(int,[r['total_attack'],r['total_benign'],r['blocked_attack'],r['blocked_benign']])
            if not(0<=a<=A and 0<=b<=B): raise ValueError('Count out of range')
            if not math.isclose(a/A,float(r['attack_suppression']),abs_tol=1e-12) or not math.isclose(b/B,float(r['benign_disruption']),abs_tol=1e-12): raise ValueError('Rate mismatch')
            if (b/B<=.01)!=(r['budget_pass'].lower()=='true'): raise ValueError('Budget mismatch')
        print(f'EXP{stage}: {n} archived rows OK')
    with (ROOT/'CBID_EXP12_1/diagnostic_v1/diagnostic_metrics.csv').open(newline='') as fh: diag=list(csv.DictReader(fh))
    assert len(diag)==18
    for fp in ROOT.rglob('*.py'): ast.parse(fp.read_text(encoding='utf-8'),filename=str(fp))
    print(f'PASS: {len(m["files"])} frozen detector artifacts, {len(risk)} risk bundles, 218 policy rows, 18 diagnostics; Python syntax OK.')
    print('This is a static integrity check, not independent model reproduction or license clearance.')

if __name__=='__main__':main()
