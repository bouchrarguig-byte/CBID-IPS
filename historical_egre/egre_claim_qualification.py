#!/usr/bin/env python3
"""EGRE: transparent qualification of EXISTING reported results (no model training).
This script verifies arithmetic and reports whether a relation-benefit claim is supported.
Inputs are summary numbers, not the underlying datasets or trained checkpoints.
The qualification is retrospective; NOT a deployable IPS or formal significance test.
"""
import json, pathlib, sys
p = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else pathlib.Path(__file__).with_name('EGRE_existing_results_input.json'))
d = json.loads(p.read_text())
for name in ('RT_IoT2022','NF_ToN_IoT_v3'):
 x=d[name]
 dm=x['edge']-x['no_messages']
 ds=None if x['shuffled'] is None else x['edge']-x['shuffled']
 semantics=x['real_host_time']
 split=x['split_integrity'].startswith('confirmed')
 independent=x['independent_campaigns'] and x['independent_uncertainty']
 positive=(dm>0 and ds is not None and ds>0)
 qualified=semantics and split and independent and positive
 print(f'{name}: {x["score"]}; delta_message={dm:+.6f}; delta_structure={"UNAVAILABLE" if ds is None else f"{ds:+.6f}"}; semantics={semantics}; split_confirmed={split}; independent={independent}; result={"ELIGIBLE FOR INDEPENDENT CONFIRMATION" if qualified else "RELATIONAL BENEFIT NOT ESTABLISHED"}')
x=d['DDoS_feasibility']; print(f'DDoS headroom: {x["missed_by_XGB_context"]}/{x["coordinated_observations"]}={x["missed_by_XGB_context"]/x["coordinated_observations"]:.4%}; source_sets={x["recurrent_source_sets"]}; no-go feasibility audit only')
