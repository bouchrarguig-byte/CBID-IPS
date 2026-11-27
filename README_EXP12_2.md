# EXP-12.2 — exploratory adaptive score gate on D

Design informed by the post-hoc E/F diagnostic. This runner reads D only. All prior detector/risk model artifacts and the original replay engine remain unchanged. D is an already explored development trace. No independent confirmation claim. E/F are also already explored and may only serve as retrospective stress tests after freezing a chosen policy. No live packet blocking is evaluated; completed-flow aggregates are indexed by flow end time and cannot establish packet-level causal prevention.

Three policies use the 90th percentile of scores in the preceding 60 exact 10-second clock bins, requiring 128 prior admitted scores. No action during warmup; the current scores enter the buffer only for subsequent windows. This is a relative score threshold, not probability calibration or a formal drift alarm. Ties are retained and ranked deterministically. Each policy has its own admitted observation history. The same CIC credit rate (.01) and capacity (10) as EXP-11.4 apply; this is not a benign-volume safety guarantee. Penalties are the earlier frozen values (M0=5, G1=20).

- adaptive_score: percentile gate, score ranking.
- adaptive_risk: percentile gate and positive predicted utility, rank by utility.
- adaptive_residual_risk: same gate, cost=max(predicted benign, max(0, max(current total flows, mean active pair volume in preceding 60 seconds)-predicted attack)). This persistence heuristic is not an upper confidence bound or a proven risk correction. No labels used in decisions.

Six candidate configurations, 3 seeds each, plus 6 no-action reference trajectories for feedback counts. All 18 candidate results and protocol hashes saved. Development gate: benign disruption <=1% in all three seeds and mean attack suppression >=2.5% (chosen exploratory tolerance before this run, not a statistical noninferiority test). The prior D M0 EGAP mean was 3.1232%; this gate only screens for substantial deterioration. If no candidate passes, stop this optimization. Otherwise select at most one candidate on D by maximum mean suppression, then minimum maximum benign disruption, then lexical model/policy tie break. Freeze before any further retrospective E/F stress evaluation; no tuning on E/F afterward. A truly new trace is needed for independent confirmation.

Install only these two new files next to the previously used scripts in Downloads. Retain original environment sklearn 1.7.2. The synthetic tests were run in a separate CPU environment (sklearn 1.8.0); real model deserialization and data replay are not tested here. Runner verifies original hashes and enforces original sklearn for real execution.

```bash
cd "$HOME/Downloads"
source "$HOME/cbid-env/bin/activate"
python cbid_exp12_2_adaptive.py --self-test
python cbid_exp12_2_adaptive.py --root "$HOME/Downloads" --out CBID_EXP12_2/D_adaptive_v1
```

The output directory must not already exist. Do not delete previous results. Check metrics.csv, summary.csv, protocol.json. Zip them with the script and README. No changes to the manuscript baseline until results are available.
