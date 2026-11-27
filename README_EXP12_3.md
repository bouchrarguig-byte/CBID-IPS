# EXP-12.3: frozen retrospective transfer of DA-EGAP

The sole candidate selected by the pre-existing EXP-12.2 README rule is M0_local / adaptive_risk, selected on D (mean suppression 0.033593579872252595, maximum benign disruption 0.009731206466397985). The residual cost heuristic is NOT part of this selected policy. No new training, grid, or threshold tuning. Previous E/F outcomes influenced design; this is retrospective stress testing, not independent confirmation. A relative percentile gate does not calibrate probabilities, and CIC is not a benign-flow guarantee. Completed-flow replay is not evidence of packet-level prevention.

Parameters and exact script/input/artifact identities are fixed in exp12_3_freeze.json. Keep all previous Python files and saved estimator artifacts unchanged. Install these three files in Downloads. Use original sklearn 1.7.2 environment. Synthetic selftests passed on CPU sklearn 1.8.0; full real replay not performed here. Six selected-policy runs (3 per day) and six no-action reference trajectories are recorded. Prior EGAP baseline E/F is already recorded in EXP-11.4; not rerun. No additional candidate selection after E/F.

Minimum retrospective feasibility: positive suppression and <=1% benign disruption for all six selected runs. Separately report whether mean suppression reaches D's 2.5% development floor on each day. Neither establishes independent validation. Stop tuning on these traces after this run, regardless of outcome.

```bash
cd "$HOME/Downloads"
source "$HOME/cbid-env/bin/activate"
python cbid_exp12_3_transfer.py --self-test
python cbid_exp12_3_transfer.py --root "$HOME/Downloads" --day E --out CBID_EXP12_3/E
python cbid_exp12_3_transfer.py --root "$HOME/Downloads" --day F --out CBID_EXP12_3/F
zip -r CBID_EXP12_3_RESULTS.zip CBID_EXP12_3/ cbid_exp12_3_transfer.py exp12_3_freeze.json README_EXP12_3.md
sha256sum CBID_EXP12_3_RESULTS.zip
unzip -t CBID_EXP12_3_RESULTS.zip | tail -n 1
```
