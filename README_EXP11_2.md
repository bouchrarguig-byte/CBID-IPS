# EXP-11.2 — Exploratory risk-ranked closed-loop replay

Keep the original EXP-11.1 script and frozen artefacts unchanged. This package adds only `cbid_exp11_2_risk.py` and a summary of the 48 EXP-11.1 runs reconstructed from the two supplied terminal logs (not the original CSV files).

Run from Downloads using the existing cbid-env, the previous scripts and the original frozen artefacts. No live network action. No new detector training. Two Poisson count models per detector/seed are fitted on the no-action day C to predict the next exact 10-second clock bin's malicious and benign flow counts for the same directed pair, including absent pairs as zero. The final clock is excluded. Evaluate on D, already used in development; results are exploratory. The forecasts are fitted on unmodified trajectories, so intervention-induced feature shift is a limitation. The risk models are saved with checksums. Do not load untrusted joblib files.

Policies: allow; score_cic (same threshold, credit rule and ordering as EXP-11.1 APRE+CIC); risk_gated_cic (same threshold and credit rule, rank by predicted attack count minus 10 times predicted benign count, abstain if <=0); risk_only_cic (same rule, without score threshold, secondary analysis). The penalty 10 is a fixed exploratory choice, not a calibrated constraint. The credit cap is 10 and replenishment is 0.01 per admitted pair-window; realised actions need not match between policies. No risk controller guarantees the 1% realised benign disruption constraint. The primary contrast is risk_gated_cic vs score_cic; report both models and all seeds, including failures. Compare the score_cic action hashes and metrics against EXP-11.1 before interpreting any improvement. No test labels enter scoring or action selection. Simulation suppresses complete future pair-window aggregates, not actual packets; offered traffic is fixed. It cannot establish prevention of attacks or real-world network safety.

```bash
cd "$HOME/Downloads"
unzip CBID_EXP11_2_risk.zip
source "$HOME/cbid-env/bin/activate"
python cbid_exp11_2_risk.py --self-test
python cbid_exp11_2_risk.py --root "$HOME/Downloads" --train CBID_TCAG_V10/05_confirmation/features/pair_windows_2019-04-26.csv --test CBID_APRE_V1/05_confirmation/inference/features/pair_windows_2019-04-27.csv --out CBID_EXP11_2/D_pilot --seeds 42
```

Run 43 and 44 to a new output folder after baseline parity check, without changing policy parameters. Package outputs and original EXP-11.1 results for analysis. Requires existing scripts and dependencies (NumPy, pandas, PyTorch, XGBoost, scikit-learn, joblib, threadpoolctl); no installation changes needed if already installed. Synthetic tests passed in the assistant environment, not an end-to-end real-data run.
