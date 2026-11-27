# EXP-12.1 — Read-only preventive transfer diagnosis

Run from the existing Downloads directory and original cbid-env (scikit-learn 1.7.2). Requires the intact EXP-11.1/2/3/4 scripts, exp11_4_freeze.json, frozen detection and prevention artifacts and the six trusted locally produced risk estimator bundles. No fitting, no new policy, no threshold sweep, no network action. D/E/F have already been evaluated and cannot serve as independent validation. Ground truth is used only for retrospective diagnostics after inference. E/F file hashes are recorded and inventories checked, not compared against an independently supplied EXP-11.4 input hash; compare protocol.json files from EXP-11.4 before reporting the diagnosis.

```bash
python cbid_exp12_1_diagnostic.py --self-test
python cbid_exp12_1_diagnostic.py --root "$HOME/Downloads" --out CBID_EXP12_1/diagnostic_v1
```

The complete run produces 18 model/day/seed diagnostics, 72 stage rows and 432 quantile rows. Output must be a new directory. All rates are fractions, not percentages. Means in summaries are unweighted over the seeds. The diagnostic uses the unmodified offered trajectory, with identical feature history and detector grouping; it does not reproduce the closed-loop policy or the credit scheduler. Forecasts are made on all current rows in this trajectory, not only on credit-eligible rows. The risk gate uses the fixed selected penalty (M0=5, G1=20) and existing C oracle threshold. No test labels are used for scoring or prediction.

- diagnostic_metrics.csv: AP, fixed gate recall/FPR, eligibility counts, next-bin predicted/observed count sums and MAE. Ratios are undefined (NaN) for zero observed totals.
- stage_coverage.csv: retrospective exact-next-bin pairing, all current candidates, score gate, positive utility alone, intersection. The final observed bin is excluded as a candidate origin (right censoring); all day flows are denominators. A pair absent from the next exact bin contributes zero. Each next row is counted at most once per stage. The all-current attack share is a structural coverage ceiling for an intervention restricted to pairs observed in the previous bin on the offered trace; stage-specific shares are diagnostic coverage, not a closed-loop counterfactual bound or achievable prevention rate.
- score_quantiles.csv: scores split by current pair-window label, only for analysis.
- summary.csv, stage_summary.csv: means for D/E/F.
- protocol.json: hashes and versions.

Interpretation: low all-current coverage suggests a short-horizon persistence limitation; low score-gate coverage relative to all-current suggests detector/threshold selection loss; low joint coverage relative to score-gate suggests additional utility-filter loss. Forecast MAE and aggregate predicted/observed ratio are descriptive, not proof of calibrated uncertainty. If a candidate solution is designed using E/F results, E/F remain retrospective diagnostic data; a new independent trace and fixed protocol are required. Completed-flow bins are not packets and cannot demonstrate actual packet interception.

Local validation of this package: syntax and synthetic diagnostic self-test executed. Full run requires the user's feature CSV files and original Python environment; these CSVs are not packaged.
