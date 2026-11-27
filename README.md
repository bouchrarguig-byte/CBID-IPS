# CBID: Evidence-Gated Intrusion Detection and Response for the Internet of Things

Reproducibility staging package for IJ-AI manuscript 31617, Bouchra RGUIG and Khalid HOUSNI. This is an author-review candidate for release, not yet a public or independently reproduced artifact. Original submission: 14 January 2026.

## Scope
The original record-window GATv2 investigation is retained and evaluated against matched controls. TCAG and EGRE assess relational benefit on chronological NF-ToN-IoT-v3 observations. APRE/RA-APRE, CIC, EGAP and DA-EGAP assess potential exposure and retrospective intervention replay. Replay removes completed-flow aggregates, not packets; it does not establish live IPS efficacy. The adaptive policy restores positive suppression on E/F but exceeds the benign-flow budget on two F seeds. Negative results are retained.

## Evidence map
| Manuscript topic | Evidence directory | Code |
|---|---|---|
| RT-IoT2022 matched window comparison | `CBID_31617_Exact_Common10_Audit/`, `CBID_31617_DeepSets_5Seeds/` | root RT scripts and `CBID_31617_DeepSets_5Seeds_Scripts/` |
| TCAG development | `CBID_TCAG_V9/` | `cbid_tcag_v9_*` |
| Frozen detector results | `CBID_TCAG_V10/` | `cbid_tcag_v10_*` |
| Final detection and APRE | `CBID_FINAL_AUDIT/10_final_confirmation/`, `CBID_APRE_V1/` | `cbid_final_ef_confirm.py`, `cbid_apre_v1_*` |
| EGRE bootstrap | `CBID_FINAL_AUDIT/07_temporal_bootstrap/`, `11_final_bootstrap/` | `historical_recovered/` (both bootstrap candidates); `historical_egre/` is an earlier procedure |
| Computation | `CBID_FINAL_AUDIT/06_computational_cost/` | `historical_recovered/` (benchmark candidate) |
| RA-APRE historical analyses | `CBID_RA_APRE_V2/` | `historical_recovered/` (dynamic and cumulative budget candidates); early drift, agreement and forecast scripts still to recover |
| Closed-loop replay | `CBID_EXP11_1/` | `cbid_exp11_closed_loop.py` |
| Risk forecasting | `CBID_EXP11_2/` | `cbid_exp11_2_risk.py` (corrected executed version) |
| Exploratory policy grid | `CBID_EXP11_3/` | `cbid_exp11_3_grid.py` |
| Fixed transfer | `CBID_EXP11_4/` | `cbid_exp11_4_transfer.py`, `exp11_4_freeze.json` |
| No-action diagnosis | `CBID_EXP12_1/` | `cbid_exp12_1_diagnostic.py` |
| Adaptive policies | `CBID_EXP12_2/` | `cbid_exp12_2_adaptive.py` |
| Adaptive transfer | `CBID_EXP12_3/` | `cbid_exp12_3_transfer.py`, `exp12_3_freeze.json` |

`figures/` contains historical source fragments; the manuscript's current figures are maintained in its separate LaTeX project. The original historical archive manifests and redaction records are retained in `docs/`. The seven new experiment archives were extracted into their original directory layout. Their archive digests are in `docs/EXP11_EXP12_ARCHIVES.json`.

## Quick checks
Use Python 3.12.3; exact reported experiment versions are in `ENVIRONMENT.json`. `requirements.txt` is not a full lock; the reported CUDA build is PyTorch 2.9.0+cu130. Install the appropriate build using the official PyTorch instructions. Export the complete actual environment from the experiment machine before release.

```bash
python verify_release.py
sha256sum -c SHA256SUMS.txt
# After installing the experimental dependencies:
python cbid_exp12_1_diagnostic.py --self-test
python cbid_exp12_3_transfer.py --self-test
```

The first command checks 37 frozen detector artifacts, six risk-estimator bundles, freeze/source hashes, 218 archived policy rows, 18 diagnoses and Python syntax. Synthetic tests are not a full rerun. Only deserialize trusted model artifacts in an isolated environment; never load untrusted joblib/pickle files.

## Datasets and reproducing runs
Obtain [RT-IoT2022](https://doi.org/10.24432/C5P338) from UCI and [NF-ToN-IoT-v3](https://doi.org/10.48610/44d7c5e) from its provider; consult their terms. Dataset payloads are not bundled. The expected NF-ToN-IoT-v3 source archive SHA-256 is `b3b6256e970a8986d87716edea4bfd436d68f04058bf5c26cf67e6d839c83698`.

See `docs/REPRODUCTION.md` for missing inputs, checksums and commands. Existing experimental folders are evidence; always write new results to a separate output path. Historical paths are preserved to keep freeze checks valid. The E/F detection runner expects a dedicated user's `~/Downloads` layout. Do not overwrite existing experimental results.

## Release status and limits
Five terminal-history candidates have been recovered and statically reviewed (see `docs/RECOVERED_SCRIPTS_AUDIT.md`); missing opening delimiters prevent byte-for-byte certification against the executed historical scripts. The complete original CBID training source, early RA-APRE drift/agreement/forecast scripts, exact input/prediction CSVs and a complete environment lock are not bundled or fully established by this package. The author reported a local environment lock and verified five input hashes on the original workstation. No full raw-data rerun has been performed here. E/F prevention results are retrospective; their diagnostics informed adaptive design. No cross-site or live IPS validation is claimed. Source and result archives should be reviewed by the authors for license and redistribution clearance before becoming public. See `docs/RELEASE_CHECKLIST.md`. A new Git commit does not prove the historic freeze dates.
