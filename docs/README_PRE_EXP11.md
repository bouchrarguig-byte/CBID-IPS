# CBID — IoT intrusion detection reproducibility package

**Private staging draft for IJ-AI manuscript 31617.** Original submission: 14 January 2026. This package consolidates archived experimental evidence and code; it is not yet a fully reproduced or publicly released software artifact.

## Scope
The original GATv2 investigation is retained through corrected RT-IoT2022 experiments. TCAG and EGRE extend relational evaluation; APRE and RA-APRE assess potential exposure through open-loop replay, not real network blocking. Both favorable and unfavorable results are preserved.

## Layout
Historical directory names are preserved because original scripts and manifests reference them. Root Python files are original archived scripts, not rewritten implementations. `CBID_TCAG_V10/04_frozen_models/` holds 37 verified artifacts plus manifest. `CBID_FINAL_AUDIT/` contains the final protocol, E/F confirmation metrics, computational benchmark and bootstrap summaries. `historical_egre/` holds the November 25 qualification implementation (not the later E/F procedure). `figures/` contains TikZ/PGFPlots source fragments and data.

## Evidence map
| Manuscript subject | Result directory / file | Code |
|---|---|---|
| RT-IoT2022 matched five-seed comparison | `CBID_31617_Exact_Common10_Audit/`, `CBID_31617_DeepSets_5Seeds/` | `cbid_rt2022_controlled_window_v2_1.py`, `cbid_31617_recompute_common10.py`, `CBID_31617_DeepSets_5Seeds_Scripts/` |
| Earlier NF-ToN-IoT-v3 relational controls | `CBID_NFToN_Validation_v6/`, `CBID_NFToN_Development_v7/`, `CBID_NFToN_EdgeAware_v8/` | root v6/v7/v8 scripts |
| TCAG features and development | `CBID_TCAG_V9/` | root `cbid_tcag_v9_*` |
| Frozen model and hybrid evaluation | `CBID_TCAG_V10/` | root `cbid_tcag_v10_*` |
| APRE | `CBID_APRE_V1/` | root `cbid_apre_v1_*` |
| RA-APRE and cumulative budget | `CBID_RA_APRE_V2/` | original scripts still to add |
| E/F confirmation | `CBID_FINAL_AUDIT/10_final_confirmation/` | `cbid_final_ef_confirm.py` |
| Bootstrap | `CBID_FINAL_AUDIT/07_temporal_bootstrap/`, `11_final_bootstrap/` | original terminal scripts still to add |
| Computational cost | `CBID_FINAL_AUDIT/06_computational_cost/` | original terminal script still to add |

## Reproduction environment
Reported: Python 3.12, PyTorch 2.9.0+cu130, PyG 2.7.0, XGBoost 3.4.1, pandas 2.3.3, NVIDIA Quadro RTX 5000. See `requirements.txt` (partial pins) and `docs/RELEASE_CHECKLIST.md`.

## Running historical scripts
Use a dedicated account or isolated workspace: the historical E/F runner expects scripts and results under `Path.home()/Downloads`. Preserve its layout and use a fresh output directory (it refuses overwriting confirmations). Existing result directories must be retained elsewhere before reproducing. The original runner additionally requires the exact C prediction CSV omitted from the supplied ZIP.

```bash
python cbid_rt2022_controlled_window_v2_1.py --help
python cbid_tcag_v9_02_temporal_features.py --help
python cbid_tcag_v10_04_confirmation.py --help
python cbid_final_ef_confirm.py --help
```

NF-ToN-IoT-v3 expected source archive SHA-256: `b3b6256e970a8986d87716edea4bfd436d68f04058bf5c26cf67e6d839c83698`. Obtain datasets from their original providers; they are not bundled. RT-IoT2022 is distributed through UCI.

## Integrity
`docs/integrity_report.json` records checks on the 37 frozen artifacts, four original model source files, policy scripts and E/F execution script. Local account paths in some non-frozen evidence files were redacted; see `docs/redacted_paths.json`. The original frozen manifests were preserved byte-for-byte. `SHA256SUMS.txt` describes this staging package, not the original experiment timeline.

## Limitations
No full rerun has been performed in the packaging environment. Large prediction and feature files are absent, and several terminal-only analysis scripts remain to be recovered. The original historical CBID source is not established by these archives. E/F independence is conditional on documented prior access. This repository is not evidence of effective prevention or universal graph superiority.

See the release checklist before publication.
