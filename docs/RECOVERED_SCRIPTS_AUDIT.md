# Audit of terminal-history candidates (27 November 2026)

Source: author-provided CBID_RECOVERED_SCRIPTS_V2.zip. The five source files and the recovery report are preserved without modification. Each candidate begins with imports, ends with the expected output reporting, parses as Python and matches its SHA-256 in recovery_report.json. The opening shell delimiters were not recovered; there is no contemporaneous source hash establishing identity to the executed script. These are plausible recovered source candidates, not certified originals.

| Candidate | Lines | Archived consistency check |
|---|---:|---|
| Dynamic budget | 155 | 378 metric rows; 126 grouped rows recomputed, all equal within floating-point tolerance; budget flags agree |
| Cumulative budget | 188 | 648 metric rows; 216 grouped rows recomputed, all equal within floating-point tolerance; budget flags agree |
| Computational benchmark | 426 | 10,800 raw timing rows; 54 seed-level and 18 aggregated rows recomputed and matched |
| Exploratory temporal bootstrap | 343 | 36 observed AP contrasts matched the archived model metrics |
| Final bootstrap | 373 | 24 observed AP contrasts matched the archived model metrics |

Both weighted-AP implementations were separately extracted from their ASTs and checked against scikit-learn weighted AP on synthetic tied scores and block weights (100 random cases attempted per implementation; single-class cases skipped). No historical script was run end-to-end. No bootstrap interval was recomputed from the original predictions. No new training or latency run was performed. Limited pattern scan found no obvious GitHub tokens, AWS access-key identifiers or PEM private-key markers in the five scripts; author privacy and rights review is still required.

## Running the recovered code

The scripts retain the historical `Path.home() / "Downloads"` root and fixed output paths with `exist_ok=True`. They can overwrite archived outputs if run in the original environment. **Do not execute them in the original workstation layout.** For a reproduction, use an isolated account/container with its own HOME/Downloads, restore required inputs and dependencies, copy the needed root modules, and run the desired candidate there. Compare metric counts, contrasts and interval bounds with the archived results; runtime values are hardware dependent. Do not change a historical source silently. Record any adaptation separately.

The budget and exploratory bootstrap candidates need C/D prediction CSVs. The benchmark needs C features, frozen detector artifacts and the matching TCAG modules. The final bootstrap needs the original protocol and E/F prediction CSVs. The candidate archive is not proof of complete raw-data reproduction. The early RA-APRE drift, agreement and forecasting source scripts are not among these five candidates.
