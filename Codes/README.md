# Code for "Ignoring hemispheric homology miscalibrates parcel-level null models and cross-validation of brain maps: a
simulation study and reanalysis"

This folder reproduces every number, figure and table of the article and its Additional files (Supplementary
Information, Supplementary Data 1) from public raw data. All random seeds are fixed, so a rerun gives the archived results (within floating-point tolerance).

## Layout

```
<project>/
├── Codes/                       this folder
│   ├── homology_lib.py          shared library: CV schemes, models, null models, generative nulls, simulators, metrics
│   ├── s1_data.py               step 1  data, geometry, correlation length, standard errors, effective number of tests
│   ├── s2_prespecified.py       step 2  analyses of the initial (pre-specified) plan and the first revision
│   ├── s3_map_correlations.py   step 3  tests of map-to-map correlations (simulated and ENIGMA maps, generative nulls)
│   ├── s4_cv_simulations.py     step 4  simulations of cross-validated prediction (null, positive control, ...)
│   ├── s5_enigma.py             step 5  reanalysis of the 12 ENIGMA maps (main, sensitivity, random forests)
│   ├── s6_statistics.py         step 6  formal tests and summary tables
│   ├── s7_figures.py            step 7  all figures, the Source Data workbook and Supplementary Data 1 (Supplementary
│   │                                    Tables 1-16)
│   ├── run_all.py               runs steps 1-7 in order (parallel where possible)
│   ├── check.py                 unit tests and a quick reproducibility check against the archived results
│   ├── compare_results.py       compares regenerated result files with an archived copy
│   ├── hemisafe/                stand-alone Python package with the remedies (MIT)
│   ├── fonts/                   Liberation Sans for the figures (SIL Open Font License 1.1)
│   ├── requirements.txt, LICENSE
├── plan/                        pre-specified analysis plan (preregistration.md, frozen 2026-09-28) and the plan of
│                                the generative-null analyses (prereg_generative_nulls.md); SHA256SUMS.txt
├── data/raw/                    unmodified public inputs (PROVENANCE.md, SHA256SUMS.txt)
├── data/derived/                analysis-ready matrices (written by step 1)
├── results/                     result tables (CSV/JSON/NPZ)
├── figures/                     Fig1-Fig5, figures/supp/SuppFig1-14, Source_Data.xlsx, Supplementary_Data_1.xlsx;
│                                figures/extra/: two figures kept in the archive only
└── logs/                        one log per step, run_all_progress.log, run_all_timing.csv
```

Every step script is run as `python <script> STEP [arguments]`; `python <script>` without arguments prints the list of
steps. The project folder is the parent of `Codes/`; set the environment variable `HEMIHOM_ROOT` to run the code on
another copy (for example a clean folder that contains only `data/raw/`).

## Installation

Python 3.11 with the pinned versions in `requirements.txt`:

```bash
python -m venv .venv
.venv/bin/pip install -r Codes/requirements.txt      # Windows: .venv\Scripts\pip install -r Codes\requirements.txt
.venv/bin/pip install -e Codes/hemisafe              # optional: the remedies as a package
```

## Running

```bash
python Codes/check.py tests                      # unit tests (about 1 minute)
python Codes/check.py quick                      # recompute a subset and compare with results/ (a few minutes)
python Codes/run_all.py --workers 4              # everything (about 8 hours with 4 workers)
python Codes/compare_results.py --ref <untouched copy of the archived project>
```

`run_all.py` writes a log per step to `logs/`, stops at the first failing step and can resume with `--start N`
(0 data, 1 pre-specified, 2 revision, 3 main analyses, 4 figures); `--dry-run` lists the steps. Long jobs resume where
they stopped when `HEMIHOM_RESUME=1`.

## Where each result comes from

Numbering of the article (BMC Neuroscience version). Supplementary Tables 1-16 are the sheets of
`figures/Supplementary_Data_1.xlsx`; Supplementary Figs 1-14 are `figures/supp/SuppFig<N>_*`.

| Item | Computed by (step) | Statistics (s6) / result file | Drawn by |
|---|---|---|---|
| Fig. 1a, c (bilateral correlation of maps and null maps) | s1 `geometry` | `v3_bilateral_null.csv` | s7 `fig1` |
| Fig. 1b, Suppl. Table 7 (sphere frames) | s1 `geometry` | `v3_misalignment.csv` | s7 `fig1` |
| Fig. 2a-b, Suppl. Tables 8a-b, Suppl. Fig. 6 | s3 `sim` | `main`: `v3_stats_bivariate_fpr.csv`, `v3_stats_bivariate_contrasts.csv` | s7 `fig2`, `supp` |
| Generative null models (Fig. 2a, Fig. 5a, Suppl. Tables 4a-c) | s3 `gen_*`, s4 `gen_cv` | `generative`: `v3_stats_gen_*.csv`, `v3_gen_real_counts.csv` | s7 `fig2`, `fig5` |
| Suppl. Table 2a (approximate correspondence, correlations) | s3 `approx` | `v3_approx_coupling_biv.csv` | |
| Suppl. Table 2b (approximate correspondence, cross-validation) | s4 `approx` | `main`: `v3_stats_approx_coupling_cv.csv` | |
| Fig. 3a, homologue leakage in Table 3 | s1 `geometry` | `v3_scheme_structure.csv` | s7 `fig3` |
| Fig. 3b-c, Fig. 4a-c, Suppl. Tables 9a-c, 10a-c, Suppl. Fig. 7 | s4 `cv` | `main`: `v3_stats_cv_*.csv` | s7 `fig3`, `fig4`, `supp` |
| Suppl. Fig. 2 (inner folds) | s2 `B1`, `rev_sim` | `sim_B1_linear.csv`, `rev_sim_B1.csv` | s7 `supp` |
| Suppl. Tables 3a-b (published distance-dependent analysis) | s4 `dd_ols` | `main`: `v3_stats_dd_ols.csv`; `v3_dd_ols_real.csv` | |
| Fig. 4d, Suppl. Figs 3 and 10, Suppl. Tables 11a-d | s4 `power`, `estimand` | `main`: `v3_stats_auc*.csv`, `v3_stats_roc_curves.csv`, `v3_stats_confusion.csv`, `v3_stats_estimation.csv`, `v3_estimand.csv` | s7 `fig4`, `supp` |
| Table 4, Suppl. Tables 5a-c (in-sample test) | s4 `insample` | `insample`: `v3_stats_insample_*.csv`; `v3_insample_*.csv` | s7 `supp_data` |
| Fig. 5a, Suppl. Table 8c | s3 `real` | `v3_bivariate_real_counts.csv`, `v3_bivariate_joint_null.csv` | s7 `fig5` |
| Fig. 5b-d, Suppl. Tables 12a-c, Suppl. Fig. 4a | s5 `cv` | `main`: `v3_stats_real_counts.csv`; `v3_real_cv.csv`, `v3_real_metrics.csv` | s7 `fig5`, `supp` |
| Suppl. Figs 4b and 5, Suppl. Tables 13a-c | s5 `hyper`, `noise`, `rf`, `rf_merge` | `v3_sens_hyper.csv`, `v3_sens_alpha.json`, `v3_sens_measurement.csv`, `v3_rf_main.csv`, `v3_rf_alt.csv` | s7 `supp` |
| Suppl. Table 1 (the 12 disorder maps) | s1 `prepare` | `table1_targets.csv` | s7 `supp_data` |
| Suppl. Table 14, Suppl. Note 4 (standard errors) | s1 `se` | `v3_target_se_sources.json`, `v3_target_se.csv` | |
| Suppl. Fig. 1 (correlation length) | s1 `lambda` | `rev_lambda_calibration.csv`, `rev_lambda_summary.csv` | s7 `supp` |
| Suppl. Figs 11-14, Suppl. Note 3, Suppl. Table 6 (pre-specified analyses and verdicts) | s2 (all steps) | `table_*.csv`, `summary.json`, `rev_*.csv`, `rev_summary.json`, `exploratory_*.csv` | s7 `prespecified`, `supp_data` |
| Effective number of target maps | s1 `meff` | `v3_meff.json` | |
| One value per homologous pair (34 regions): Suppl. Fig. 9, Suppl. Tables 15a-e | s3 `half`, s4 `half_cv`, s5 `half` | `extensions`: `v3_stats_half_*.csv` | s7 `supp` |
| Glasser parcellation (360 regions): Suppl. Fig. 8, Suppl. Tables 16a-b | s1 `glasser`, s3 `glasser`, s4 `glasser_cv` | `extensions`: `v3_stats_glasser_*.csv` | s7 `supp` |
| Supplementary Data 1 (Suppl. Tables 1-16, one sheet per table) | the steps above | the result files above | s7 `supp_data` |
| Archive-only figures (`figures/extra/`) | s3 `sim`; s2 `partition` | `v3_bivariate_sim.csv.gz`; `exploratory_partition_robustness.csv` | s7 `supp` |

## Licences

Code: MIT (`LICENSE`). Fonts: SIL Open Font License 1.1 (`fonts/LICENSE_OFL.txt`). Data: see
`data/raw/PROVENANCE.md`; derived files that include the receptor atlas are CC BY-NC-SA 4.0.
