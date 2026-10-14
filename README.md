# Ignoring hemispheric homology inflates false positives in spatial null models and cross-validation of brain maps

Code, data and results of the study by Yavuz Selim İşler (Department of Electrical and Electronics Engineering,
Faculty of Engineering and Natural Sciences, Osmaniye Korkut Ata University, Osmaniye, Türkiye).

A cortical region and its contralateral homologue are often near-duplicates. This repository shows how ignoring this
hemispheric homology makes spin tests of map-to-map correlations and cross-validated prediction across regions report
false positives, and evaluates remedies (symmetry-preserving nulls, homologue-aware spatial blocks). Every number,
figure and table of the article and its Supplementary Information can be reproduced from the public raw data.

## Contents

| Folder | Contents |
|---|---|
| `Codes/` | analysis code (steps 1-7), unit and reproducibility checks, the `hemisafe` package; see `Codes/README.md` |
| `plan/` | pre-specified analysis plan (frozen 2026-09-28) and the plan of the generative-null analyses, with checksums |
| `data/raw/` | unmodified public inputs, their provenance, licences and checksums |
| `data/derived/` | analysis-ready matrices written by step 1 |
| `results/` | all result tables (CSV, JSON, NPZ) |
| `figures/` | Figs 1-5, Supplementary Figs 1-13, Source Data and Supplementary Data 1 (Supplementary Tables 1-12) |

## Reproducing the results

Python 3.11 with the pinned packages in `Codes/requirements.txt`:

```bash
python -m venv .venv
.venv/bin/pip install -r Codes/requirements.txt        # Windows: .venv\Scripts\pip install -r Codes\requirements.txt
python Codes/check.py tests                            # unit tests (about 1 minute)
python Codes/check.py quick                            # recompute a subset and compare it with results/ (a few minutes)
python Codes/run_all.py --workers 4                    # everything from the raw data (about 8 hours)
```

All random seeds are fixed, so a rerun reproduces the result files to within floating-point precision;
`Codes/compare_results.py` compares a rerun with this archive. `Codes/README.md` lists, for every figure and table, the
step that computes it and the result file it reads.

## Licences

- Code (`Codes/`, including `hemisafe`): MIT (`LICENSE`).
- Fonts used for the figures (`Codes/fonts/`): SIL Open Font License 1.1.
- Input data (`data/raw/`): licences of the original sources (ENIGMA Toolbox; Hansen et al. receptor atlas, CC BY-NC-SA
  4.0), see `data/raw/PROVENANCE.md`. Derived data and results that incorporate the receptor atlas are shared under
  CC BY-NC-SA 4.0.

## Citation

Please cite the article (reference to be added on publication) and this archive; citation metadata are in
`CITATION.cff`.

## Contact

Yavuz Selim İşler, yavuzselimisler@osmaniye.edu.tr
