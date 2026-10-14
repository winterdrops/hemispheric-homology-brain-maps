# Pre-specified analysis plan — Spatial and homologue leakage in machine-learning models of brain maps

**Original recorded plan date:** 2026-09-28T18:08Z (before any predictive-performance statistic was computed on real or simulated targets)
**Editorial version (2026-10-12):** Descriptive metadata has been corrected. The scientific plan below is unchanged. The original file SHA-256 was `7b7d25375bc43f0bf4a89cef94a6b120ebfbfd20aa94db9dcc555e9df3a039d5`; the checksum of this editorial version is recorded in `SHA256SUMS.txt`. The date above refers to the original plan, not to this edited file.

What had already been done when this was frozen: data retrieval and parcellation alignment; parity check against Hansen et al. (2022), with an exact match; descriptive spatial statistics of the maps (correlograms, Moran's I, bilateral correlations); a symmetry check of the spin null models; speed benchmarks on random data. **No model was fitted to any ENIGMA target map.** No simulated null map was analysed.

## 1. Background and aim
Machine-learning (ML) models are increasingly trained across cortical regions, treating each region as a sample. They predict disorder-related maps (e.g. ENIGMA case–control cortical-thickness effect sizes) from multimodal brain annotations such as neurotransmitter receptors, connectivity and oscillatory power.

Regions are not independent observations, for two reasons:
- **Spatial autocorrelation:** neighbouring regions have similar values.
- **Hemispheric homology:** a region and its contralateral homologue have near-identical values in both predictors and targets. In three ENIGMA maps (ADHD, ASD, conduct disorder), the two hemispheres are exactly equal.

The study quantifies how these two dependencies affect:
- (i) cross-validated (CV) performance estimates;
- (ii) the false-positive rate of permutation tests.

It does this with simulations and with 12 ENIGMA disorder maps.

## 2. Hypotheses
- **H1 (inflation).** Simulated null targets are independent of the predictors by construction. For them, mean out-of-sample R² will be larger under random CV (R) than under spatial-plus-homologue blocked CV (SH), for every model. The R–SH gap will be larger for kNN and random forest than for ridge.
- **H2 (homology vs proximity).** Main simulation, bilateral coupling ρ = 0.9. For kNN and RF, the homology component [R²(R) − R²(H)] will exceed the proximity component [R²(H) − R²(SH)].
- **H3 (calibration).** Main simulation.
  - (a) With random CV, shuffle-permutation tests at α = 0.05 will have a false-positive rate (FPR) > 0.10 for kNN and RF (95% Wilson CI lower bound > 0.05).
  - (b) The symmetric spin test (spin-sym) will have FPR ≤ 0.075 (point estimate) for every model and CV scheme.
- **H4 (spin variant).** Under random CV, at ρ ∈ {0.9, 1.0}, the conventional mirrored-rotation spin (spin-std) will have a higher FPR than spin-sym.
- **H5 (real data).** For each model, fewer ENIGMA maps will be FDR-significant (q < 0.05) under (SH, spin-sym) than under (R, shuffle).

**Exploratory, no directional hypotheses:**
- Which disorder maps remain predictable under (SH, spin-sym)?
- Power and bias of R² under a positive-control simulation.
- A receptor-only predictor set (for comparability with Hansen et al., 2022).
- Sensitivity to the random-partition seed.

## 3. Data (all retrieved from public GitHub repositories; commits in `data/raw/PROVENANCE.md`)
- **Parcellation:** Desikan–Killiany, 68 cortical regions, ENIGMA order.
- **Targets (primary set, 12 maps).** One broadest case–control cortical-thickness Cohen's d map per condition, from the ENIGMA Toolbox. Maps: 22q11.2 deletion syndrome; ADHD (all ages); ASD (meta-analysis); epilepsy (all); MDD (adults); OCD (adults); schizophrenia; bipolar disorder (adults); Parkinson's disease (all vs controls); anorexia nervosa; conduct disorder / antisocial behaviour; clinical high risk for psychosis (CHR vs controls).
  - Excluded: schizotypy (not a case–control contrast) and nested subtype maps (GGE, TLE-L, TLE-R).
- **Predictors (28), from sources independent of the target cohorts:**
  - 19 PET receptor/transporter density maps (Hansen et al., 2022; Lausanne scale033 mapped to DK by region name).
  - 6 MEG power bands (HCP).
  - Structural and functional connectivity strength (HCP group matrices, ENIGMA Toolbox).
  - Mesulam laminar-differentiation class (ordinal 1–4).

## 4. Models (hyper-parameters fixed a priori)
Features are standardised inside each training fold.
- **Ridge regression.** α ∈ logspace(−2, 4, 25), chosen per training fold by exact leave-one-out error.
- **k-nearest-neighbour regression.** k = 5, uniform weights, Euclidean distance.
- **Random forest.** 100 trees, max_features = 1/3, min_samples_leaf = 3, bootstrap, random_state = 0.

## 5. Cross-validation schemes (10 folds; seed 2026)
The four schemes form a 2 × 2 factorial design (homologue grouping × spatial blocking).
- **R:** random 10-fold over regions.
- **H:** random 10-fold over the 34 homologous pairs; homologues always share a fold.
- **S:** k-means (k = 10) on surface centroids (x, y, z).
- **SH:** k-means (k = 10) on mirrored pair centroids (|x|, y, z); homologues share a fold.

**Metric:** R²_cv = 1 − SSE/SST, computed on pooled out-of-fold predictions. Negative values are allowed. Pearson r is reported as secondary.

## 6. Null models and inference
- **shuffle:** uniform random permutation of the target across regions.
- **spin-std:** parcel spins with the right-hemisphere rotation mirrored from the left, and Hungarian re-assignment per hemisphere (Alexander-Bloch et al., 2018; Váša et al., 2018).
- **spin-sym:** left-hemisphere spin plus homologous re-assignment of the right hemisphere. This preserves bilateral structure exactly.
  - Design check: spin-std reduced the bilateral correlation of the target maps to a median of about 0.35–0.64 (originals 0.71–1.00). spin-sym preserved it.
  - On this basis, spin-sym is the **primary** null for the real-data analysis and spin-std the conventional comparator.

**p-values:** one-sided (greater), (1 + #null ≥ observed)/(1 + N) (Phipson & Smyth, 2010). Benjamini–Hochberg FDR across the 12 maps within each model × scheme × null.

**Number of null draws:**

| Model | Real data | Simulation |
|---|---|---|
| Ridge, kNN | 5,000 per null | 500 per map |
| RF | 250 per null, schemes R and SH only (compute limit: 2 CPU cores) | see §7 |

## 7. Simulations
All simulated maps are generated independently of the predictors.

- **B1 (main null).** 1,000 Gaussian random-field (GRF) maps at parcel level.
  - Within-hemisphere covariance exp(−d_geo/λ), with λ = 50 mm. This is the exponential fit to the empirical within-hemisphere correlogram of the 12 targets (51.5 mm), rounded.
  - Bilateral coupling ρ = 0.9 (median bilateral r of the targets = 0.91).
  - Ridge and kNN: all schemes, three null types with 500 draws each.
  - RF:
    - observed R² on the first 500 maps;
    - shuffle FPR against a common shuffle null (500 shuffles of standard-normal vectors per scheme; valid because GRF marginals are Gaussian and shuffling destroys spatial structure);
    - spin-std FPR on 50 maps × 100 spins for R and SH.
- **B2 (sensitivity grid).** 300 maps per condition, ridge and kNN, schemes R and SH (all four schemes for R²). Conditions:
  - ρ ∈ {0, 0.5, 0.9, 1.0} at λ = 50;
  - λ ∈ {25, 100} at ρ = 0.9.
- **B3 (alternative generator).** BrainSMASH variogram-matched surrogates (Burt et al., 2020).
  - 50 surrogates of each target's left hemisphere.
  - Right hemisphere = ρ̂ × mirrored left surrogate + sqrt(1 − ρ̂²) × independent right surrogate, where ρ̂ is that target's bilateral r.
  - Ridge and kNN.
- **B4 (positive control).** y = sqrt(s)·z(X_k β) + sqrt(1 − s)·e.
  - X_k = 3 predictors drawn at random; β ~ N(0, 1); e ~ GRF(λ = 50, ρ = 0.9).
  - s ∈ {0.1, 0.3, 0.5}; 300 maps each; ridge and kNN.
  - Outcomes: power under each null (200 draws) and the bias of R²_cv relative to s.

## 8. Decision rules
- **H1:** paired bootstrap 95% CI (10,000 resamples of maps) for the mean R²(R) − R²(SH) excludes 0 for each model. Supported if the kNN and RF gaps exceed the ridge gap with non-overlapping CIs.
- **H2:** 95% bootstrap CI of the difference between the two components excludes 0.
- **H3, H4:** Wilson 95% CIs for FPR, applied as stated in §2.
- **H5:** count comparison per model; descriptive (no test).

## 9. Deviations
Any deviation from this plan will be listed, with reasons, in the manuscript section *Deviations from preregistration*.

## 10. Software
Python 3.11.15, NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.2, scikit-learn 1.8.0, BrainSMASH 0.11.0, nibabel 5.4.2.
