# Pre-specified addition: generative null models (written before any analysis was run)

Rationale. Reviewers will ask whether generative nulls — variogram matching (BrainSMASH) and Moran spectral
randomization (MSR) — share the problem. Inspection of the neuromaps source code (v0.0.7, `nulls._make_surrogates`)
shows that for surface data both are generated separately for each hemisphere; with `seed=None` (the default) the two
hemispheres receive independent random numbers, and with a fixed integer seed they receive the same random numbers.

Variants (parcellated data, 34 regions per hemisphere, homologous order):
- BrainSMASH `Base` with neuromaps settings (default parameters, batch size 50): (i) independent per hemisphere
  (geodesic distances), (ii) shared seed per hemisphere, (iii) whole brain with Euclidean centroid distances.
- MSR (BrainSpace `MoranRandomization`, neuromaps settings: inverse distance weights with unit diagonal, joint=True,
  tol=1e-6, singleton procedure): (i) independent per hemisphere, (ii) shared seed per hemisphere.
- Proposed remedy ("symmetric"): apply the generator separately to the homologous sum and difference maps,
  s = (x_L + x_R)/2 and a = (x_L - x_R)/2, on the mean of the two hemispheric distance matrices, and recombine
  x_L* = s* + a*, x_R* = s* - a*; this preserves the bilateral covariance.

Hypotheses.
- H6a: independent variants produce null maps without bilateral structure (median absolute bilateral correlation of
  the surrogates < 0.2 for the 12 ENIGMA maps).
- H6b: at rho = 0.9, independent variants are liberal for map-to-map correlations (whole 95% CI above 7.5%) and more
  liberal than the conventional spin.
- H6c: symmetric variants have false-positive rates within the 2.5-7.5% band or within 2 percentage points of the
  symmetric spin on the same maps.
- H6d: in cross-validation (200 maps), independent variants are liberal under leaky schemes (R, Sw), and symmetric
  variants under SH and DD-H have false-positive rates <= 7.5%.
- Control: false-positive rates at rho = 0 are reported for every variant (calibration of the generators themselves).

Design. Map-to-map correlations: first 500 maps of the main simulation (lambda = 77 mm, rho = 0.9; the same maps as
in the main analysis) and first 250 maps of the rho = 0 condition; 500 draws per null; 28 annotations. ENIGMA data:
336 correlations, 10,000 draws per null, BH-FDR and Holm. Cross-validation: first 200 maps of the main lambda = 77 mm
simulation; schemes R, Sw, SH and DD-H; ridge (group-matched tuning) and kNN; 500 draws; independent and symmetric
variants of BrainSMASH and MSR. Bilateral structure: 1,000 surrogates of each of the 12 ENIGMA maps.

Decision rules. All results are reported whether or not they support the hypotheses. Classification as in the main
analysis: Bradley band with map-level bootstrap CIs and t-tests against 5% (map-to-map correlations); Wilson CIs and
exact binomial tests (cross-validation).
