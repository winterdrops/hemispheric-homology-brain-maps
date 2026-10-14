# hemisafe

Homology-aware cross-validation and symmetry-preserving spatial nulls for parcellated brain maps.

Most brain maps are near-duplicated across hemispheres: a region and its contralateral homologue carry almost the same value. Two common practices ignore this.

- **Spin tests on fsaverage spheres.** They rotate the right hemisphere by the mirror image of the left-hemisphere rotation. The fsaverage left and right sphere frames are not mirror images (about 14° apart for Desikan–Killiany and Glasser parcels), so the "mirrored" rotation sends homologues to different places. The null maps lose their bilateral structure and the test becomes liberal.
- **Cross-validation that separates homologues.** Random, within-hemisphere and distance-dependent splits let a test region's homologue sit in the training set. This inflates cross-validated performance and permutation tests.

`hemisafe` provides:

| Function / class | Purpose |
|---|---|
| `homologue_index(names)` | Homologue index from region names (`L_`/`R_`, `lh_`/`rh_` prefixes) |
| `homologue_angles(coords, hom, hemi)` | Diagnose sphere-frame misalignment (should be ~0°) |
| `align_hemispheres(coords, hom, hemi)` | Rigid (Kabsch) alignment of the right-hemisphere sphere frame |
| `spin(coords, hemi, n, align=True, homologue=hom)` | Conventional parcel spin (Hungarian), optionally on aligned frames |
| `spin_symmetric(coords, hemi, hom, n)` | Symmetric spin: preserves each map's bilateral structure exactly |
| `symmetric_surrogates(y, hom, hemi, generator, n)` | Makes any surrogate generator (BrainSMASH, Moran spectral randomization, eigenmodes) symmetry-preserving by applying it to the homologous sum and difference maps |
| `check_bilateral(y, perms, hom, hemi)` | Does a null keep the map's bilateral correlation? (accepts permutation indices or surrogate maps) |
| `HomologueKFold`, `BilateralSpatialKFold` | scikit-learn splitters that keep homologues in the same fold |
| `DistanceDependentCV(coords, homologue=hom, hemi=hemi)` | Distance-dependent CV, optionally homologue-aware |
| `homologue_leakage(splits, hom)` | Share of test regions whose homologue is in the training set |
| `corr_test(x, y, perms)` | Two-sided spin/permutation test of a map correlation |
| `nested_cv_predict(est, X, y, splits, param_grid, inner_groups)` | Out-of-fold predictions with tuning that respects homologues/blocks |
| `cv_permutation_test(est, X, y, cv, perms, param_grid=..., inner_groups=...)` | Re-runs the whole CV pipeline (including tuning) for every null draw |

Generators such as BrainSMASH or Moran spectral randomization are usually applied to each hemisphere separately; with
independent random numbers for the two hemispheres (the neuromaps default) their null maps have no bilateral
structure. Wrap them with `symmetric_surrogates`:

```python
from brainsmash.mapgen.base import Base
L = np.where(hemi == "L")[0]
D = (dist[np.ix_(L, L)] + dist[np.ix_(hom[L], hom[L])]) / 2      # mean of the two hemispheric distance matrices
S = hs.symmetric_surrogates(disorder_map, hom, hemi, lambda x, n, s: Base(x, D, seed=s)(n, 50), n_perm=10000, seed=1)
r, p, _ = hs.corr_test(annotation_map, disorder_map, S)          # surrogate maps instead of permutation indices
```

## Install

```bash
pip install .            # from this folder (Python >= 3.9; numpy, scipy, scikit-learn)
python -m pytest tests   # 9 tests
```

## Quick start

```python
import numpy as np
import hemisafe as hs
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

hom, hemi = hs.homologue_index(region_names)          # e.g. ENIGMA / Desikan-Killiany names ("L_...", "R_...")
print(hs.homologue_angles(sphere_centroids, hom, hemi).mean())   # ~14 deg on fsaverage spheres, ~0-2 deg on fs_LR

# 1) Map-to-map correlation with a symmetry-preserving spin (two-sided p-value)
perms = hs.spin_symmetric(sphere_centroids, hemi, hom, n_perm=10000, seed=1)
print(hs.check_bilateral(disorder_map, perms, hom, hemi))        # null maps keep the bilateral r exactly
r, p, _ = hs.corr_test(annotation_map, disorder_map, perms)

# 2) Cross-validated prediction: homologue-aware spatial folds, nested tuning that keeps homologues together,
#    and a spin test that re-runs the whole pipeline for every null map
outer = hs.BilateralSpatialKFold(surface_centroids, hom, hemi, n_splits=10)
model = make_pipeline(StandardScaler(), Ridge())
grid = {"ridge__alpha": np.logspace(-2, 4, 25)}
obs, p, null = hs.cv_permutation_test(model, X, disorder_map, outer, perms[:1000], param_grid=grid,
                                      inner_groups=outer.folds())   # inner loop leaves out whole outer blocks
```

A complete, runnable example with the 12 ENIGMA maps and 28 annotations of the article is in `examples/example_enigma.py`.

Use `inner_groups=hs.pair_groups(hom)` with `HomologueKFold`. For distance-dependent CV (overlapping test sets), compute per-split metrics from `DistanceDependentCV(..., homologue=hom, hemi=hemi).split()`.

## Tests

The unit tests in `tests/` compare the implementations with scikit-learn and with explicit reference computations
(`python -m pytest tests`).

## Citation

If you use `hemisafe`, please cite the accompanying article (see `CITATION.cff` in the study archive).

## Licence

MIT.
