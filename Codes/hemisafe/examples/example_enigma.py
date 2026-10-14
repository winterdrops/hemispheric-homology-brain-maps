"""hemisafe example: the 12 ENIGMA cortical-thickness maps and 28 annotations used in the accompanying article.

Run from anywhere, pointing to the root of the study archive (the folder that contains data/derived/):
    python example_enigma.py /path/to/spatialcv_v3
Runtime: a few minutes on a laptop (100 null draws and a coarse penalty grid for the cross-validation test; use
at least 1,000 draws and a finer grid in practice).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import hemisafe as hs

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
reg = pd.read_csv(root / "data/derived/regions.csv")
Y = pd.read_csv(root / "data/derived/targets_primary.csv", index_col=0).loc[reg["name"]]
Xdf = pd.read_csv(root / "data/derived/predictors.csv", index_col=0).loc[reg["name"]]
X = Xdf.to_numpy()
hom, hemi = hs.homologue_index(reg["name"])
sphere = reg[["sph_x", "sph_y", "sph_z"]].to_numpy()   # fsaverage5 sphere centroids
surface = reg[["x", "y", "z"]].to_numpy()              # surface centroids (for spatial blocks)

# 1. Are the hemispheres' sphere frames mirror images? (should be close to 0 degrees)
ang = hs.homologue_angles(sphere, hom, hemi)
print(f"1. fsaverage5: homologous centroids deviate from mirror symmetry by {ang.mean():.1f} deg on average")
print(f"   after rigid alignment: {hs.homologue_angles(hs.align_hemispheres(sphere, hom, hemi), hom, hemi).mean():.1f} deg")

# 2. Do the null maps keep the bilateral structure of the data?
y = Y["OCD"].to_numpy()                                  # obsessive-compulsive disorder map
conv = hs.spin(sphere, hemi, n_perm=1000, seed=1)       # conventional spin on fsaverage5 (reflected rotation)
sym = hs.spin_symmetric(sphere, hemi, hom, n_perm=1000, seed=1)
print("2. bilateral r (observed / median of null maps):")
for name, perms in [("conventional spin", conv), ("symmetric spin", sym)]:
    b = hs.check_bilateral(y, perms, hom, hemi)
    print(f"   {name:18s} {b['observed']:.2f} / {b['null_median']:.2f}")

# 3. Map-to-map correlation with a two-sided spin test
x = Xdf["PET_5HT2a"].to_numpy()                          # serotonin 5-HT2a receptor density
for name, perms in [("conventional spin", conv), ("symmetric spin", sym)]:
    r, p, _ = hs.corr_test(x, y, perms)
    print(f"3. r = {r:.2f}, two-sided P = {p:.3f} ({name})")

# 4. Cross-validated prediction with homologue-aware spatial folds, nested tuning and a spin test of the whole pipeline
outer = hs.BilateralSpatialKFold(surface, hom, hemi, n_splits=10)
print(f"4. homologue leakage of the folds: {hs.homologue_leakage(list(outer.split(X)), hom):.0%}")
model = make_pipeline(StandardScaler(), Ridge())
grid = {"ridge__alpha": np.logspace(-2, 4, 7)}
obs, p, null = hs.cv_permutation_test(model, X, y, outer, sym[:100], param_grid=grid, inner_groups=outer.folds())
print(f"   R2_cv = {obs:.2f}, one-sided P = {p:.3f} (symmetric spin, 100 draws; use >= 1,000 in practice)")
