"""hemisafe - homology-aware cross-validation and spatial null models for parcellated brain maps.

Brain maps are near-duplicated across hemispheres. hemisafe provides
  * splitters that keep homologous regions on the same side of every train/test split
    (HomologueKFold, BilateralSpatialKFold, DistanceDependentCV with ``homologue=``),
  * spin nulls that preserve bilateral structure (spin_symmetric; spin(..., align=True)), a wrapper that makes any
    surrogate generator symmetry-preserving (symmetric_surrogates) and a diagnostic that checks whether a null does
    (check_bilateral),
  * permutation tests that re-run the complete CV pipeline for every null draw (cv_permutation_test).
"""
from .geometry import homologue_index, pair_groups, bilateral_r, homologue_angles, align_hemispheres
from .nulls import shuffle, spin, spin_symmetric, symmetric_surrogates, check_bilateral
from .cv import HomologueKFold, BilateralSpatialKFold, DistanceDependentCV, homologue_leakage
from .inference import perm_pvalue, corr_test, nested_cv_predict, cv_permutation_test

__version__ = "0.2.0"
__all__ = ["homologue_index", "pair_groups", "bilateral_r", "homologue_angles", "align_hemispheres", "shuffle",
           "spin", "spin_symmetric", "symmetric_surrogates", "check_bilateral", "HomologueKFold", "BilateralSpatialKFold",
           "DistanceDependentCV", "homologue_leakage", "perm_pvalue", "corr_test", "nested_cv_predict", "cv_permutation_test"]
