"""Cross-validation splitters for region-level prediction that keep hemispheric homologues on the same side.

All splitters follow the scikit-learn cross-validator interface (``split(X, y=None, groups=None)`` and
``get_n_splits``) and can be passed to ``cross_val_score`` or ``GridSearchCV``. HomologueKFold and
BilateralSpatialKFold partition the regions and can also be passed to ``cross_val_predict``; DistanceDependentCV
has overlapping test sets, so use it with per-split metrics, not ``cross_val_predict``.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.model_selection import BaseCrossValidator

from .geometry import pair_groups


class HomologueKFold(BaseCrossValidator):
    """Random K-fold over homologous pairs: a region and its homologue are always in the same fold ("H")."""

    def __init__(self, homologue, n_splits=10, random_state=None):
        self.homologue = np.asarray(homologue)
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _iter_test_masks(self, X=None, y=None, groups=None):
        g = pair_groups(self.homologue)
        pairs = np.unique(g)
        rng = np.random.default_rng(self.random_state)
        fold_of_pair = np.empty(len(pairs), int)
        fold_of_pair[rng.permutation(len(pairs))] = np.arange(len(pairs)) % self.n_splits
        lookup = dict(zip(pairs, fold_of_pair))
        folds = np.array([lookup[v] for v in g])
        for f in range(self.n_splits):
            yield folds == f


class BilateralSpatialKFold(BaseCrossValidator):
    """Spatial + homologue blocks ("SH"): k-means on mirrored pair centroids (|x|, y, z), so every test fold is a
    contiguous bilateral block. A test region's homologue is never in the training set; its spatial neighbours are
    in the training set only where they lie across a block border, so blocking reduces but does not remove
    leakage between neighbouring regions."""

    def __init__(self, coords, homologue, hemi, n_splits=10, random_state=2026, n_init=100):
        self.coords = np.asarray(coords, float)
        self.homologue = np.asarray(homologue)
        self.hemi = np.asarray(hemi)
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_init = n_init

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def folds(self):
        L = np.where(self.hemi == "L")[0]
        xyz = self.coords
        m = (np.c_[np.abs(xyz[L, 0]), xyz[L, 1:]] + np.c_[np.abs(xyz[self.homologue[L], 0]), xyz[self.homologue[L], 1:]]) / 2
        km = KMeans(n_clusters=self.n_splits, n_init=self.n_init, random_state=self.random_state).fit(m)
        folds = np.empty(len(xyz), int)
        folds[L] = km.labels_
        folds[self.homologue[L]] = km.labels_
        return folds

    def _iter_test_masks(self, X=None, y=None, groups=None):
        f = self.folds()
        for k in range(self.n_splits):
            yield f == k


class DistanceDependentCV(BaseCrossValidator):
    """Distance-dependent CV (as in Hansen et al. 2022): for every source region the ``train_frac`` closest regions
    are the training set and the rest the test set. Test sets overlap, so use it with per-split metrics, not
    ``cross_val_predict``. With ``homologue`` given, the construction is applied to homologous pairs using mirrored
    pair centroids, so both members of a pair are always on the same side (homologue-aware variant)."""

    def __init__(self, coords, train_frac=0.75, homologue=None, hemi=None):
        self.coords = np.asarray(coords, float)
        self.train_frac = train_frac
        self.homologue = None if homologue is None else np.asarray(homologue)
        self.hemi = None if hemi is None else np.asarray(hemi)

    def _units(self):
        if self.homologue is None:
            return self.coords, [np.array([i]) for i in range(len(self.coords))]
        if self.hemi is None:
            raise ValueError("hemi is required with homologue")
        L = np.where(self.hemi == "L")[0]
        xyz = self.coords
        m = (np.c_[np.abs(xyz[L, 0]), xyz[L, 1:]] + np.c_[np.abs(xyz[self.homologue[L], 0]), xyz[self.homologue[L], 1:]]) / 2
        return m, [np.array([i, self.homologue[i]]) for i in L]

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._units()[1])

    def split(self, X=None, y=None, groups=None):
        pts, members = self._units()
        D = cdist(pts, pts)
        ntr = int(np.floor(self.train_frac * len(pts)))
        for i in range(len(pts)):
            order = np.argsort(D[i], kind="stable")
            tr = np.sort(np.concatenate([members[j] for j in order[:ntr]]))
            te = np.sort(np.concatenate([members[j] for j in order[ntr:]]))
            yield tr, te


def homologue_leakage(splits, homologue):
    """Share of test regions whose homologue is in the training set (0 = homologue-safe)."""
    homologue = np.asarray(homologue)
    num = den = 0
    for tr, te in splits:
        s = set(np.asarray(tr).tolist())
        num += sum(int(homologue[t] in s) for t in te)
        den += len(te)
    return num / den
