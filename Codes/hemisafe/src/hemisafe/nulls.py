"""Spatial null models for parcellated brain maps that preserve (or deliberately test) bilateral structure.

Spin generators return an ``(n_perm, n_regions)`` integer array ``P``; ``y[P[k]]`` is the k-th null map.
``symmetric_surrogates`` returns surrogate values instead (float array of the same shape).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import align_hemispheres, bilateral_r

_REFLECT = np.diag([-1.0, 1.0, 1.0])


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _assign(c, c_rot):
    cost = np.linalg.norm(c[:, None, :] - c_rot[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return rows, cols


def shuffle(n_regions, n_perm=1000, seed=None):
    """Uniform random permutations (ignores all spatial structure; not a valid null for brain maps)."""
    rng = np.random.default_rng(seed)
    return np.array([rng.permutation(n_regions) for _ in range(n_perm)])


def spin(sphere_coords, hemi, n_perm=1000, seed=None, homologue=None, align=False):
    """Conventional parcel-level spin (Alexander-Bloch et al. 2018; Hungarian re-assignment as in Vasa et al. 2018).

    The left hemisphere is rotated at random and the right hemisphere receives the mirror-image rotation. This
    preserves bilateral structure only if the two hemispheres' sphere frames are mirror images (true for fs_LR, not
    for fsaverage). With ``align=True`` (requires ``homologue``) the right-hemisphere centroids are first rigidly
    aligned to the mirrored left hemisphere, which removes the fsaverage frame offset."""
    c = np.asarray(sphere_coords, float)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    hemi = np.asarray(hemi)
    if align:
        if homologue is None:
            raise ValueError("align=True requires homologue")
        c = align_hemispheres(c, homologue, hemi)
    rng = np.random.default_rng(seed)
    idx = {h: np.where(hemi == h)[0] for h in ("L", "R")}
    out = np.zeros((n_perm, len(c)), int)
    for k in range(n_perm):
        rot = _random_rotation(rng)
        rots = {"L": rot, "R": _REFLECT @ rot @ _REFLECT}
        for h in ("L", "R"):
            cc = c[idx[h]]
            rows, cols = _assign(cc, cc @ rots[h].T)
            out[k, idx[h][rows]] = idx[h][cols]
    return out


def spin_symmetric(sphere_coords, hemi, homologue, n_perm=1000, seed=None):
    """Symmetric spin: rotate and re-assign the left hemisphere only and give the right hemisphere the homologous
    re-assignment. Every null map keeps the original map's bilateral (homologue) structure exactly."""
    c = np.asarray(sphere_coords, float)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    hemi = np.asarray(hemi)
    homologue = np.asarray(homologue)
    L = np.where(hemi == "L")[0]
    R = np.where(hemi == "R")[0]
    rng = np.random.default_rng(seed)
    out = np.zeros((n_perm, len(c)), int)
    cl = c[L]
    for k in range(n_perm):
        rot = _random_rotation(rng)
        rows, cols = _assign(cl, cl @ rot.T)
        out[k, L[rows]] = L[cols]
        out[k, R] = homologue[out[k, homologue[R]]]
    return out


def _null_maps(y, perms):
    """(n_perm, n) null maps from permutation indices (integer array) or from surrogate values (float array)."""
    P = np.asarray(perms)
    return P.astype(float) if np.issubdtype(P.dtype, np.floating) else np.asarray(y, float)[P]


def symmetric_surrogates(y, homologue, hemi, generator, n_perm=1000, seed=None):
    """Make any surrogate generator symmetry-preserving (for example BrainSMASH, Moran spectral randomization or
    geometric eigenmodes).

    Generators applied separately to each hemisphere with independent random numbers produce null maps without
    bilateral structure. Here the generator is applied to the homologous sum and difference maps,
    s = (y_L + y_R) / 2 and a = (y_L - y_R) / 2 (in left-hemisphere order), and the surrogates are recombined as
    y_L* = s* + a* and y_R* = s* - a*, which preserves the bilateral covariance of ``y``.

    ``generator(x, n_perm, seed)`` must return an ``(n_perm, len(x))`` array of surrogates of the left-hemisphere-
    ordered map ``x``, for example ``lambda x, n, s: brainsmash.mapgen.base.Base(x, D, seed=s)(n, 50)`` with ``D`` a
    distance matrix of the left-hemisphere regions (or the mean of the two hemispheric matrices).
    Returns an ``(n_perm, n_regions)`` float array of surrogate maps (values, not indices), which ``corr_test`` and
    ``check_bilateral`` accept in place of permutation indices."""
    y = np.asarray(y, float)
    hemi = np.asarray(hemi)
    hom = np.asarray(homologue)
    L = np.where(hemi == "L")[0]
    R = hom[L]
    s, a = (y[L] + y[R]) / 2, (y[L] - y[R]) / 2
    s1, s2 = (int(v) for v in np.random.default_rng(seed).integers(0, 2 ** 31 - 1, 2))
    S = np.asarray(generator(s, n_perm, s1), float).reshape(n_perm, -1)
    if np.std(a) > 1e-12 * max(np.std(s), 1e-12):
        A = np.asarray(generator(a, n_perm, s2), float).reshape(n_perm, -1)
    else:                                   # exactly symmetric map (e.g. hemisphere-averaged effects)
        A = np.zeros_like(S)
    out = np.empty((n_perm, len(y)))
    out[:, L], out[:, R] = S + A, S - A
    return out


def check_bilateral(y, perms, homologue, hemi):
    """Diagnostic: bilateral r of the observed map and the median / 5th percentile across its null maps.

    ``perms`` are permutation indices or surrogate maps (float array, n_perm x n). A null whose median is far below
    the observed value does not preserve bilateral structure and will be liberal whenever homologues carry shared
    signal."""
    y = np.asarray(y, float)
    b = bilateral_r(_null_maps(y, perms).T, homologue, hemi)
    return {"observed": bilateral_r(y, homologue, hemi), "null_median": float(np.median(b)),
            "null_q05": float(np.quantile(b, 0.05))}
