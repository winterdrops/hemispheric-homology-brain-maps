"""Hemispheric geometry: homologue indices, bilateral correlation and sphere-frame alignment."""
from __future__ import annotations

import numpy as np

_REFLECT = np.diag([-1.0, 1.0, 1.0])


def homologue_index(names, left=("L_", "lh_", "lh.", "Left_", "LH_"), right=("R_", "rh_", "rh.", "Right_", "RH_")):
    """Index of each region's contralateral homologue, matched by name after stripping a hemisphere prefix.

    Parameters
    ----------
    names : sequence of str
        Region names such as ``"L_precuneus"`` / ``"R_precuneus"`` or ``"lh_precuneus"``.

    Returns
    -------
    hom : (n,) int array, ``hom[i]`` is the index of the homologue of region ``i``.
    hemi : (n,) array of ``"L"``/``"R"``.
    """
    names = list(names)
    hemi, base = [], []
    for n in names:
        for pre in left:
            if n.startswith(pre):
                hemi.append("L")
                base.append(n[len(pre):])
                break
        else:
            for pre in right:
                if n.startswith(pre):
                    hemi.append("R")
                    base.append(n[len(pre):])
                    break
            else:
                raise ValueError(f"cannot find a hemisphere prefix in {n!r}")
    lookup = {(h, b): i for i, (h, b) in enumerate(zip(hemi, base))}
    hom = np.empty(len(names), int)
    for i, (h, b) in enumerate(zip(hemi, base)):
        other = "R" if h == "L" else "L"
        if (other, b) not in lookup:
            raise ValueError(f"no homologue for {names[i]!r}")
        hom[i] = lookup[(other, b)]
    return hom, np.array(hemi)


def pair_groups(homologue):
    """Group label per region such that homologues share a label (use for GroupKFold / nested tuning)."""
    homologue = np.asarray(homologue)
    return np.minimum(np.arange(len(homologue)), homologue)


def bilateral_r(y, homologue, hemi):
    """Pearson correlation between left-hemisphere values and their right homologues.

    ``y`` may be 1-D (one map) or 2-D (maps in columns); returns a float or an array."""
    y = np.asarray(y, float)
    L = np.where(np.asarray(hemi) == "L")[0]
    a, b = y[L], y[np.asarray(homologue)[L]]
    a = a - a.mean(0)
    b = b - b.mean(0)
    r = (a * b).sum(0) / np.sqrt((a**2).sum(0) * (b**2).sum(0))
    return float(r) if np.ndim(r) == 0 else r


def homologue_angles(sphere_coords, homologue, hemi):
    """Angle (degrees) between each left sphere centroid and the mirror image (x -> -x) of its right homologue.

    Mirrored-rotation spin tests assume these angles are ~0. On fsaverage spheres they are ~14 degrees for
    Desikan-Killiany and Glasser parcels; on fs_LR spheres they are ~0-2 degrees."""
    c = np.asarray(sphere_coords, float)
    L = np.where(np.asarray(hemi) == "L")[0]
    a = c[L] / np.linalg.norm(c[L], axis=1, keepdims=True)
    b = c[np.asarray(homologue)[L]] @ _REFLECT
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1.0, 1.0)))


def align_hemispheres(sphere_coords, homologue, hemi):
    """Rigidly rotate right-hemisphere sphere centroids (Kabsch) onto the mirror image of their left homologues.

    Returns a copy of the coordinates (unit length) in which mirrored rotations preserve bilateral structure."""
    c = np.asarray(sphere_coords, float).copy()
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    hemi = np.asarray(hemi)
    homologue = np.asarray(homologue)
    L = np.where(hemi == "L")[0]
    R = np.where(hemi == "R")[0]
    A = c[homologue[L]]
    B = c[L] @ _REFLECT
    U, _, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    rot = U @ D @ Vt
    c[R] = c[R] @ rot
    c[R] /= np.linalg.norm(c[R], axis=1, keepdims=True)
    return c
