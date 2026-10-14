import numpy as np
import pytest
from sklearn.linear_model import Ridge

import hemisafe as hs


def rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 30
    lh = rng.normal(size=(n, 3))
    lh[:, 0] = -np.abs(lh[:, 0])            # left hemisphere at x < 0
    lh /= np.linalg.norm(lh, axis=1, keepdims=True)
    rh = (lh * [-1, 1, 1]) @ rot_z(15).T    # mirrored, then offset by 15 degrees (fsaverage-like)
    coords = np.vstack([lh, rh])
    names = [f"L_r{i}" for i in range(n)] + [f"R_r{i}" for i in range(n)]
    hom, hemi = hs.homologue_index(names)
    y = np.r_[lh[:, 1] + 0.3 * lh[:, 2], lh[:, 1] + 0.3 * lh[:, 2]] + 0.05 * rng.normal(size=2 * n)
    return coords, hom, hemi, y


def test_homologue_index(toy):
    _, hom, hemi, _ = toy
    assert (hom[hom] == np.arange(len(hom))).all() and (hemi[:30] == "L").all() and (hemi[30:] == "R").all()


def test_alignment_removes_offset(toy):
    coords, hom, hemi, _ = toy
    assert hs.homologue_angles(coords, hom, hemi).mean() > 10
    assert hs.homologue_angles(hs.align_hemispheres(coords, hom, hemi), hom, hemi).mean() < 1e-6


def test_symmetric_spin_preserves_bilateral_structure(toy):
    coords, hom, hemi, y = toy
    P = hs.spin_symmetric(coords, hemi, hom, n_perm=50, seed=1)
    for p in P:
        assert sorted(p[:30]) == list(range(30)) and sorted(p[30:]) == list(range(30, 60))
    d = hs.check_bilateral(y, P, hom, hemi)
    assert abs(d["null_median"] - d["observed"]) < 1e-12


def test_conventional_spin_loses_and_alignment_restores(toy):
    coords, hom, hemi, y = toy
    conv = hs.check_bilateral(y, hs.spin(coords, hemi, 200, seed=2), hom, hemi)
    ali = hs.check_bilateral(y, hs.spin(coords, hemi, 200, seed=2, homologue=hom, align=True), hom, hemi)
    assert ali["null_median"] > conv["null_median"]


def test_splitters_keep_homologues_together(toy):
    coords, hom, hemi, _ = toy
    for cv in [hs.HomologueKFold(hom, n_splits=5, random_state=0),
               hs.BilateralSpatialKFold(coords, hom, hemi, n_splits=5, n_init=10)]:
        splits = list(cv.split(coords))
        assert hs.homologue_leakage(splits, hom) == 0.0
        assert sorted(np.concatenate([te for _, te in splits])) == list(range(60))
    dd = list(hs.DistanceDependentCV(coords, 0.75).split())
    ddh = list(hs.DistanceDependentCV(coords, 0.75, homologue=hom, hemi=hemi).split())
    assert len(dd) == 60 and len(ddh) == 30 and hs.homologue_leakage(ddh, hom) == 0.0


def test_perm_pvalue():
    assert hs.perm_pvalue(1.0, np.zeros(99), "greater") == pytest.approx(0.01)
    assert hs.perm_pvalue(0.5, np.array([-0.6, 0.1, 0.2]), "two-sided") == pytest.approx(0.5)


def test_corr_and_cv_tests_run(toy):
    coords, hom, hemi, y = toy
    X = np.c_[coords[:, 1], coords[:, 2]]
    P = hs.spin_symmetric(coords, hemi, hom, n_perm=20, seed=3)
    r, p, null = hs.corr_test(X[:, 0], y, P)
    assert -1 <= r <= 1 and 0 < p <= 1 and null.shape == (20,)
    obs, p, null = hs.cv_permutation_test(Ridge(alpha=1.0), X, y, hs.HomologueKFold(hom, 5, random_state=0), P)
    assert np.isfinite(obs) and 0 < p <= 1 and null.shape == (20,)


def test_nested_tuning_respects_groups(toy):
    coords, hom, hemi, y = toy
    X = np.c_[coords[:, 1], coords[:, 2], coords[:, 0] ** 2]
    outer = hs.HomologueKFold(hom, 5, random_state=0)
    splits = list(outer.split(X))
    p = hs.nested_cv_predict(Ridge(), X, y, splits, param_grid={"alpha": [0.01, 1, 100]},
                             inner_groups=hs.pair_groups(hom))
    assert np.isfinite(p).all()
    obs, pv, null = hs.cv_permutation_test(Ridge(), X, y, outer, hs.spin_symmetric(coords, hemi, hom, 5, seed=0),
                                           param_grid={"alpha": [0.1, 10]}, inner_groups=hs.pair_groups(hom))
    assert np.isfinite(obs) and null.shape == (5,)


def test_symmetric_surrogates_preserve_bilateral_structure(toy):
    _, hom, hemi, y = toy

    def perm_generator(x, n, seed):         # any univariate generator; here random permutations of the input
        rng = np.random.default_rng(seed)
        return np.stack([rng.permutation(x) for _ in range(n)])
    S = hs.symmetric_surrogates(y, hom, hemi, perm_generator, n_perm=500, seed=3)
    assert S.shape == (500, len(y))
    chk = hs.check_bilateral(y, S, hom, hemi)
    assert abs(chk["null_median"] - chk["observed"]) < 0.1
    L = np.where(hemi == "L")[0]            # independent permutations of each hemisphere destroy the structure
    ind = np.array(y[None, :].repeat(500, 0))
    rng = np.random.default_rng(4)
    for k in range(500):
        ind[k, L] = rng.permutation(y[L])
        ind[k, hom[L]] = rng.permutation(y[hom[L]])
    assert abs(hs.check_bilateral(y, ind, hom, hemi)["null_median"]) < 0.2
    r, p, null = hs.corr_test(y, y, S)
    assert null.shape == (500,) and 0 < p <= 1
