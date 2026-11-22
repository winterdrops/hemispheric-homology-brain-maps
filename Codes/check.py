"""
check.py - correctness and reproducibility checks.

  tests      unit tests of the analysis library (about 1 minute): split-based and fold-based models agree; the
             distance-dependent splits are correct; the ENIGMA Toolbox port reproduces the toolbox source exactly; the
             vectorised P values equal a brute-force loop; closed-form leave-one-group-out errors equal explicit
             refits; out-of-fold ridge and kNN predictions equal scikit-learn refits; the hemisafe package
             reproduces the tuned ridge predictions; sphere-frame misalignment; the 34-region reduction and the
             Glasser geometry; Monte Carlo ties
  quick      fast reproducibility check (a few minutes): recomputes a subset of the archived results from scratch and
             compares them with the result files in results/ (seeds are fixed, so values must agree to numerical
             precision). Rewrites the ENIGMA map-correlation result files (s3_map_correlations.py real) in place.
  hemisafe   tests of the hemisafe package (pytest)

Usage: python check.py [tests] [quick] [hemisafe]   (default: tests and quick)
Exit status 0 if every check passed.
"""
import ast
import contextlib
import io
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import homology_lib as hl

warnings.filterwarnings("ignore")
CODE = Path(__file__).resolve().parent
ok = []


def check(name, cond, detail=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


# ======================================================================================================================
# unit tests
# ======================================================================================================================
def tests():
    X, Y, reg, geo = hl.load_data()
    Xv, Yv = X.to_numpy(float), Y.to_numpy(float)

    # split-based models reproduce the fold-based implementations
    for s in ["R", "H", "SH"]:
        f = hl.make_folds(s, reg)
        sp = hl.folds_to_splits(f)
        g = hl.inner_groups(s, reg, f)
        for name, P1, preds in [("ridge LOO", hl.ridge_oof(Xv, Yv, f), hl.ridge_split_predict(Xv, Yv, sp)),
                                ("ridge grouped", hl.ridge_oof(Xv, Yv, f, inner=g),
                                 hl.ridge_split_predict(Xv, Yv, sp, inner=g)),
                                ("kNN", hl.knn_oof(Xv, Yv, f), hl.knn_split_predict(Xv, Yv, sp))]:
            P2 = np.zeros_like(Yv)
            for (tr, te), P in zip(sp, preds):
                P2[te] = P
            check(f"{name} split == fold ({s})", np.allclose(P1, P2, atol=1e-10))

    # distance-dependent splits
    dd = hl.dd_splits(reg)
    check("DD: 68 splits, 51 train / 17 test, source in train",
          len(dd) == 68 and all(len(a) == 51 and len(b) == 17 and i in a for i, (a, b) in enumerate(dd)))
    check("DD: train/test disjoint and exhaustive",
          all(len(set(a) & set(b)) == 0 and len(set(a) | set(b)) == 68 for a, b in dd))
    ddh = hl.ddh_splits(reg)
    check("DD-H: 34 splits, 50/18, homologues never split",
          len(ddh) == 34 and all(len(a) == 50 and len(b) == 18 for a, b in ddh) and hl.homologue_in_training(reg, ddh) == 0)
    print("   homologue leakage: DD %.3f  DD-H %.3f  R %.3f  Sw %.3f  SH %.3f" % (
        hl.homologue_in_training(reg, dd), hl.homologue_in_training(reg, ddh),
        hl.homologue_in_training(reg, hl.folds_to_splits(hl.make_folds("R", reg, seed=0))),
        hl.homologue_in_training(reg, hl.folds_to_splits(hl.make_folds_within(reg))),
        hl.homologue_in_training(reg, hl.folds_to_splits(hl.make_folds("SH", reg)))))

    # the ENIGMA Toolbox port matches the toolbox source exactly
    src = (hl.RAW_ENIGMA / "enigmatoolbox_permutation_testing.py").read_text(encoding="utf-8")
    ns = {"np": np, "cdist": cdist, "pd": pd, "warnings": warnings}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("rotate_parcellation", "perm_sphere_p"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "enigma", "exec"), ns)
    c = hl.sphere_centroids(reg, "fsa5", normalise=False)
    np.random.seed(7)
    with contextlib.redirect_stdout(io.StringIO()):
        ref = ns["rotate_parcellation"](c[:34].copy(), c[34:].copy(), 25).astype(int)
    np.random.seed(7)
    check("ENIGMA rotate_parcellation port identical to toolbox",
          np.array_equal(ref, hl.enigma_rotate_parcellation(c[:34].copy(), c[34:].copy(), 25)))
    pr, pm = [], []
    for i, j in [(4, 3), (0, 10), (7, 20), (11, 25), (2, 5), (9, 27)]:
        x, y = Yv[:, i], Xv[:, j]
        with contextlib.redirect_stdout(io.StringIO()):
            pr.append(ns["perm_sphere_p"](x, y, ref))
        pm.append(hl.enigma_pvalue(x, y[:, None], ref.T)[0])
    check(f"ENIGMA perm_sphere_p port identical on 6 pairs (p = {np.round(pr, 3)})", np.allclose(pr, pm, atol=1e-12))

    # ENIGMA centroid extraction: label-file centroids equal the toolbox's .annot-based centroids
    cents = []
    for h in ["lh", "rh"]:
        sph = nib.load(hl.RAW_ENIGMA / f"fsa5_sphere_{h}.gii").darrays[0].data.astype(float)
        labels, ctab, names = nib.freesurfer.io.read_annot(hl.RAW_ENIGMA / f"fsa5_{h}_aparc.annot", orig_ids=True)
        for ic in range(ctab.shape[0]):
            if names[ic].decode("utf-8") not in ("unknown", "corpus", "corpuscallosum", "medialwall",
                                                 "Background+FreeSurfer_Defined_Medial_Wall"):
                cents.append(sph[labels == ctab[ic, -1]].mean(0))
    cents = np.array(cents)
    check(f"ENIGMA .annot centroids == label-file centroids ({cents.shape[0]} parcels)",
          cents.shape == c.shape and np.allclose(cents, c, atol=1e-6))

    # vectorised two-sided P values equal a brute-force loop
    rng = np.random.default_rng(0)
    perms = np.array([rng.permutation(68) for _ in range(99)])
    _, p = hl.corr_pvalues(Xv[:, :3], Yv[:, :4], perms, chunk=3)
    bf = np.zeros_like(p)
    for i in range(3):
        for j in range(4):
            rn = np.array([np.corrcoef(Xv[:, i], Yv[pp, j])[0, 1] for pp in perms])
            bf[i, j] = (1 + np.sum(np.abs(rn) >= np.abs(np.corrcoef(Xv[:, i], Yv[:, j])[0, 1]) - 1e-12)) / 100
    check("corr_pvalues == brute force", np.allclose(p, bf))

    # sphere-frame misalignment; null permutations are valid
    a5 = hl.homologue_angles(reg, hl.sphere_centroids(reg, "fsa5"))
    a69 = hl.homologue_angles(reg, hl.sphere_centroids(reg, "conte69"))
    check(f"misalignment fsa5 {a5.mean():.1f} deg > 10, conte69 {a69.mean():.1f} deg < 3", a5.mean() > 10 and a69.mean() < 3)
    for name in ["spin_std", "spin_fslr", "spin_enigma", "spin_aligned", "spin_sym"]:
        P = hl.null_permutations(reg, 20, {name: 1})[name]
        check(f"{name}: permutations within hemisphere",
              all(sorted(p[:34]) == list(range(34)) and sorted(p[34:]) == list(range(34, 68)) for p in P))
    Psym = hl.spin_permutations_sym(reg, 20, seed=1)
    hom = reg["homologue_idx"].to_numpy()
    check("symmetric spin moves homologous pairs together", all(np.array_equal(p[hom], hom[p]) for p in Psym))

    # regression metrics; closed-form leave-one-group-out errors equal explicit refits
    m = hl.regression_metrics(Yv[:, 0], Yv[:, 0] * 0.5 + 0.1)
    check("calibration slope of y on 0.5y+0.1 is 2", abs(m["calib_slope"] - 2) < 1e-9)
    from sklearn.linear_model import Ridge
    f = hl.make_folds("SH", reg)
    tr = np.where(f != 0)[0]
    g = hl.inner_groups("SH", reg, f)[tr]
    Xtr_s, _ = hl._standardise(Xv[tr], Xv[tr])
    ytr = Yv[tr, 0]
    a = 10.0
    U, s_, Vt = np.linalg.svd(Xtr_s, full_matrices=False)
    d_ = s_**2 / (s_**2 + a)
    H = (U * d_) @ U.T + 1.0 / len(tr)
    resid = (ytr - ytr.mean()) - U @ (d_ * (U.T @ (ytr - ytr.mean())))
    maxdiff = 0.0
    for gg in np.unique(g):
        G_ = np.where(g == gg)[0]
        e_closed = np.linalg.solve(np.eye(len(G_)) - H[np.ix_(G_, G_)], resid[G_])
        keep = np.setdiff1d(np.arange(len(tr)), G_)
        mod = Ridge(alpha=a, fit_intercept=True).fit(Xtr_s[keep], ytr[keep])
        maxdiff = max(maxdiff, np.max(np.abs((ytr[G_] - mod.predict(Xtr_s[G_])) - e_closed)))
    check(f"closed-form leave-one-group-out errors == explicit refits (max diff {maxdiff:.1e})", maxdiff < 1e-10)

    # vectorised out-of-fold predictions equal scikit-learn refits (every penalty of the grid; kNN with k = 5)
    from sklearn.neighbors import KNeighborsRegressor
    dr = dk = 0.0
    for sch in ("R", "H", "S", "SH"):
        f = hl.make_folds(sch, reg)
        Pk = hl.knn_oof(Xv, Yv[:, :3], f)
        for fo in np.unique(f):
            te_, tr_ = f == fo, f != fo
            Xtr_, Xte_ = hl._standardise(Xv[tr_], Xv[te_])
            ref = KNeighborsRegressor(n_neighbors=hl.K_NEIGHBOURS).fit(Xtr_, Yv[tr_, :3]).predict(Xte_)
            dk = max(dk, float(np.max(np.abs(Pk[te_] - ref))))
        for a in hl.ALPHAS:
            Pr = hl.ridge_oof(Xv, Yv[:, :3], f, alphas=[a])
            for fo in np.unique(f):
                te_, tr_ = f == fo, f != fo
                Xtr_, Xte_ = hl._standardise(Xv[tr_], Xv[te_])
                ref = Ridge(alpha=a, fit_intercept=True).fit(Xtr_, Yv[tr_, :3]).predict(Xte_)
                dr = max(dr, float(np.max(np.abs(Pr[te_] - ref))))
    check(f"ridge out-of-fold predictions == scikit-learn Ridge, whole grid (max diff {dr:.1e})", dr < 1e-8)
    check(f"kNN out-of-fold predictions == scikit-learn KNeighborsRegressor (max diff {dk:.1e})", dk < 1e-10)

    # the hemisafe package reproduces the article's tuned ridge predictions (matched inner folds)
    sys.path.insert(0, str(CODE / "hemisafe" / "src"))
    import hemisafe as hs
    dh = 0.0
    for sch in ("R", "H", "SH"):
        f = hl.make_folds(sch, reg)
        inner_ = hl.inner_groups(sch, reg, f)
        ref = hl.ridge_oof(Xv, Yv[:, :2], f, inner=inner_)
        for j in range(2):
            got = hs.nested_cv_predict(Ridge(), Xv, Yv[:, j], hl.folds_to_splits(f), {"alpha": hl.ALPHAS},
                                       inner_groups=inner_, standardize=True)
            dh = max(dh, float(np.max(np.abs(got - ref[:, j]))))
    check(f"hemisafe nested_cv_predict == analysis code, tuned ridge (max diff {dh:.1e})", dh < 1e-10)

    # one value per homologous pair; Glasser geometry (fine-parcellation simulations)
    r34, g34, f = hl.half_brain(reg, geo, "avg")
    L = np.where(reg["hemi"].to_numpy() == "L")[0]
    check("homologue average: 34 regions, mean of each homologous pair, left-hemisphere geometry",
          len(r34) == 34 and np.allclose(f(Yv), (Yv[L] + Yv[hom[L]]) / 2) and np.array_equal(g34, geo[np.ix_(L, L)])
          and np.array_equal(hl.half_brain(reg, geo, "left")[2](Yv), Yv[L]))
    if (hl.DER / "glasser_regions.csv").exists():
        rg, gg = hl.load_glasser()
        mis = pd.read_csv(hl.RES / "v3_misalignment.csv").set_index(["sphere", "parcellation"])
        ag = hl.homologue_angles(rg, rg[["sph_x", "sph_y", "sph_z"]].to_numpy()).mean()
        hg = rg["homologue_idx"].to_numpy()
        check(f"Glasser: 360 regions, 180 pairs, misalignment {ag:.2f} deg as in Supplementary Table 7",
              len(rg) == 360 and np.array_equal(hg[hg], np.arange(360))
              and abs(ag - mis.loc[("fsaverage5", "Glasser (360)"), "mean_deg"]) < 1e-6
              and np.isfinite(gg[:180, :180]).all() and np.allclose(gg[:180, :180], gg[:180, :180].T, atol=1e-6))
        Pg = hl.spin_permutations_sym(rg, 10, seed=1)
        check("Glasser: symmetric spin moves homologous pairs together", all(np.array_equal(p[hg], hg[p]) for p in Pg))

    # Monte Carlo P value: null values equal to the observed one up to floating-point noise count as ties
    o = 0.1 + 0.2   # 0.30000000000000004
    check("perm_pvalue counts floating-point ties on any platform",
          hl.perm_pvalue(o, np.array([0.3, o, np.nextafter(o, 0), 0.29])) == 4 / 5
          and hl.perm_pvalue(0.3, np.array([0.31, 0.2])) == 2 / 3)


# ======================================================================================================================
# quick reproducibility check against the archived result files
# ======================================================================================================================
def quick():
    import s3_map_correlations as s3
    import s4_cv_simulations as s4
    import s5_enigma as s5
    res = hl.RES
    print(f"Python {platform.python_version()} on {platform.platform()}; numpy {np.__version__}, "
          f"pandas {pd.__version__}", flush=True)
    X, Y, reg, geo = hl.load_data()

    # geometry
    a = hl.homologue_angles(reg, hl.sphere_centroids(reg, "fsa5")).mean()
    mis = pd.read_csv(res / "v3_misalignment.csv")
    check(f"fsaverage5 DK misalignment reproduced ({a:.2f} deg)", abs(a - mis.iloc[0]["mean_deg"]) < 1e-6)

    # cross-validation null simulation: first 3 maps at lambda = 77 mm, all schemes, models and nulls
    M = s4.null_maps("lam77")
    nulls = s4.cv_nulls()
    new = pd.DataFrame([r for j in range(3) for r in s4.null_map_rows(j, M[:, j], "lam77", nulls)])
    new = new.set_index(["map", "model", "scheme"]).sort_index()
    arch = pd.concat([pd.read_csv(f) for f in res.glob("v3_cv_sim_lam77_*.csv")])
    old = arch.set_index(["map", "model", "scheme"]).loc[new.index].sort_index()
    cols = [c for c in new.columns if c != "cond"]
    diff = np.nanmax(np.abs(new[cols].to_numpy(float) - old[cols].to_numpy(float)))
    check("CV null simulation, maps 0-2 (statistics, P values, null summaries identical)",
          np.allclose(new[cols].to_numpy(float), old[cols].to_numpy(float), atol=1e-10), f"max |difference| {diff:.1e}")

    # ENIGMA map-to-map correlations: the step rewrites its result files, so the archived counts are read first
    before = pd.read_csv(res / "v3_bivariate_real_counts.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        s3.real(joint=False)
    after = pd.read_csv(res / "v3_bivariate_real_counts.csv")
    num = [c for c in before.columns if pd.api.types.is_numeric_dtype(before[c])]
    same = before.shape == after.shape and np.allclose(before[num].to_numpy(float), after[num].to_numpy(float))
    check("ENIGMA map-to-map correlations: counts of significant tests reproduced (all nulls)", same)

    # ENIGMA reanalysis: MDD, ridge, SH and DD-H, symmetric spin
    k = list(Y.columns).index("MDD")
    arch = pd.read_csv(res / "v3_real_cv.csv")
    arch = arch[(arch["set"] == "multimodal") & (arch["model"] == "ridge") & (arch["map"] == "MDD")]
    perms = hl.spin_permutations_sym(reg, 2000, seed=21005)
    for s in ["SH", "DD-H"]:
        st = s5.statistic(s5.Yv, s, s5.predict("ridge", s5.Xv, s5.Yv, s))[k]
        Yn = s5.Yv[:, k][perms].T
        p = hl.perm_pvalue(st, s5.statistic(Yn, s, s5.predict("ridge", s5.Xv, Yn, s)))
        a_ = arch[arch["scheme"] == s].iloc[0]
        check(f"ENIGMA MDD ridge {s}: statistic and symmetric-spin P reproduced",
              abs(st - a_["stat"]) < 1e-10 and abs(p - a_["p_spin_sym"]) < 1e-12,
              f"|difference| statistic {abs(st - a_['stat']):.1e}, P {abs(p - a_['p_spin_sym']):.1e}")

    # generative null models: self-tests and the P values of simulated map 0 (all seven variants)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            s3.gen_selftest()
        check("generative nulls: self-tests (bilateral structure of independent and symmetric variants)", True)
    except AssertionError as e:
        check("generative nulls: self-tests", False, str(e))
    ref = res / "v3_genpart_biv_lam77_rho0.9_0_250.csv"
    ref = ref if ref.exists() else res / "v3_gen_biv_map0_reference.csv"
    if ref.exists():
        archg = pd.read_csv(ref)
        gn = hl.GenerativeNulls(reg, geo)
        G0 = hl.grf_maps(reg, geo, lam=77, rho=0.9, n_maps=1000, seed=1701)[:, 0]
        ndiff = []
        for kk, v in enumerate(gn.VARIANTS):
            _, pv = hl.corr_pvalues_surrogates(s3.Xv, G0, gn.surrogates(G0, v, 500, 29100000 + 10 * kk))
            old = archg[(archg["map"] == 0) & (archg["null"] == v)].set_index("annotation").loc[list(X.columns), "p"]
            if not np.allclose(pv, old.to_numpy(), atol=1e-12):
                ndiff.append(f"{v}: {int((~np.isclose(pv, old.to_numpy(), atol=1e-12)).sum())} P values differ")
        check("generative nulls: P values of simulated map 0 reproduced (all seven variants)", not ndiff, "; ".join(ndiff))
    else:
        print("SKIP generative-null P values of map 0 (archived parts not found)")


def hemisafe():
    # test the package source in this folder (src layout), not an installed copy or the Codes/hemisafe directory
    pkg = CODE / "hemisafe"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(pkg / "src"), os.environ.get("PYTHONPATH", "")])}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=pkg, env=env)
    check("hemisafe package tests (pytest)", r.returncode == 0)


STEPS = {"tests": tests, "quick": quick, "hemisafe": hemisafe}

if __name__ == "__main__":
    t0 = time.time()
    for step in (sys.argv[1:] or ["tests", "quick"]):
        if step not in STEPS:
            raise SystemExit(__doc__)
        print(f"--- {step}", flush=True)
        STEPS[step]()
    print(f"{sum(ok)}/{len(ok)} checks passed in {time.time() - t0:.0f} s")
    sys.exit(0 if all(ok) else 1)
