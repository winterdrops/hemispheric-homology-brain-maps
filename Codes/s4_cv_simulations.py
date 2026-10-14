"""
s4_cv_simulations.py - step 4: simulations of cross-validated prediction (Figs 3 and 4, Table 2B,
Supplementary Tables 3-6).

Target maps are Gaussian random fields independent of the 28 real predictors (or, in the positive control, with a signal
built from three of them). Models: ridge (inner tuning mirrors the outer scheme; DD uses leave-one-out tuning as in
common practice, DD-H leave-one-pair-out) and kNN (k = 5). Schemes: R and H (5 random partitions; mean statistic), S,
SH, Sw, DD and DD-H. Statistic tested: R2_cv (partition schemes) or the mean test-set Pearson r (DD, DD-H).

  cv COND [I0 I1]   null simulation: COND lam77 (500 new maps at the calibrated lambda) or lam50 (first 500 maps of
                    the pre-specified simulation B1); shuffle, conventional, fs_LR, aligned and symmetric spins (500 draws)
  power [LEVEL ...] positive control at lambda = 77 mm (s = 0.1, 0.3, 0.5 for LEVEL 0, 1, 2; 150 maps per level)
  estimand          replicate R2: a model fitted on a whole map, evaluated on an independent replicate (same signal,
                    new noise field); ridge tuned by leave-one-pair-out (main), by leave-one-out and in hindsight
  approx            cross-validation with approximate hemispheric correspondence (200 maps; R, Sw, SH, DD-H)
  dd_ols            re-implementation of the published distance-dependent analysis (least squares on the 19 receptor
                    maps, no tuning) on 500 null maps and on the 12 ENIGMA maps
  gen_cv I0 I1      cross-validation with generative nulls (first 200 maps of the lam77 simulation; R, Sw, SH, DD-H)
  half_cv HOW       one value per homologous pair (34 regions; HOW avg or left): null simulation and positive control
                    (R, S, DD; shuffle and a spin of the left sphere)
  glasser_cv I0 I1  fine parcellation (Glasser, 360 regions): null maps I0..I1-1 of 300 (R, H, S, SH, Sw; shuffle,
                    conventional and symmetric spins)
Long jobs resume where they stopped when HEMIHOM_RESUME=1.

Usage: python s4_cv_simulations.py STEP [arguments]
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

import homology_lib as hl

warnings.filterwarnings("ignore")
X, Y, reg, geo = hl.load_data()
Xv, Yv = X.to_numpy(float), Y.to_numpy(float)
Xs = (Xv - Xv.mean(0)) / Xv.std(0)
RES = hl.RES
RESUME = os.environ.get("HEMIHOM_RESUME") == "1"
PAIR = hl.pair_ids(reg)
PARTS = {s: hl.partitions(s, reg, 5) for s in ["R", "H", "S", "SH", "Sw"]}
SPLITS = {"DD": hl.dd_splits(reg), "DD-H": hl.ddh_splits(reg)}
SCHEMES = ["R", "H", "S", "SH", "Sw", "DD", "DD-H"]
LEVELS = [0.1, 0.3, 0.5]


stats = hl.cv_stats_fn(Xv, reg, PARTS, SPLITS)     # the 68-region design of the main analyses
design = hl.cv_design


def cv_nulls():
    return hl.null_permutations(reg, 500, {"shuffle": 19000, "spin_std": 19001, "spin_fslr": 19003,
                                           "spin_aligned": 19004, "spin_sym": 19005})


def null_map_rows(j, y, tag, nulls, stats=stats, schemes=SCHEMES):
    """All models x schemes for one map: observed statistics and P values / null summaries for every null."""
    rows = []
    for model in ["ridge", "knn"]:
        for s in schemes:
            st, r2, sk, r = stats(model, y[:, None], s)
            row = {"cond": tag, "map": j, "model": model, "scheme": s, "stat": float(st[0]), "r2": float(r2[0]),
                   "skill": float(sk[0]), "r": float(r[0])}
            for nt, perms in nulls.items():
                ns = stats(model, y[perms].T, s)[0]
                row[f"p_{nt}"] = hl.perm_pvalue(st[0], ns)
                row[f"mu_{nt}"], row[f"sd_{nt}"] = float(ns.mean()), float(ns.std(ddof=1))
                row[f"q95_{nt}"] = float(np.quantile(ns, 0.95))
            rows.append(row)
    return rows


def null_maps(tag):
    if tag == "lam77":
        return hl.grf_maps(reg, geo, lam=77, rho=0.9, n_maps=500, seed=1901)
    if tag == "lam50":
        return np.load(RES / "sim_B1_maps.npy")[:, :500]
    raise ValueError(tag)


# ======================================================================================================================
# cv: null simulation
# ======================================================================================================================
def cv(tag, a=0, b=500):
    M = null_maps(tag)
    nulls = cv_nulls()
    out_f = RES / f"v3_cv_sim_{tag}_{a}_{b}.csv"
    t0 = time.time()
    out = []
    for j in range(a, b):
        out += null_map_rows(j, M[:, j], tag, nulls)
        if (j - a) % 10 == 9 or j == b - 1:
            pd.DataFrame(out).to_csv(out_f, index=False)
            print(f"{tag}: map {j} done at {time.time() - t0:.0f}s", flush=True)


# ======================================================================================================================
# power: positive control at lambda = 77 mm
# ======================================================================================================================
def power(levels, n_analyse=None):
    """n_analyse: analyse only the first maps (partial reruns; all 150 maps are still generated)."""
    schemes = ["R", "H", "SH", "Sw", "DD", "DD-H"]
    n_map = 150
    nulls = hl.null_permutations(reg, 500, {"spin_std": 20001, "spin_fslr": 20003, "spin_sym": 20005})
    t0 = time.time()
    for li in levels:
        s_level = LEVELS[li]
        E = hl.grf_maps(reg, geo, lam=77.0, rho=0.9, n_maps=n_map, seed=20700 + li)
        M, S, C = hl.signal_maps(Xs, E, s_level, np.random.default_rng(20600 + li))
        out = RES / f"v3_power_s{s_level}.csv"
        rows, done = [], set()
        if out.exists() and RESUME:   # keep maps with all model x scheme rows
            prev = pd.read_csv(out)
            cnt = prev.groupby("map").size()
            done = set(cnt[cnt == 2 * len(schemes)].index)
            rows = prev[prev["map"].isin(done)].to_dict("records")
        for j in range(n_analyse or n_map):
            if j in done:
                continue
            y = M[:, j]
            realised = float(np.corrcoef(y, S[:, j])[0, 1] ** 2)
            for model in ["ridge", "knn"]:
                for sc in schemes:
                    st, r2, sk, r = stats(model, y[:, None], sc)
                    row = {"s": s_level, "map": j, "model": model, "scheme": sc, "stat": float(st[0]),
                           "r2": float(r2[0]), "skill": float(sk[0]), "r": float(r[0]), "realised_share": realised,
                           "predictors": ",".join(map(str, C[j]))}
                    for nt, perms in nulls.items():
                        ns = stats(model, y[perms].T, sc)[0]
                        row[f"p_{nt}"] = hl.perm_pvalue(st[0], ns)
                        row[f"mu_{nt}"], row[f"sd_{nt}"] = float(ns.mean()), float(ns.std(ddof=1))
                    rows.append(row)
            if j % 20 == 19:
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"s={s_level}: map {j} at {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)


# ======================================================================================================================
# estimand: replicate R2 for the positive control
# ======================================================================================================================
def estimand():
    """The in-sample signal share s is an upper bound that no model with estimated coefficients reaches out of sample.
    Operational estimand: the replicate R2 - the model is fitted on a whole map (all 68 regions) and evaluated on an
    independent replicate with the same signal and a new noise field. kNN: k = 5. Ridge, three choices of the penalty:
      replicate_r2       leave-one-pair-out tuning (main): the homologue-aware counterpart of leave-one-out
      replicate_r2_loo   plain leave-one-out tuning, which lets the homologue of each left-out region inform the choice
      replicate_r2_best  the single penalty per signal share that is best in hindsight (largest median replicate R2
                         over the 150 maps; an upper bound for ridge on this grid)
    The positive-control maps are regenerated exactly as in `power`."""
    Z, _ = hl._standardise(Xv, Xv)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    nb = np.argsort(D, axis=1)[:, :5]
    pairs = hl._groups_by_size([np.where(PAIR == g)[0] for g in np.unique(PAIR)])

    def ridge_fits(y):
        """Whole-map fits for every penalty of the grid and the penalties chosen by leave-one-out and leave-one-pair-out
        error (exact closed forms)."""
        ybar = y.mean()
        uty = U.T @ (y - ybar)
        fits, e_loo, e_pair = [], [], []
        for a in hl.ALPHAS:
            d = s**2 / (s**2 + a)
            fit = ybar + U @ (d * uty)
            resid = (y - fit)[:, None]
            fits.append(fit)
            e_loo.append((hl._loo_errors(U, d, resid, len(y), None) ** 2).mean())
            e_pair.append((hl._loo_errors(U, d, resid, len(y), pairs) ** 2).mean())
        return np.array(fits), int(np.argmin(e_loo)), int(np.argmin(e_pair))

    def r2(y, p):
        return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    rows, grid = [], [f"rep_a{k}" for k in range(len(hl.ALPHAS))]
    for li, s_level in enumerate(LEVELS):
        E = hl.grf_maps(reg, geo, lam=77.0, rho=0.9, n_maps=150, seed=20700 + li)
        E2 = hl.grf_maps(reg, geo, lam=77.0, rho=0.9, n_maps=150, seed=26700 + li)  # independent noise replicates
        M, S, _ = hl.signal_maps(Xs, E, s_level, np.random.default_rng(20600 + li))
        for j in range(150):
            y = M[:, j]
            yr = np.sqrt(s_level) * S[:, j] + np.sqrt(1 - s_level) * E2[:, j]
            yr = (yr - yr.mean()) / yr.std()
            share = float(np.corrcoef(y, S[:, j])[0, 1] ** 2)
            F, i_loo, i_pair = ridge_fits(y)
            rep = [r2(yr, f) for f in F]
            rows.append({"s": s_level, "map": j, "model": "ridge", "replicate_r2": rep[i_pair],
                         "replicate_r2_loo": rep[i_loo], "insample_r2": r2(y, F[i_pair]), "realised_share": share,
                         "alpha": hl.ALPHAS[i_pair], "alpha_loo": hl.ALPHAS[i_loo], **dict(zip(grid, rep))})
            fit = y[nb].mean(1)
            rows.append({"s": s_level, "map": j, "model": "knn", "replicate_r2": r2(yr, fit), "insample_r2": r2(y, fit),
                         "realised_share": share})
    d = pd.DataFrame(rows)
    rid = d["model"] == "ridge"
    best = d[rid].groupby("s")[grid].median().idxmax(axis=1)   # column of the best penalty for each signal share
    d.loc[rid, "replicate_r2_best"] = [r[best[r["s"]]] for _, r in d[rid].iterrows()]
    d.loc[rid, "alpha_best"] = [hl.ALPHAS[grid.index(best[v])] for v in d.loc[rid, "s"]]
    d = d[["s", "map", "model", "replicate_r2", "replicate_r2_loo", "replicate_r2_best", "insample_r2",
           "realised_share", "alpha", "alpha_loo", "alpha_best"]]
    d.to_csv(RES / "v3_estimand.csv", index=False)
    print(d.groupby(["model", "s"])[["replicate_r2", "replicate_r2_loo", "replicate_r2_best", "insample_r2",
                                     "realised_share"]].median().round(3))


# ======================================================================================================================
# approx: approximate hemispheric correspondence
# ======================================================================================================================
def approx(n_analyse=None):
    """n_analyse: analyse only the first maps (partial reruns; all 200 maps are still generated)."""
    nulls = hl.null_permutations(reg, 500, {"shuffle": 27100, "spin_std": 27101, "spin_fslr": 27103,
                                            "spin_aligned": 27104, "spin_sym": 27105})
    G = hl.approx_maps(reg, geo, 200, 2702)
    rows, t0 = [], time.time()
    for j in range(n_analyse or G.shape[1]):
        y = G[:, j]
        for model in ["ridge", "knn"]:
            for s in ["R", "Sw", "SH", "DD-H"]:
                st = stats(model, y[:, None], s)[0][0]
                row = {"map": j, "model": model, "scheme": s, "stat": st}
                for nt, P in nulls.items():
                    row[f"p_{nt}"] = hl.perm_pvalue(st, stats(model, y[P].T, s)[0])
                rows.append(row)
        if j % 20 == 19:
            pd.DataFrame(rows).to_csv(RES / "v3_approx_coupling_cv.csv", index=False)
            print(f"map {j} at {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RES / "v3_approx_coupling_cv.csv", index=False)


# ======================================================================================================================
# dd_ols: the published distance-dependent analysis
# ======================================================================================================================
def dd_ols():
    """Replication of the distance-dependent CV of Hansen et al. (2022, code/disease.py): least squares (with
    intercept, no tuning) on the 19 PET receptor maps; statistic = mean test-set Pearson r. Because least squares has no
    tuning, the DD vs DD-H contrast is not confounded by the inner loop."""
    pet = X[[c for c in X.columns if c.startswith("PET_")]].to_numpy(float)
    proj = {}
    for s, sp in SPLITS.items():  # least-squares projection (with intercept) from training values to test predictions
        proj[s] = [(tr, te, np.c_[np.ones(len(te)), pet[te]] @ np.linalg.pinv(np.c_[np.ones(len(tr)), pet[tr]]))
                   for tr, te in sp]

    def stat(Ymat, s):
        Ymat = np.asarray(Ymat, float).reshape(68, -1)
        rs = []
        for tr, te, Mp in proj[s]:
            P = Mp @ Ymat[tr]
            yt = Ymat[te]
            yc, pc = yt - yt.mean(0), P - P.mean(0)
            rs.append((yc * pc).sum(0) / np.sqrt((yc**2).sum(0) * (pc**2).sum(0)))
        return np.mean(rs, 0)

    nulls = hl.null_permutations(reg, 2000, {"shuffle": 28000, "spin_std": 28001, "spin_sym": 28005})
    G = null_maps("lam77")
    rows = []
    for j in range(G.shape[1]):
        y = G[:, j]
        for s in SPLITS:
            st = stat(y, s)[0]
            row = {"map": j, "scheme": s, "mean_test_r": st}
            for nt, P in nulls.items():
                row[f"p_{nt}"] = hl.perm_pvalue(st, stat(y[P[:500]].T, s))
            rows.append(row)
    sim = pd.DataFrame(rows)
    sim.to_csv(RES / "v3_dd_ols_sim.csv", index=False)
    print(sim.groupby("scheme").agg(mean_r=("mean_test_r", "mean"),
                                    **{f"fpr_{n}": (f"p_{n}", lambda p: (p < 0.05).mean()) for n in nulls}).round(3))
    rows = []
    for k, name in enumerate(Y.columns):
        for s in SPLITS:
            st = stat(Yv[:, k], s)[0]
            row = {"map": name, "scheme": s, "mean_test_r": st}
            for nt, P in nulls.items():
                row[f"p_{nt}"] = hl.perm_pvalue(st, stat(Yv[:, k][P].T, s))
            rows.append(row)
    real = pd.DataFrame(rows)
    real.to_csv(RES / "v3_dd_ols_real.csv", index=False)
    print(real.round(3).to_string())


# ======================================================================================================================
# gen_cv: cross-validation with generative nulls (pre-specified in plan/prereg_generative_nulls.md)
# ======================================================================================================================
def gen_cv(i0, i1):
    gn = hl.GenerativeNulls(reg, geo)
    variants = ["bs_indep", "bs_sym", "msr_indep", "msr_sym"]
    M = null_maps("lam77")[:, :200]
    i1 = min(i1, 200)
    out = RES / f"v3_genpart_cv_{i0}_{i1}.csv"
    rows, done = [], set()
    if RESUME and out.exists():
        d0 = pd.read_csv(out)
        rows, done = d0.to_dict("records"), set(d0["map"].unique())
    t0 = time.time()
    for j in range(i0, i1):
        if j in done:
            continue
        y = M[:, j]
        S = {v: gn.surrogates(y, v, 500, 29400000 + 100 * j + 10 * k) for k, v in enumerate(variants)}
        for model in ["ridge", "knn"]:
            for s in ["R", "Sw", "SH", "DD-H"]:
                st = float(stats(model, y[:, None], s)[0][0])
                row = {"map": j, "model": model, "scheme": s, "stat": st}
                for v in variants:
                    row[f"p_{v}"] = hl.perm_pvalue(st, stats(model, S[v], s)[0])
                rows.append(row)
        if (j - i0) % 5 == 4 or j == i1 - 1:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"cv: map {j} at {time.time() - t0:.0f}s", flush=True)


# ======================================================================================================================
# half: one value per homologous pair (34 regions)
# ======================================================================================================================
HALF_SCHEMES = ["R", "S", "DD"]


def half_cv(how):
    """Cross-validation when every map is reduced to one value per homologous pair (how = avg: homologue average;
    left: left hemisphere only). Null simulation: the 500 maps of `cv` at lambda = 77 mm (cond 'null'); positive control:
    the maps of `power` (cond 's0.1', 's0.3', 's0.5'; 150 maps each). Schemes R (5 partitions), S and DD; ridge (inner
    tuning mirrors the outer scheme; leave-one-out for DD) and kNN; shuffle and a spin of the left sphere (500 draws)."""
    r34, g34, f = hl.half_brain(reg, geo, how)
    st34 = design(f(Xv), r34, HALF_SCHEMES)
    seed = {"avg": 30100, "left": 30200}[how]
    nulls = hl.null_permutations(r34, 500, {"shuffle": seed, "spin_std": seed + 1})
    out, rows, t0 = RES / f"v3_half_cv_{how}.csv", [], time.time()
    M = f(null_maps("lam77"))
    for j in range(M.shape[1]):
        rows += null_map_rows(j, M[:, j], "signal_free", nulls, st34, HALF_SCHEMES)
    print(f"{how}: null maps done at {time.time() - t0:.0f}s", flush=True)
    for li, s_level in enumerate(LEVELS):
        E = hl.grf_maps(reg, geo, lam=77.0, rho=0.9, n_maps=150, seed=20700 + li)
        Mp = f(hl.signal_maps(Xs, E, s_level, np.random.default_rng(20600 + li))[0])
        for j in range(Mp.shape[1]):
            rows += null_map_rows(j, Mp[:, j], f"s{s_level}", nulls, st34, HALF_SCHEMES)
        print(f"{how}: s = {s_level} done at {time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    d.insert(0, "how", how)
    d.to_csv(out, index=False)
    null = d[d["cond"] == "signal_free"]
    print(null.groupby(["model", "scheme"])[["p_shuffle", "p_spin_std"]].agg(lambda p: (p < 0.05).mean()).round(3))


# ======================================================================================================================
# glasser: fine parcellation (360 regions)
# ======================================================================================================================
GLASSER_SCHEMES = ["R", "H", "S", "SH", "Sw"]
GLASSER_MAPS = 300


def glasser_cv(i0, i1):
    """Cross-validation null simulation on the Glasser parcellation: maps I0..I1-1 of the first 300 null maps (lambda =
    77 mm, rho = 0.9) predicted from 28 simulated annotation maps; schemes R, H, S, SH and Sw with one partition each
    (with 180 homologous pairs, the share of homologues that share a random fold hardly varies between partitions);
    ridge and kNN; shuffle, conventional and symmetric spins (500 draws)."""
    rg, gg, Xg, G = hl.glasser_simulation()
    stg = design(Xg, rg, GLASSER_SCHEMES, n_part=1)
    nulls = hl.null_permutations(rg, 500, {"shuffle": 31100, "spin_std": 31101, "spin_sym": 31105})
    i1 = min(i1, GLASSER_MAPS)
    out = RES / f"v3_glasser_cv_{i0}_{i1}.csv"
    rows, done = [], set()
    if RESUME and out.exists():
        d0 = pd.read_csv(out)
        rows, done = d0.to_dict("records"), set(d0["map"].unique())
    t0 = time.time()
    for j in range(i0, i1):
        if j in done:
            continue
        rows += null_map_rows(j, G[:, j], "glasser", nulls, stg, GLASSER_SCHEMES)
        if (j - i0) % 5 == 4 or j == i1 - 1:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"glasser: map {j} at {time.time() - t0:.0f}s", flush=True)


STEPS = {"cv": lambda tag, a=0, b=500: cv(tag, int(a), int(b)),
         "power": lambda *lv: power([int(v) for v in lv] or [0, 1, 2]), "estimand": estimand, "approx": approx,
         "dd_ols": dd_ols, "gen_cv": lambda i0, i1: gen_cv(int(i0), int(i1)), "half_cv": half_cv,
         "glasser_cv": lambda i0, i1: glasser_cv(int(i0), int(i1))}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        raise SystemExit(__doc__)
    hl.ensure_dirs()
    STEPS[sys.argv[1]](*sys.argv[2:])
