"""
s5_enigma.py - step 5: reanalysis of the 12 ENIGMA cortical-thickness maps (Figs 5b-d, Supplementary Figs 4-5,
Supplementary Tables 3b, 12a-c and 13a-c).

  cv          ridge and kNN under all seven schemes and six nulls (2,000 draws each), for the 28 annotations and for the
              19 receptor maps; out-of-fold predictions, full regression metrics, z-scores relative to every null,
              BH-FDR and Holm within each (predictor set, model, scheme, null) family of 12 maps
  hyper       hyper-parameter sensitivity: ridge with a wider penalty grid (10^-3 .. 10^6), kNN with k = 3, 10, 20;
              schemes R (10 partitions), SH and DD-H; symmetric spin (2,000 draws)
  noise       noise perturbation on the scale of the reported standard errors (100 replicates; ridge and kNN under R and
              SH; symmetric spin, 500 draws per replicate). This adds, rather than removes, measurement-scale noise.
  rf MODE [I0 I1]   random forests for maps I0..I1-1; MODE mainR (5 random partitions, 300 draws), mainSH (500 draws)
                    or alt (alternative hyper-parameters under SH, 500 draws); symmetric spin
  rf_merge    combine the random-forest parts; BH q-values across the 12 maps of each variant and scheme
  half        one value per homologous pair (34 regions; homologue average or left hemisphere): R, S and DD; shuffle
              and a spin of the left sphere (2,000 draws)
Long random-forest jobs resume where they stopped when HEMIHOM_RESUME=1.

Usage: python s5_enigma.py STEP [arguments]
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import homology_lib as hl

warnings.filterwarnings("ignore")
X, Y, reg, geo = hl.load_data()
Xv, Yv = X.to_numpy(float), Y.to_numpy(float)
RES = hl.RES
RESUME = os.environ.get("HEMIHOM_RESUME") == "1"
PAIR = hl.pair_ids(reg)
PET = [c for c in X.columns if c.startswith("PET_")]
PARTS = {s: hl.partitions(s, reg, 10) for s in ["R", "H", "S", "SH", "Sw"]}   # R and H: 10 random partitions
SPLITS = {"DD": hl.dd_splits(reg), "DD-H": hl.ddh_splits(reg)}
SCHEMES = ["R", "H", "S", "SH", "Sw", "DD", "DD-H"]


# ======================================================================================================================
# cv: main reanalysis
# ======================================================================================================================
def cv_nulls():
    return hl.null_permutations(reg, 2000, {"shuffle": 21000, "spin_std": 21001, "spin_enigma": 21002,
                                            "spin_fslr": 21003, "spin_aligned": 21004, "spin_sym": 21005})


def predict(model, Xm, Ymat, s):
    """Partition schemes: list of (n x m) out-of-fold matrices, one per partition; DD schemes: list of per-split test
    predictions."""
    if s in PARTS:
        out = []
        for f in PARTS[s]:
            if model == "ridge":
                out.append(hl.ridge_oof(Xm, Ymat, f, inner=hl.inner_groups(s, reg, f)))
            else:
                out.append(hl.knn_oof(Xm, Ymat, f))
        return out
    sp = SPLITS[s]
    if model == "ridge":
        return hl.ridge_split_predict(Xm, Ymat, sp, inner=None if s == "DD" else PAIR)
    return hl.knn_split_predict(Xm, Ymat, sp)


def statistic(Ymat, s, preds):
    """R2_cv averaged over partitions (partition schemes) or mean test-set r (DD schemes)."""
    if s in PARTS:
        return np.mean([hl.r2_cv(Ymat, P) for P in preds], 0)
    return hl.split_stats(Ymat, SPLITS[s], preds)[0]


def cv():
    nulls = cv_nulls()

    def run(Xm, tag, models):
        rows, mrows, store = [], [], {}
        for model in models:
            for s in SCHEMES:
                t = time.time()
                preds = predict(model, Xm, Yv, s)
                store[f"{tag}|{model}|{s}"] = np.array(preds, dtype=float)
                st = statistic(Yv, s, preds)
                for k, name in enumerate(Y.columns):
                    row = {"set": tag, "model": model, "scheme": s, "map": name, "stat": st[k],
                           "stat_name": "R2_cv" if s in PARTS else "mean_test_r"}
                    if s in PARTS:
                        met = pd.DataFrame([hl.regression_metrics(Yv[:, k], P[:, k]) for P in preds]).mean()
                        met["skill"] = np.mean([hl.skill_cv(Yv[:, k], P[:, k], f)[0] for P, f in zip(preds, PARTS[s])])
                    else:
                        r_, r2_, sk_ = hl.split_stats(Yv[:, [k]], SPLITS[s], [P[:, [k]] for P in preds])
                        met = pd.Series({"mean_test_r": r_[0], "mean_test_r2": r2_[0], "skill": sk_[0]})
                    mrows.append({"set": tag, "model": model, "scheme": s, "map": name, **met.to_dict()})
                    for nt, perms in nulls.items():
                        Yn = Yv[:, k][perms].T
                        ns = statistic(Yn, s, predict(model, Xm, Yn, s))
                        row[f"p_{nt}"] = hl.perm_pvalue(st[k], ns)
                        row[f"null_mean_{nt}"], row[f"null_sd_{nt}"] = float(ns.mean()), float(ns.std(ddof=1))
                        row[f"null50_{nt}"], row[f"null95_{nt}"] = float(np.median(ns)), float(np.quantile(ns, 0.95))
                        row[f"z_{nt}"] = (st[k] - ns.mean()) / ns.std(ddof=1)
                    rows.append(row)
                print(tag, model, s, f"{time.time() - t:.0f}s", flush=True)
        df = pd.DataFrame(rows)
        for _, g in df.groupby(["set", "model", "scheme"]):
            for nt in nulls:
                df.loc[g.index, f"q_{nt}"] = multipletests(g[f"p_{nt}"], method="fdr_bh")[1]
                df.loc[g.index, f"holm_{nt}"] = multipletests(g[f"p_{nt}"], method="holm")[1]
        return df, pd.DataFrame(mrows), store

    d1, m1, s1 = run(Xv, "multimodal", ["ridge", "knn"])
    d1.to_csv(RES / "v3_real_cv.csv", index=False)
    d2, m2, s2 = run(X[PET].to_numpy(float), "receptor_only", ["ridge", "knn"])
    pd.concat([d1, d2]).to_csv(RES / "v3_real_cv.csv", index=False)
    pd.concat([m1, m2]).to_csv(RES / "v3_real_metrics.csv", index=False)
    np.savez_compressed(RES / "v3_real_predictions.npz", **{**s1, **s2})
    print("done")


# ======================================================================================================================
# sensitivity analyses
# ======================================================================================================================
R10, SH1, DDH = PARTS["R"], PARTS["SH"], SPLITS["DD-H"]


def sens_stat(model, Ymat, scheme, alphas=hl.ALPHAS, k=5):
    if scheme == "DD-H":
        preds = (hl.ridge_split_predict(Xv, Ymat, DDH, alphas=alphas, inner=PAIR) if model == "ridge"
                 else hl.knn_split_predict(Xv, Ymat, DDH, k=k))
        return hl.split_stats(Ymat, DDH, preds)[0]
    out = []
    for f in (R10 if scheme == "R" else SH1):
        if model == "ridge":
            out.append(hl.r2_cv(Ymat, hl.ridge_oof(Xv, Ymat, f, alphas=alphas, inner=hl.inner_groups(scheme, reg, f))))
        else:
            out.append(hl.r2_cv(Ymat, hl.knn_oof(Xv, Ymat, f, k=k)))
    return np.mean(out, 0)


def hyper():
    perms = hl.spin_permutations_sym(reg, 2000, seed=22001)
    variants = [("ridge", "alpha 1e-2..1e4 (main)", dict(alphas=hl.ALPHAS)),
                ("ridge", "alpha 1e-3..1e6", dict(alphas=hl.ALPHAS_WIDE)),
                ("knn", "k=3", dict(k=3)), ("knn", "k=5 (main)", dict(k=5)),
                ("knn", "k=10", dict(k=10)), ("knn", "k=20", dict(k=20))]
    rows = []
    for model, label, kw in variants:
        for scheme in ["R", "SH", "DD-H"]:
            st = sens_stat(model, Yv, scheme, **kw)
            for j, name in enumerate(Y.columns):
                ns = sens_stat(model, Yv[:, j][perms].T, scheme, **kw)
                rows.append({"model": model, "variant": label, "scheme": scheme, "map": name, "stat": st[j],
                             "p_spin_sym": hl.perm_pvalue(st[j], ns)})
            print(model, label, scheme, flush=True)
    d = pd.DataFrame(rows)
    d["q_spin_sym"] = d.groupby(["model", "variant", "scheme"])["p_spin_sym"].transform(
        lambda p: multipletests(p, method="fdr_bh")[1])
    d.to_csv(RES / "v3_sens_hyper.csv", index=False)
    # where do the nested penalties fall on the grid (real maps, SH)?
    f = SH1[0]
    _, a_main = hl.ridge_oof(Xv, Yv, f, inner=hl.inner_groups("SH", reg, f), return_alpha=True)
    _, a_wide = hl.ridge_oof(Xv, Yv, f, alphas=hl.ALPHAS_WIDE, inner=hl.inner_groups("SH", reg, f), return_alpha=True)
    with open(RES / "v3_sens_alpha.json", "w") as fh:
        json.dump({"real_SH_alpha_at_max_main": float(np.mean(a_main == hl.ALPHAS.max())),
                   "real_SH_alpha_at_max_wide": float(np.mean(a_wide == hl.ALPHAS_WIDE.max())),
                   "real_SH_alpha_median_main": float(np.median(a_main)),
                   "real_SH_alpha_median_wide": float(np.median(a_wide))}, fh, indent=1)
    print(d.groupby(["model", "variant", "scheme"])["q_spin_sym"].apply(lambda q: int((q < 0.05).sum())))


def noise(B=100):
    SE, _ = hl.target_standard_errors(reg, Y)
    perms = hl.spin_permutations_sym(reg, 500, seed=22002)
    rng = np.random.default_rng(22003)
    hom = reg["homologue_idx"].to_numpy()
    Lh = np.where(reg["hemi"].to_numpy() == "L")[0]
    identical = [j for j in range(Yv.shape[1]) if np.allclose(Yv[Lh, j], Yv[hom[Lh], j])]  # hemisphere-averaged maps
    rows = []
    t = time.time()
    for b in range(B):
        E = rng.normal(size=Yv.shape)
        for j in identical:  # one error draw per homologous pair keeps the released hemisphere duplication
            E[hom[Lh], j] = E[Lh, j]
        Yb = Yv + SE.to_numpy() * E
        for model in ["ridge", "knn"]:
            for scheme in ["R", "SH"]:
                st = sens_stat(model, Yb, scheme)
                for j, name in enumerate(Y.columns):
                    ns = sens_stat(model, Yb[:, j][perms].T, scheme)
                    rows.append({"rep": b, "model": model, "scheme": scheme, "map": name, "r2": st[j],
                                 "p_spin_sym": hl.perm_pvalue(st[j], ns)})
        if b % 10 == 9:
            pd.DataFrame(rows).to_csv(RES / "v3_sens_measurement.csv", index=False)
            print(f"replicate {b + 1}/{B} at {time.time() - t:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    d["q_spin_sym"] = d.groupby(["rep", "model", "scheme"])["p_spin_sym"].transform(
        lambda p: multipletests(p, method="fdr_bh")[1])
    d.to_csv(RES / "v3_sens_measurement.csv", index=False)


# ======================================================================================================================
# random forests over repeated partitions
# ======================================================================================================================
# With 12 maps and BH-FDR the smallest attainable P value must fall below 0.05 / 12 = 0.0042, hence >= 300 draws.
RF_JOBS = {"mainR": ("R", hl.RF_PARAMS, "main", 300), "mainSH": ("SH", hl.RF_PARAMS, "main", 500),
           "alt": ("SH", hl.RF_PARAMS_ALT, "alt", 500)}
RF_PARTS = {"R": hl.partitions("R", reg, 5), "SH": hl.partitions("SH", reg, 5)}


def _rf_stat(y, scheme, params):
    return np.mean([float(hl.r2_cv(y, hl.rf_oof(Xv, y, f, seed=0, params=params))[0]) for f in RF_PARTS[scheme]])


def _rf_done(mode):
    fs = sorted(RES.glob(f"v3_rfpart_{mode}_*.csv"))
    if not fs:
        return pd.DataFrame(columns=["variant", "scheme", "map"])
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(["variant", "scheme", "map"])


def rf(mode, i0=0, i1=None):
    """Random forests for maps i0..i1-1 (maps are independent: fixed permutations, random_state = 0)."""
    scheme, params, label, n_draw = RF_JOBS[mode]
    perms = hl.spin_permutations_sym(reg, 1000, seed=23001)[:n_draw]
    i1 = len(Y.columns) if i1 is None else i1
    done = set(_rf_done(mode)["map"]) if RESUME else set()
    out = RES / f"v3_rfpart_{mode}_{i0}_{i1}.csv"
    rows = pd.read_csv(out).to_dict("records") if (RESUME and out.exists()) else []
    t0 = time.time()
    for j in range(i0, i1):
        name = Y.columns[j]
        if name in done:
            print("already done:", name, flush=True)
            continue
        y = Yv[:, j]
        obs = _rf_stat(y, scheme, params)
        null = np.array([_rf_stat(y[p], scheme, params) for p in perms])
        rows.append({"variant": label, "scheme": scheme, "map": name, "n_draws": len(null), "r2": obs,
                     "p_spin_sym": hl.perm_pvalue(obs, null), "null95": float(np.quantile(null, 0.95))})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(label, scheme, name, f"{time.time() - t0:.0f}s", flush=True)


def rf_merge():
    for modes, name in [(("mainR", "mainSH"), "v3_rf_main.csv"), (("alt",), "v3_rf_alt.csv")]:
        d = pd.concat([_rf_done(m) for m in modes], ignore_index=True)
        if not len(d):
            continue
        d = d.drop(columns=[c for c in d.columns if c.startswith("q_")])
        d["q_spin_sym"] = d.groupby(["variant", "scheme"])["p_spin_sym"].transform(
            lambda p: multipletests(p, method="fdr_bh")[1])
        d.to_csv(RES / name, index=False)
        print(name, d.groupby(["variant", "scheme"])["map"].nunique().to_dict())


# ======================================================================================================================
# half: one value per homologous pair (34 regions)
# ======================================================================================================================
def half():
    """The 12 ENIGMA maps and 28 annotations reduced to one value per homologous pair (homologue average or left
    hemisphere only): ridge and kNN under R (10 partitions), S and DD; shuffle and a spin of the left sphere (2,000
    draws); BH-FDR and Holm within each (version, model, scheme, null) family of 12 maps."""
    rows = []
    for k, how in enumerate(["avg", "left"]):
        r34, _, f = hl.half_brain(reg, geo, how)
        stats = hl.cv_design(f(Xv), r34, ["R", "S", "DD"], n_part=10)
        nulls = hl.null_permutations(r34, 2000, {"shuffle": 30300 + 100 * k, "spin_std": 30301 + 100 * k})
        Y34 = f(Yv)
        for model in ["ridge", "knn"]:
            for s in ["R", "S", "DD"]:
                st = stats(model, Y34, s)[0]
                for j, name in enumerate(Y.columns):
                    row = {"how": how, "model": model, "scheme": s, "map": name, "stat": st[j]}
                    for nt, P in nulls.items():
                        row["p_spin" if nt == "spin_std" else f"p_{nt}"] = hl.perm_pvalue(
                            st[j], stats(model, Y34[:, j][P].T, s)[0])
                    rows.append(row)
                print(how, model, s, "done", flush=True)
    df = pd.DataFrame(rows)
    for _, g in df.groupby(["how", "model", "scheme"]):
        for nt in ["shuffle", "spin"]:
            df.loc[g.index, f"q_{nt}"] = multipletests(g[f"p_{nt}"], method="fdr_bh")[1]
            df.loc[g.index, f"holm_{nt}"] = multipletests(g[f"p_{nt}"], method="holm")[1]
    df.to_csv(RES / "v3_half_real_cv.csv", index=False)
    print(df.groupby(["how", "model", "scheme"]).agg(
        median_stat=("stat", "median"), n_bh_shuffle=("q_shuffle", lambda q: int((q < 0.05).sum())),
        n_bh_spin=("q_spin", lambda q: int((q < 0.05).sum()))).round(3).to_string())


STEPS = {"cv": cv, "hyper": hyper, "noise": noise,
         "rf": lambda mode, *r: rf(mode, *(int(a) for a in r)), "rf_merge": rf_merge, "half": half}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        raise SystemExit(__doc__)
    hl.ensure_dirs()
    STEPS[sys.argv[1]](*sys.argv[2:])
