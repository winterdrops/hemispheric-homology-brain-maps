"""
s2_prespecified.py - step 2: the analyses of the initial (pre-specified) plan and the first revision
(Supplementary Note 3, Supplementary Figs 2 and 11-14, Supplementary Table 6).

Pre-specified (analysis plan of 28 September 2026)
  B1          null simulation: 1,000 maps (lambda = 50 mm, rho = 0.9), schemes R, H, S, SH, three nulls (500 draws)
  B2          sensitivity grid over rho and lambda (300 maps per condition; schemes R and SH)
  B3          BrainSMASH-generated target maps (50 per ENIGMA map)
  B4          positive control at lambda = 50 mm (s = 0.1, 0.3, 0.5; 300 maps per level)
  real        ENIGMA maps (5,000 draws per null), receptor-only predictors, in-sample fits, partition-seed sensitivity,
              feature-group drop-out
  rf_real NJ BLOCKS   random forests on the ENIGMA maps for the listed scheme:null blocks (250 draws; NJ workers)
  rf_sim NJ N         random forests on the first N B1 maps (schemes R and SH) and a common shuffle null of N draws
  summarize   hypothesis tests H1-H5 and summary tables
  baseline    R2_cv of the no-skill training-mean predictor under each scheme (exploratory)
  partition   B1 false-positive rates of scheme R under six random partitions (exploratory; archive-only figure figures/extra/Extra_partition_dose_response)
First revision (disclosed deviations; Supplementary Note 2)
  rev_real    ENIGMA maps with inner tuning that mirrors the outer scheme, 10 random partitions for R and H, scheme Sw,
              the aligned spin, skill and Pearson r (1,000 draws)
  rev_sim [LAMBDA]   first 200 B1 maps with group-matched tuning (and 200 maps at LAMBDA mm; schemes R, SH, Sw)
  rev_power   positive control with group-matched tuning (first 100 maps per level)
  percentiles where the pre-specified random partition falls among 100 alternative partitions
  rev_summarize   summaries of the revised analyses (null R2 decomposition, false-positive rates, power)
  checks      ADHD parity, penalty selected by plain vs group-matched tuning, feature-group drop-out with nested tuning

Usage: python s2_prespecified.py STEP [arguments]   (see run_all.py for the order of the steps)
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import homology_lib as hl

warnings.filterwarnings("ignore")
X, Y, reg, geo = hl.load_data()
Xv, Yv = X.to_numpy(float), Y.to_numpy(float)
Xs = (Xv - Xv.mean(0)) / Xv.std(0)
hom = reg["homologue_idx"].to_numpy()
RES = hl.RES
PET = [c for c in X.columns if c.startswith("PET_")]
SCHEMES4 = ["R", "H", "S", "SH"]                    # pre-specified schemes
SCHEMES5 = ["R", "H", "S", "SH", "Sw"]              # revision: + within-hemisphere blocks
FOLDS = {s: hl.make_folds(s, reg) for s in SCHEMES4}  # pre-specified single partitions (seed 2026)
MODELS = ["ridge", "knn"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (p, c - h, c + h)


def nested_stats(model, Ymat, scheme, parts, Xmat=Xv, with_r=False):
    """Mean over partitions of R2_cv and skill (and Pearson r) for every column of Ymat; ridge with inner tuning that
    mirrors the outer scheme."""
    r2s, sks, rs = [], [], []
    for f in parts[scheme]:
        if model == "ridge":
            P = hl.ridge_oof(Xmat, Ymat, f, inner=hl.inner_groups(scheme, reg, f))
        else:
            P = hl.knn_oof(Xmat, Ymat, f)
        r2s.append(hl.r2_cv(Ymat, P))
        sks.append(hl.skill_cv(Ymat, P, f))
        if with_r:
            rs.append(hl.pearson_cols(Ymat, P))
    out = (np.mean(r2s, 0), np.mean(sks, 0))
    return out + (np.mean(rs, 0),) if with_r else out


# ======================================================================================================================
# B1-B3: null simulations (pre-specified)
# ======================================================================================================================
def _analyse_null_map(j, y, nulls, schemes_p, tag):
    rows = []
    for model in MODELS:
        for s in SCHEMES4:
            obs = float(hl.r2_cv(y, hl.oof(model, Xv, y[:, None], FOLDS[s]))[0])
            row = {"sim": tag, "map": j, "model": model, "scheme": s, "r2": obs}
            if s in schemes_p:
                for nt, perms in nulls.items():
                    Yn = y[perms].T
                    row[f"p_{nt}"] = hl.perm_pvalue(obs, hl.r2_cv(Yn, hl.oof(model, Xv, Yn, FOLDS[s])))
            rows.append(row)
    return rows


def _run_null_sim(tag, G, n_draw, schemes_p, seed, extra=None):
    t = time.time()
    nulls = hl.null_permutations(reg, n_draw, {"shuffle": seed, "spin_std": seed + 1, "spin_sym": seed + 2})
    out = Parallel(n_jobs=2)(delayed(_analyse_null_map)(j, G[:, j], nulls, schemes_p, tag) for j in range(G.shape[1]))
    df = pd.DataFrame([r for rows in out for r in rows])
    for k, v in (extra or {}).items():
        df[k] = v
    print(f"{tag}: {G.shape[1]} maps in {time.time() - t:.0f}s", flush=True)
    return df


def sim_b1():
    G = hl.grf_maps(reg, geo, lam=50.0, rho=0.9, n_maps=1000, seed=101)
    np.save(RES / "sim_B1_maps.npy", G)
    _run_null_sim("B1", G, 500, SCHEMES4, seed=5000, extra={"lam": 50.0, "rho": 0.9}).to_csv(
        RES / "sim_B1_linear.csv", index=False)


def sim_b2():
    dfs = []
    for ci, (rho, lam) in enumerate([(0.0, 50.0), (0.5, 50.0), (0.9, 50.0), (1.0, 50.0), (0.9, 25.0), (0.9, 100.0)]):
        G = hl.grf_maps(reg, geo, lam=lam, rho=rho, n_maps=300, seed=200 + ci)
        dfs.append(_run_null_sim(f"B2_rho{rho}_lam{int(lam)}", G, 500, ["R", "SH"], seed=6000 + 10 * ci,
                                 extra={"lam": lam, "rho": rho}))
    pd.concat(dfs).to_csv(RES / "sim_B2_linear.csv", index=False)


def sim_b3():
    from brainsmash.mapgen.base import Base
    hemi = reg["hemi"].to_numpy()
    L, R = np.where(hemi == "L")[0], np.where(hemi == "R")[0]
    pos_in_R = {r: i for i, r in enumerate(R)}
    maps, src = [], []
    for k, name in enumerate(Y.columns):
        y = Yv[:, k]
        rho_hat = float(np.clip(np.corrcoef(y[L], y[hom[L]])[0, 1], 0, 1))
        bl = Base(x=y[L], D=geo[np.ix_(L, L)], resample=False, seed=300 + k)
        br = Base(x=y[R], D=geo[np.ix_(R, R)], resample=False, seed=400 + k)
        sl, sr = bl(n=50), br(n=50)  # (50, 34) each
        for m in range(50):
            v = np.zeros(68)
            zl = (sl[m] - sl[m].mean()) / sl[m].std()
            zr = (sr[m] - sr[m].mean()) / sr[m].std()
            v[L] = zl
            v[hom[L]] = rho_hat * zl + np.sqrt(1 - rho_hat**2) * zr[[pos_in_R[r] for r in hom[L]]]
            maps.append((v - v.mean()) / v.std())
            src.append(name)
    G = np.column_stack(maps)
    np.save(RES / "sim_B3_maps.npy", G)
    df = _run_null_sim("B3", G, 500, ["R", "SH"], seed=7000)
    df["source_map"] = [src[j] for j in df["map"]]
    df.to_csv(RES / "sim_B3_linear.csv", index=False)


# ======================================================================================================================
# B4: positive control (pre-specified)
# ======================================================================================================================
def _analyse_signal_map(j, y, s_level, realised, nulls):
    rows = []
    for model in MODELS:
        for sc in SCHEMES4:
            obs = float(hl.r2_cv(y, hl.oof(model, Xv, y[:, None], FOLDS[sc]))[0])
            row = {"s": s_level, "map": j, "model": model, "scheme": sc, "r2": obs, "realised_share": realised}
            if sc in ("R", "SH"):
                for nt, perms in nulls.items():
                    Yn = y[perms].T
                    row[f"p_{nt}"] = hl.perm_pvalue(obs, hl.r2_cv(Yn, hl.oof(model, Xv, Yn, FOLDS[sc])))
            rows.append(row)
    return rows


def sim_b4():
    t = time.time()
    nulls = hl.null_permutations(reg, 200, {"shuffle": 9500, "spin_std": 9501, "spin_sym": 9502})
    rng = np.random.default_rng(9600)
    allrows = []
    for li, s_level in enumerate([0.1, 0.3, 0.5]):
        E = hl.grf_maps(reg, geo, lam=50.0, rho=0.9, n_maps=300, seed=9700 + li)
        M, S, _ = hl.signal_maps(Xs, E, s_level, rng)
        real = [float(np.corrcoef(M[:, j], S[:, j])[0, 1] ** 2) for j in range(M.shape[1])]
        out = Parallel(n_jobs=2)(delayed(_analyse_signal_map)(j, M[:, j], s_level, real[j], nulls) for j in range(300))
        allrows += [r for rows in out for r in rows]
        print(f"s={s_level} done at {time.time() - t:.0f}s", flush=True)
    pd.DataFrame(allrows).to_csv(RES / "sim_B4_power.csv", index=False)


# ======================================================================================================================
# real: ENIGMA maps (pre-specified analysis A) and secondary analyses
# ======================================================================================================================
def real_prespecified():
    from sklearn.linear_model import LinearRegression
    nulls = hl.null_permutations(reg, 5000, {"shuffle": 9000, "spin_std": 9001, "spin_sym": 9002})
    np.savez_compressed(RES / "real_null_indices.npz", **nulls)

    def evaluate(Xmat, models, tag):
        rows, nulldist = [], {}
        for model in models:
            for s in SCHEMES4:
                P = hl.oof(model, Xmat, Yv, FOLDS[s])
                r2 = hl.r2_cv(Yv, P)
                rr = [np.corrcoef(Yv[:, k], P[:, k])[0, 1] for k in range(Yv.shape[1])]
                for k, name in enumerate(Y.columns):
                    row = {"set": tag, "model": model, "scheme": s, "map": name, "r2": r2[k], "pearson_r": rr[k]}
                    for nt, perms in nulls.items():
                        Yn = Yv[:, k][perms].T
                        r2n = hl.r2_cv(Yn, hl.oof(model, Xmat, Yn, FOLDS[s]))
                        row[f"p_{nt}"] = hl.perm_pvalue(r2[k], r2n)
                        row[f"null95_{nt}"] = float(np.percentile(r2n, 95))
                        nulldist[(tag, model, s, name, nt)] = r2n.astype(np.float32)
                    rows.append(row)
        df = pd.DataFrame(rows)
        for _, g in df.groupby(["set", "model", "scheme"]):
            for nt in nulls:
                df.loc[g.index, f"q_{nt}"] = hl.bh_fdr(g[f"p_{nt}"].to_numpy())
        return df, nulldist

    df, nd = evaluate(Xv, MODELS, "multimodal")                         # primary: 28 predictors, ridge + kNN
    df2, nd2 = evaluate(X[PET].to_numpy(float), ["ridge"], "receptor_only")  # secondary: 19 PET maps, ridge
    pd.concat([df, df2]).to_csv(RES / "real_linear.csv", index=False)
    np.savez_compressed(RES / "real_linear_nulldist.npz", **{"|".join(k): v for k, v in {**nd, **nd2}.items()})

    # in-sample fits (descriptive): least squares with the 19 receptor maps, as in Hansen et al. (2022)
    ins = []
    Xp = X[PET].to_numpy(float)
    Xp = (Xp - Xp.mean(0)) / Xp.std(0)
    for k, name in enumerate(Y.columns):
        r2 = LinearRegression().fit(Xp, Yv[:, k]).score(Xp, Yv[:, k])
        n, p = Xp.shape
        ins.append({"map": name, "ols_receptor_r2": r2, "ols_receptor_adj_r2": 1 - (1 - r2) * (n - 1) / (n - p - 1)})
    pd.DataFrame(ins).to_csv(RES / "real_insample.csv", index=False)

    # partition-seed sensitivity of schemes R and H (observed R2 only)
    seed_rows = []
    for seed in range(20):
        for s in ["R", "H"]:
            f = hl.make_folds(s, reg, seed=seed)
            for model, fn in [("ridge", hl.ridge_oof), ("knn", hl.knn_oof)]:
                r2 = hl.r2_cv(Yv, fn(Xv, Yv, f))
                seed_rows += [{"seed": seed, "scheme": s, "model": model, "map": nm, "r2": v}
                              for nm, v in zip(Y.columns, r2)]
            r2 = hl.r2_cv(Yv, hl.rf_oof_many(Xv, Yv, f, seed=0))
            seed_rows += [{"seed": seed, "scheme": s, "model": "rf", "map": nm, "r2": v} for nm, v in zip(Y.columns, r2)]
    pd.DataFrame(seed_rows).to_csv(RES / "real_seed_sensitivity.csv", index=False)

    # feature-group drop-out (ridge, schemes R and SH)
    groups = {"PET": PET, "MEG": [c for c in X.columns if c.startswith("MEG_")],
              "Connectivity": ["SC_strength", "FC_strength"], "Cytoarchitecture": ["Mesulam_class"]}
    drop_rows = []
    for s in ["R", "SH"]:
        full = hl.r2_cv(Yv, hl.ridge_oof(Xv, Yv, FOLDS[s]))
        for gname, cols in groups.items():
            keep = [c for c in X.columns if c not in cols]
            red = hl.r2_cv(Yv, hl.ridge_oof(X[keep].to_numpy(float), Yv, FOLDS[s]))
            drop_rows += [{"scheme": s, "group": gname, "map": nm, "r2_full": a, "r2_without": b, "delta": a - b}
                          for nm, a, b in zip(Y.columns, full, red)]
    pd.DataFrame(drop_rows).to_csv(RES / "real_group_dropout.csv", index=False)
    print("done")


# ======================================================================================================================
# random forests (pre-specified; reduced for compute, see Supplementary Note 2)
# ======================================================================================================================
def _r2_rf(y, s):
    return float(hl.r2_cv(y, hl.rf_oof(Xv, y, FOLDS[s], seed=0))[0])


def rf_real(nj, blocks):
    """Random forests on the ENIGMA maps: observed R2 for the four schemes and P values for the listed scheme:null
    blocks (250 draws, the first draws of the pre-specified real-data analysis)."""
    t = time.time()
    idx = np.load(RES / "real_null_indices.npz")
    n_draw = 250
    obs = {}
    for s in SCHEMES4:
        vals = Parallel(n_jobs=nj)(delayed(_r2_rf)(Yv[:, k], s) for k in range(Yv.shape[1]))
        for k, v in enumerate(vals):
            obs[(s, k)] = v
    pd.DataFrame([{"scheme": s, "map": Y.columns[k], "r2": v} for (s, k), v in obs.items()]).to_csv(
        RES / "real_rf_obs.csv", index=False)
    rows = []
    for s, nt in [b.split(":") for b in blocks.split(",")]:
        perms = idx[nt][:n_draw]
        tasks = [(k, i) for k in range(Yv.shape[1]) for i in range(n_draw)]
        vals = Parallel(n_jobs=nj)(delayed(_r2_rf)(Yv[:, k][perms[i]], s) for k, i in tasks)
        null = np.array(vals).reshape(Yv.shape[1], n_draw)
        for k in range(Yv.shape[1]):
            rows.append({"scheme": s, "null": nt, "map": Y.columns[k], "r2": obs[(s, k)],
                         "p": hl.perm_pvalue(obs[(s, k)], null[k]), "null95": float(np.percentile(null[k], 95))})
        pd.DataFrame(rows).to_csv(RES / "real_rf_partial.csv", index=False)
        print(f"  real {s}/{nt} done at {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows).drop_duplicates(subset=["scheme", "null", "map"], keep="last")
    for _, g in df.groupby(["scheme", "null"]):
        df.loc[g.index, "q"] = hl.bh_fdr(g["p"].to_numpy())
    df.to_csv(RES / "real_rf.csv", index=False)


def rf_sim(nj, nmap):
    """Random forests on the first nmap B1 maps (schemes R and SH) and on nmap standard-normal targets (common
    shuffle null)."""
    G = np.load(RES / "sim_B1_maps.npy")[:, :nmap]
    tasks = [(j, s) for j in range(nmap) for s in ["R", "SH"]]
    vals = Parallel(n_jobs=nj)(delayed(_r2_rf)(G[:, j], s) for j, s in tasks)
    pd.DataFrame([{"map": j, "scheme": s, "r2": v, "model": "rf"} for (j, s), v in zip(tasks, vals)]) \
        .to_csv(RES / "sim_B1_rf_obs.csv", index=False)
    Z = np.random.default_rng(8000).normal(size=(68, nmap))
    vals = Parallel(n_jobs=nj)(delayed(_r2_rf)(Z[:, j], s) for j, s in tasks)
    pd.DataFrame([{"draw": j, "scheme": s, "r2": v} for (j, s), v in zip(tasks, vals)]) \
        .to_csv(RES / "sim_B1_rf_shufnull.csv", index=False)


# ======================================================================================================================
# summarize: hypotheses H1-H5 (pre-specified decision rules)
# ======================================================================================================================
def summarize():
    rng = np.random.default_rng(12345)
    S = {}

    def boot_mean_ci(x, nboot=10000):
        x = np.asarray(x, float)
        bm = x[rng.integers(0, len(x), size=(nboot, len(x)))].mean(1)
        return float(x.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))

    def fpr_table(df, pcols, by):
        rows = []
        for keys, g in df.groupby(by):
            keys = keys if isinstance(keys, tuple) else (keys,)
            for pc in pcols:
                if pc not in g or g[pc].isna().all():
                    continue
                pv = g[pc].dropna()
                p, lo, hi = wilson(int((pv < 0.05).sum()), len(pv))
                rows.append({**dict(zip(by, keys)), "null": pc.replace("p_", ""), "fpr": p, "lo": lo, "hi": hi,
                             "n": len(pv)})
        return pd.DataFrame(rows)

    pcols = ["p_shuffle", "p_spin_std", "p_spin_sym"]
    b1 = RES / "sim_B1_linear.csv"
    if b1.exists():
        d = pd.read_csv(b1)
        rf_obs = RES / "sim_B1_rf_obs.csv"
        if rf_obs.exists():
            d = pd.concat([d, pd.read_csv(rf_obs).assign(sim="B1")], ignore_index=True)
        wide = d.pivot_table(index=["model", "map"], columns="scheme", values="r2")
        h1, h2 = {}, {}
        for m in wide.index.get_level_values(0).unique():
            w = wide.loc[m].dropna()
            h1[m] = {"mean_R": float(w["R"].mean()), "mean_H": float(w["H"].mean()), "mean_S": float(w["S"].mean()),
                     "mean_SH": float(w["SH"].mean()), "median_R": float(w["R"].median()),
                     "frac_R_positive": float((w["R"] > 0).mean()), "n_maps": int(len(w)),
                     "gap_R_minus_SH": boot_mean_ci(w["R"] - w["SH"]),
                     "gap_R_minus_H": boot_mean_ci(w["R"] - w["H"]),
                     "gap_H_minus_SH": boot_mean_ci(w["H"] - w["SH"]),
                     "gap_R_minus_S": boot_mean_ci(w["R"] - w["S"])}
            h2[m] = {"homology_minus_proximity": boot_mean_ci((w["R"] - w["H"]) - (w["H"] - w["SH"]))}
        common = wide.dropna().reset_index()          # gap comparison between models on common maps
        comp = {}
        if "rf" in common["model"].unique():
            maps_rf = set(common.query("model == 'rf'")["map"])
            for a, b in [("knn", "ridge"), ("rf", "ridge"), ("rf", "knn")]:
                ga = wide.loc[a].loc[sorted(maps_rf)]
                gb = wide.loc[b].loc[sorted(maps_rf)]
                comp[f"{a}_vs_{b}"] = boot_mean_ci((ga["R"] - ga["SH"]).to_numpy() - (gb["R"] - gb["SH"]).to_numpy())
        else:
            ga, gb = wide.loc["knn"], wide.loc["ridge"]
            comp["knn_vs_ridge"] = boot_mean_ci((ga["R"] - ga["SH"]).to_numpy() - (gb["R"] - gb["SH"]).to_numpy())
        S["H1"] = {"per_model": h1, "gap_comparisons": comp}
        S["H2"] = h2
        f1 = fpr_table(d[d["model"] != "rf"], pcols, ["model", "scheme"])
        rf_null = RES / "sim_B1_rf_shufnull.csv"
        if rf_obs.exists() and rf_null.exists():   # RF: shuffle P values against the common null
            r, nn = pd.read_csv(rf_obs), pd.read_csv(rf_null)
            for s in SCHEMES4:
                null = nn.query("scheme == @s")["r2"].to_numpy()
                obs = r.query("scheme == @s")["r2"].to_numpy()
                p = (1 + (null[None, :] >= obs[:, None]).sum(1)) / (1 + len(null))
                fp, lo, hi = wilson(int((p < 0.05).sum()), len(p))
                f1 = pd.concat([f1, pd.DataFrame([{"model": "rf", "scheme": s, "null": "shuffle", "fpr": fp, "lo": lo,
                                                   "hi": hi, "n": len(p)}])], ignore_index=True)
        f1.to_csv(RES / "table_B1_fpr.csv", index=False)
        S["H3_H4_B1_fpr"] = f1.round(4).to_dict(orient="records")

    b2 = RES / "sim_B2_linear.csv"
    if b2.exists():
        d2 = pd.read_csv(b2)
        f2 = fpr_table(d2, pcols, ["rho", "lam", "model", "scheme"])
        f2.to_csv(RES / "table_B2_fpr.csv", index=False)
        d2.groupby(["rho", "lam", "model", "scheme"])["r2"].agg(["mean", "median"]).reset_index().to_csv(
            RES / "table_B2_r2.csv", index=False)
        S["B2_fpr"] = f2.round(4).to_dict(orient="records")

    b3 = RES / "sim_B3_linear.csv"
    if b3.exists():
        d3 = pd.read_csv(b3)
        f3 = fpr_table(d3, pcols, ["model", "scheme"])
        f3.to_csv(RES / "table_B3_fpr.csv", index=False)
        S["B3_fpr"] = f3.round(4).to_dict(orient="records")
        S["B3_r2"] = d3.groupby(["model", "scheme"])["r2"].mean().round(4).reset_index().to_dict(orient="records")

    b4 = RES / "sim_B4_power.csv"
    if b4.exists():
        d4 = pd.read_csv(b4)
        pw = fpr_table(d4, pcols, ["s", "model", "scheme"]).rename(columns={"fpr": "power"})
        pw.to_csv(RES / "table_B4_power.csv", index=False)
        bias = d4.groupby(["s", "model", "scheme"]).agg(r2_mean=("r2", "mean"), r2_median=("r2", "median"),
                                                         realised=("realised_share", "mean")).reset_index()
        bias.to_csv(RES / "table_B4_r2.csv", index=False)
        S["B4_power"] = pw.round(4).to_dict(orient="records")
        S["B4_r2"] = bias.round(4).to_dict(orient="records")

    rl = RES / "real_linear.csv"
    if rl.exists():
        dr = pd.read_csv(rl)
        rows = []
        for (st, m, s), g in dr.groupby(["set", "model", "scheme"]):
            rec = {"set": st, "model": m, "scheme": s, "mean_r2": g["r2"].mean()}
            for nt in ["shuffle", "spin_std", "spin_sym"]:
                rec[f"n_sig_{nt}"] = int((g[f"q_{nt}"] < 0.05).sum())
                rec[f"sig_{nt}"] = ",".join(g.loc[g[f"q_{nt}"] < 0.05, "map"])
            rows.append(rec)
        rrf = RES / "real_rf.csv"
        if rrf.exists():
            f = pd.read_csv(rrf)
            ob = pd.read_csv(RES / "real_rf_obs.csv")
            for s in SCHEMES4:
                rec = {"set": "multimodal", "model": "rf", "scheme": s, "mean_r2": ob.query("scheme == @s")["r2"].mean()}
                for nt in ["shuffle", "spin_std", "spin_sym"]:
                    g = f.query("scheme == @s and null == @nt")
                    if len(g):
                        rec[f"n_sig_{nt}"] = int((g["q"] < 0.05).sum())
                        rec[f"sig_{nt}"] = ",".join(g.loc[g["q"] < 0.05, "map"])
                rows.append(rec)
        t = pd.DataFrame(rows)
        t.to_csv(RES / "table_real_counts.csv", index=False)
        S["H5_real_counts"] = t.round(4).to_dict(orient="records")

    (RES / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps({k: (v if k.startswith(("H1", "H2")) else "...") for k, v in S.items()}, indent=1,
                     default=float)[:6000])


# ======================================================================================================================
# exploratory checks of the pre-specified design
# ======================================================================================================================
def baseline():
    """R2_cv of the no-skill training-mean predictor under each scheme (B1 null maps and ENIGMA maps): how much of the
    negative R2 under blocked CV is a property of the scheme rather than of the model."""
    G = np.load(RES / "sim_B1_maps.npy")
    rows = []
    for s in SCHEMES4:
        f = hl.make_folds(s, reg)
        for name, M in [("B1_null", G), ("ENIGMA", Yv)]:
            P = np.zeros_like(M)
            for k in np.unique(f):
                P[f == k] = M[f != k].mean(0)
            r2 = hl.r2_cv(M, P)
            rows.append({"data": name, "scheme": s, "baseline_r2_mean": float(r2.mean()),
                         "baseline_r2_median": float(np.median(r2))})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "exploratory_baseline_r2.csv", index=False)
    print(out.round(3).to_string())


def partition_robustness():
    """The pre-specified random partition (seed 2026) keeps 23.5% of regions in the same fold as their homologue
    (expected about 8.7%). B1 false-positive rates of scheme R under five other random partitions (first 200 maps)."""
    G = np.load(RES / "sim_B1_maps.npy")[:, :200]
    nulls = hl.null_permutations(reg, 500, {"shuffle": 5000, "spin_std": 5001, "spin_sym": 5002})
    rows = []
    for seed in [0, 1, 2, 3, 4, 2026]:
        f = hl.make_folds("R", reg, seed=seed)
        same = float(np.mean(f == f[hom]))
        for model in MODELS:
            obs = hl.r2_cv(G, hl.oof(model, Xv, G, f))
            for nt, perms in nulls.items():
                p = []
                for j in range(G.shape[1]):
                    Yn = G[:, j][perms].T
                    p.append(hl.perm_pvalue(obs[j], hl.r2_cv(Yn, hl.oof(model, Xv, Yn, f))))
                p = np.array(p)
                rows.append({"partition_seed": seed, "homologue_same_fold": same, "model": model, "null": nt,
                             "mean_r2": float(obs.mean()), "fpr": float((p < 0.05).mean())})
        print("seed", seed, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "exploratory_partition_robustness.csv", index=False)
    print(out.pivot_table(index=["partition_seed", "homologue_same_fold"], columns=["model", "null"],
                          values="fpr").round(3))


def partition_percentiles():
    """Where the pre-specified random partition (seed 2026) falls among 100 alternative random partitions (observed
    R2_cv on the ENIGMA maps; plain-LOO ridge and kNN)."""
    rows = []
    for seed in list(range(100)) + [2026]:
        f = hl.make_folds("R", reg, seed=seed)
        for model, fn in [("ridge", hl.ridge_oof), ("knn", hl.knn_oof)]:
            r2 = hl.r2_cv(Yv, fn(Xv, Yv, f))
            rows += [{"seed": seed, "model": model, "map": m, "r2": v, "homologue_same_fold": float(np.mean(f == f[hom]))}
                     for m, v in zip(Y.columns, r2)]
    d = pd.DataFrame(rows)
    d.to_csv(RES / "exploratory_partition_percentiles_raw.csv", index=False)
    alt, pre = d.query("seed != 2026"), d.query("seed == 2026")
    out = []
    for (m, mp), g in alt.groupby(["model", "map"]):
        v = pre.query("model == @m and map == @mp")["r2"].iloc[0]
        out.append({"model": m, "map": mp, "prereg_r2": v, "percentile": float((g["r2"] < v).mean() * 100),
                    "median_alt": float(g["r2"].median())})
    out = pd.DataFrame(out)
    out.to_csv(RES / "exploratory_partition_percentiles.csv", index=False)
    print(out.pivot_table(index="map", columns="model", values="percentile").round(0).to_string())
    hs = d.drop_duplicates("seed").query("seed != 2026")["homologue_same_fold"]
    print("homologue same-fold fraction across 100 partitions: mean %.3f, range %.3f-%.3f" % (hs.mean(), hs.min(), hs.max()))


# ======================================================================================================================
# first revision: group-matched inner tuning, repeated random partitions, scheme Sw, aligned spin
# ======================================================================================================================
def rev_real():
    parts = {s: hl.partitions(s, reg, 10) for s in SCHEMES5}
    nulls = hl.null_permutations(reg, 1000, {"shuffle": 9100, "spin_std": 9101, "spin_aligned": 9102, "spin_sym": 9103})

    def run(Xmat, tag, models):
        rows = []
        for model in models:
            for s in SCHEMES5:
                r2, sk, rr = nested_stats(model, Yv, s, parts, Xmat, with_r=True)
                for k, name in enumerate(Y.columns):
                    row = {"set": tag, "model": model, "scheme": s, "map": name, "r2": r2[k], "skill": sk[k],
                           "pearson_r": rr[k]}
                    for nt, perms in nulls.items():
                        n2, nsk, _ = nested_stats(model, Yv[:, k][perms].T, s, parts, Xmat, with_r=True)
                        row[f"p_{nt}"] = hl.perm_pvalue(r2[k], n2)
                        row[f"null50_{nt}"] = float(np.median(n2))
                        row[f"null95_{nt}"] = float(np.percentile(n2, 95))
                        row[f"skill_null50_{nt}"] = float(np.median(nsk))
                    rows.append(row)
                print(tag, model, s, "done", flush=True)
        df = pd.DataFrame(rows)
        for _, g in df.groupby(["model", "scheme"]):
            for nt in nulls:
                df.loc[g.index, f"q_{nt}"] = hl.bh_fdr(g[f"p_{nt}"].to_numpy())
        return df

    same = {s: float(np.mean([np.mean(f == f[hom]) for f in parts[s]])) for s in SCHEMES5}
    pd.Series(same).to_csv(RES / "rev_homologue_same_fold.csv")
    df = run(Xv, "multimodal", MODELS)
    df2 = run(X[PET].to_numpy(float), "receptor_only", ["ridge"])
    pd.concat([df, df2]).to_csv(RES / "rev_real.csv", index=False)
    print("done")


def rev_sim(lam=None):
    """First 200 B1 maps (schemes R5, H5, S, SH, Sw) and optionally 200 maps at another lambda (schemes R5, SH, Sw);
    R5/H5 = 5 random partitions, mean statistic; 200 draws per null."""
    t = time.time()
    parts = {s: hl.partitions(s, reg, 5) for s in SCHEMES5}
    nulls = hl.null_permutations(reg, 200, {"shuffle": 9200, "spin_std": 9201, "spin_aligned": 9202, "spin_sym": 9203})
    conds = [("B1_lam50", np.load(RES / "sim_B1_maps.npy")[:, :200], SCHEMES5)]
    if lam is not None:
        conds.append((f"B1_lam{int(round(lam))}", hl.grf_maps(reg, geo, lam=lam, rho=0.9, n_maps=200, seed=111),
                      ["R", "SH", "Sw"]))
    out = []
    for tag, M, schemes in conds:
        for j in range(M.shape[1]):
            y = M[:, j]
            for model in MODELS:
                for s in schemes:
                    r2, sk = nested_stats(model, y[:, None], s, parts)
                    row = {"cond": tag, "map": j, "model": model, "scheme": s, "r2": float(r2[0]), "skill": float(sk[0])}
                    for nt, perms in nulls.items():
                        row[f"p_{nt}"] = hl.perm_pvalue(r2[0], nested_stats(model, y[perms].T, s, parts)[0])
                    out.append(row)
        pd.DataFrame(out).to_csv(RES / "rev_sim_B1.csv", index=False)
        print(tag, f"done at {time.time() - t:.0f}s", flush=True)


def rev_power():
    """Positive control of B4 regenerated exactly (same seeds), first 100 maps per level; group-matched tuning;
    schemes R5, H5, SH, Sw; aligned and symmetric spins (200 draws)."""
    t = time.time()
    schemes = ["R", "H", "SH", "Sw"]
    parts = {s: hl.partitions(s, reg, 5) for s in schemes}
    nulls = hl.null_permutations(reg, 200, {"spin_aligned": 9302, "spin_sym": 9303})
    rng = np.random.default_rng(9600)  # identical to B4
    rows = []
    for li, s_level in enumerate([0.1, 0.3, 0.5]):
        E = hl.grf_maps(reg, geo, lam=50.0, rho=0.9, n_maps=300, seed=9700 + li)
        M, S, _ = hl.signal_maps(Xs, E, s_level, rng)
        for j in range(100):
            y = M[:, j]
            realised = float(np.corrcoef(y, S[:, j])[0, 1] ** 2)
            for model in MODELS:
                for sc in schemes:
                    r2, sk = nested_stats(model, y[:, None], sc, parts)
                    row = {"s": s_level, "map": j, "model": model, "scheme": sc, "r2": float(r2[0]),
                           "skill": float(sk[0]), "realised_share": realised}
                    for nt, perms in nulls.items():
                        row[f"p_{nt}"] = hl.perm_pvalue(r2[0], nested_stats(model, y[perms].T, sc, parts)[0])
                    rows.append(row)
        pd.DataFrame(rows).to_csv(RES / "rev_power.csv", index=False)
        print(f"s={s_level} done at {time.time() - t:.0f}s", flush=True)


def rev_summarize():
    rng = np.random.default_rng(777)
    S = {}
    nulls = ["shuffle", "spin_std", "spin_aligned", "spin_sym"]

    def boot(x, nb=10000):
        x = np.asarray(x, float)
        bm = x[rng.integers(0, len(x), (nb, len(x)))].mean(1)
        return [float(x.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]

    f = RES / "rev_sim_B1.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows = []
        for (c, m, s), g in d.groupby(["cond", "model", "scheme"]):
            for nt in nulls:
                p, lo, hi = wilson(int((g[f"p_{nt}"] < 0.05).sum()), len(g))
                rows.append({"cond": c, "model": m, "scheme": s, "null": nt, "fpr": p, "lo": lo, "hi": hi, "n": len(g),
                             "crit_p": float(np.quantile(g[f"p_{nt}"], 0.05))})
        t = pd.DataFrame(rows)
        t.to_csv(RES / "rev_table_fpr.csv", index=False)
        S["fpr"] = t.round(4).to_dict(orient="records")
        r2 = d.groupby(["cond", "model", "scheme"])[["r2", "skill"]].mean().reset_index()
        r2.to_csv(RES / "rev_table_null_r2.csv", index=False)
        S["null_r2"] = r2.round(4).to_dict(orient="records")
        # H1/H2 with nested tuning and repeated partitions (lambda = 50 mm)
        w = d.query("cond == 'B1_lam50'").pivot_table(index=["model", "map"], columns="scheme", values="r2")
        h = {}
        for m in MODELS:
            x = w.loc[m]
            h[m] = {"gap_R_SH": boot(x["R"] - x["SH"]), "homology_random": boot(x["R"] - x["H"]),
                    "homology_spatial": boot(x["Sw"] - x["SH"]), "proximity_split": boot(x["R"] - x["Sw"]),
                    "proximity_grouped": boot(x["H"] - x["SH"]),
                    "interaction": boot((x["R"] - x["H"]) - (x["Sw"] - x["SH"]))}
        h["gap_knn_minus_ridge"] = boot((w.loc["knn"]["R"] - w.loc["knn"]["SH"]).to_numpy()
                                        - (w.loc["ridge"]["R"] - w.loc["ridge"]["SH"]).to_numpy())
        S["decomposition"] = h

    f = RES / "rev_power.csv"
    if f.exists() and (RES / "rev_table_fpr.csv").exists():
        p = pd.read_csv(f)
        crit = pd.read_csv(RES / "rev_table_fpr.csv").query("cond == 'B1_lam50'").set_index(
            ["model", "scheme", "null"])["crit_p"]
        rows = []
        for (s, m, sc), g in p.groupby(["s", "model", "scheme"]):
            for nt in ["spin_aligned", "spin_sym"]:
                if f"p_{nt}" not in g:
                    continue
                cp = min(0.05, float(crit.loc[(m, sc, nt)]))
                rows.append({"s": s, "model": m, "scheme": sc, "null": nt,
                             "power_nominal": float((g[f"p_{nt}"] < 0.05).mean()),
                             "power_size_adj": float((g[f"p_{nt}"] <= cp).mean()), "crit_p": cp, "n": len(g)})
        pw = pd.DataFrame(rows)
        pw.to_csv(RES / "rev_table_power.csv", index=False)
        est = p.groupby(["s", "model", "scheme"]).agg(r2_median=("r2", "median"), skill_median=("skill", "median"),
                                                       true_share=("realised_share", "mean")).reset_index()
        est.to_csv(RES / "rev_table_estimation.csv", index=False)
        S["power"] = pw.round(4).to_dict(orient="records")
        S["estimation"] = est.round(4).to_dict(orient="records")

    f = RES / "rev_real.csv"
    if f.exists():
        r = pd.read_csv(f)
        rows = []
        for (st, m, s), g in r.groupby(["set", "model", "scheme"]):
            rec = {"set": st, "model": m, "scheme": s, "mean_r2": g["r2"].mean(), "mean_skill": g["skill"].mean()}
            for nt in nulls:
                rec[f"n_{nt}"] = int((g[f"q_{nt}"] < 0.05).sum())
                rec[f"maps_{nt}"] = ",".join(g.loc[g[f"q_{nt}"] < 0.05, "map"])
            rows.append(rec)
        c = pd.DataFrame(rows)
        c.to_csv(RES / "rev_table_real_counts.csv", index=False)
        S["real_counts"] = c.round(4).to_dict(orient="records")

    f = RES / "real_rf.csv"
    if f.exists():
        rf = pd.read_csv(f)
        S["rf_real"] = {f"{s}_{n}": {"n_sig": int((g["q"] < 0.05).sum()), "maps": ",".join(g.loc[g["q"] < 0.05, "map"])}
                        for (s, n), g in rf.groupby(["scheme", "null"])}
    f1, f2 = RES / "sim_B1_rf_obs.csv", RES / "sim_B1_rf_shufnull.csv"
    if f1.exists() and f2.exists():
        o, nn = pd.read_csv(f1), pd.read_csv(f2)
        out = {}
        for s in sorted(o["scheme"].unique()):
            obs = o.query("scheme == @s")["r2"].to_numpy()
            null = nn.query("scheme == @s")["r2"].to_numpy()
            pv = (1 + (null[None, :] >= obs[:, None]).sum(1)) / (1 + len(null))
            fp, lo, hi = wilson(int((pv < 0.05).sum()), len(pv))
            out[s] = {"mean_r2": float(obs.mean()), "frac_pos": float((obs > 0).mean()), "shuffle_fpr": fp, "lo": lo,
                      "hi": hi, "n": len(pv)}
        S["rf_sim"] = out
    (RES / "rev_summary.json").write_text(json.dumps(S, indent=1, default=float))
    print("keys:", list(S.keys()))


def checks():
    """(a) parity of the ADHD map with Hansen et al. (2022): adult vs all-ages ENIGMA files; (b) penalty selected under
    nested vs plain tuning on B1 null maps (H and SH); (c) feature-group drop-out with nested tuning."""
    names = reg["name"].tolist()
    out = {}
    info = pd.read_excel(hl.RAW_HANSEN / "region_info_scale033.xlsx").query("structure == 'cortex'")
    laus = (info["hemisphere"].str.strip() + "_" + info["label"].str.strip()).tolist()
    h = np.loadtxt(hl.RAW_HANSEN / "enigma_atrophy.csv", delimiter=",")[[laus.index(n) for n in names]][:, 1]  # 'adhd'
    for f in ["adhdallages_case-controls_CortThick", "adhdadult_case-controls_CortThick"]:
        d = pd.read_csv(hl.RAW_ENIGMA / f"{f}.csv")
        d["Structure"] = d["Structure"].str.strip()
        v = d.set_index("Structure").loc[names, "d_icv"].astype(float).to_numpy()
        out[f"adhd_parity_{f.split('_')[0]}"] = {"pearson_r": float(np.corrcoef(h, v)[0, 1]),
                                                 "max_abs_diff": float(np.max(np.abs(h - v)))}
    G = np.load(RES / "sim_B1_maps.npy")[:, :200]
    for s in ["H", "SH"]:
        f = hl.make_folds(s, reg)
        _, a0 = hl.ridge_oof(Xv, G, f, return_alpha=True)
        _, a1 = hl.ridge_oof(Xv, G, f, inner=hl.inner_groups(s, reg, f), return_alpha=True)
        out[f"alpha_{s}"] = {"median_plain": float(np.median(a0)), "median_nested": float(np.median(a1)),
                             "frac_nested_at_max": float(np.mean(a1 == hl.ALPHAS.max()))}
    groups = {"PET": PET, "MEG": [c for c in X.columns if c.startswith("MEG_")],
              "Connectivity": ["SC_strength", "FC_strength"], "Cytoarchitecture": ["Mesulam_class"]}

    def perf(cols, scheme):
        Xm = X[cols].to_numpy(float)
        parts = hl.partitions(scheme, reg, 10)
        return np.mean([hl.r2_cv(Yv, hl.ridge_oof(Xm, Yv, f, inner=hl.inner_groups(scheme, reg, f))) for f in parts], 0)

    rows = []
    for s in ["R", "SH"]:
        full = perf(list(X.columns), s)
        for gname, cols in groups.items():
            red = perf([c for c in X.columns if c not in cols], s)
            rows += [{"scheme": s, "group": gname, "map": m, "delta": a - b} for m, a, b in zip(Y.columns, full, red)]
    dd = pd.DataFrame(rows)
    dd.to_csv(RES / "rev_group_dropout.csv", index=False)
    out["dropout_median"] = dd.groupby(["scheme", "group"])["delta"].median().round(4).to_dict()
    out = {k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in out.items()}
    (RES / "rev_checks.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


STEPS = {"B1": sim_b1, "B2": sim_b2, "B3": sim_b3, "B4": sim_b4, "real": real_prespecified,
         "rf_real": lambda nj, blocks: rf_real(int(nj), blocks), "rf_sim": lambda nj, n: rf_sim(int(nj), int(n)),
         "summarize": summarize, "baseline": baseline, "partition": partition_robustness, "rev_real": rev_real,
         "rev_sim": lambda lam=None: rev_sim(None if lam is None else float(lam)), "rev_power": rev_power,
         "percentiles": partition_percentiles, "rev_summarize": rev_summarize, "checks": checks}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        raise SystemExit(__doc__)
    hl.ensure_dirs()
    STEPS[sys.argv[1]](*sys.argv[2:])
