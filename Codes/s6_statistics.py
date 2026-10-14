"""
s6_statistics.py - step 6: statistical summaries and formal tests (reads result files; no model fitting).

main        A. map-to-map correlations: false-positive rates with map-level bootstrap CIs, t-tests against 5%, Holm,
               classification against the 2.5-7.5% band, paired differences between nulls (Supplementary Tables 2a-b)
            B. cross-validation null simulation: Wilson CIs, Monte Carlo SEs, exact binomial tests (Holm), Cochran's Q,
               exact McNemar contrasts, null optimism with Wilcoxon / Hodges-Lehmann and Friedman + Nemenyi
               (Supplementary Tables 3a, 3c, 3d, 4a, 5a, 5b)
            C. positive control: ROC curves and AUC (score = z relative to the null) with stratified bootstrap CIs,
               paired bootstrap AUC differences, confusion matrices at the nominal threshold (P < 0.05) and at the
               size-adjusted threshold (z above its 95th percentile among the signal-free maps: exactly 5% false
               positives), PPV/NPV, MCC, estimation (Supplementary Tables 6a-c)
            D. ENIGMA maps: significant-map counts (BH and Holm) and Wilcoxon tests (Supplementary Table 7c)
            E. least-squares distance-dependent analysis and approximate correspondence (Supplementary Tables 3b, 4b)
generative  generative null models: false-positive rates for map-to-map correlations and cross-validation, contrasts
            with the spins, bilateral structure (Supplementary Tables 2e, 2f, 4c)
extensions  one value per homologous pair (34 regions) and the Glasser parcellation (360 regions): false-positive
            rates, AUC and power, ENIGMA counts (Supplementary Tables 11 and 12)

Usage: python s6_statistics.py [main] [generative] [extensions]   (default: all)
"""
import glob
import json
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

import homology_lib as hl

warnings.filterwarnings("ignore")
RES = hl.RES
NULLS_CV = ["shuffle", "spin_std", "spin_fslr", "spin_aligned", "spin_sym"]
GEN_VARIANTS = ["bs_indep", "bs_shared", "bs_whole", "bs_sym", "msr_indep", "msr_shared", "msr_sym"]
GEN_CV_VARIANTS = ["bs_indep", "bs_sym", "msr_indep", "msr_sym"]


# ======================================================================================================================
# helpers
# ======================================================================================================================
def wilson(k, n):
    return proportion_confint(k, n, alpha=0.05, method="wilson")


def bradley(fpr, lo, hi):
    """Classification against Bradley's (1978) liberal robustness band [2.5%, 7.5%] for a nominal 5% test:
    'liberal' if the whole 95% CI lies above 7.5%, 'conservative' if it lies below 2.5%, 'acceptable' if the point
    estimate lies within the band, otherwise 'borderline'."""
    if lo > 0.075:
        return "liberal"
    if hi < 0.025:
        return "conservative"
    if 0.025 <= fpr <= 0.075:
        return "acceptable"
    return "borderline"


def binom_two_sided(k, n, p0=0.05):
    return stats.binomtest(int(k), int(n), p0, alternative="two-sided").pvalue


def mcnemar_exact(a, b):
    """a, b boolean arrays (rejections of two procedures on the same maps). Returns (n10, n01, exact P)."""
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    return n10, n01, (1.0 if n == 0 else stats.binomtest(n10, n, 0.5).pvalue)


def cochran_q(M):
    """Cochran's Q for an (n_subjects x k) binary matrix."""
    M = np.asarray(M, int)
    k = M.shape[1]
    col, row = M.sum(0), M.sum(1)
    N = M.sum()
    den = k * N - np.sum(row**2)
    if den == 0:
        return 0.0, 1.0
    Q = (k - 1) * (k * np.sum(col**2) - N**2) / den
    return float(Q), float(stats.chi2.sf(Q, k - 1))


def hodges_lehmann(d):
    """Hodges-Lehmann estimate (median of Walsh averages) and 95% CI (normal approximation of the Wilcoxon statistic)."""
    d = np.asarray(d, float)
    n = len(d)
    walsh = np.sort(((d[:, None] + d[None, :]) / 2)[np.triu_indices(n)])
    m = len(walsh)
    c = n * (n + 1) / 4 - 1.96 * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    lo_i, hi_i = int(np.floor(c)), int(np.ceil(m - 1 - c))
    return np.median(walsh), walsh[max(lo_i, 0)], walsh[min(hi_i, m - 1)]


def nemenyi(M, labels):
    """Friedman test and Nemenyi post hoc on an (n_blocks x k) matrix (higher value = higher rank)."""
    M = np.asarray(M, float)
    n, k = M.shape
    fr = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    Rbar = np.apply_along_axis(stats.rankdata, 1, M).mean(0)
    se = np.sqrt(k * (k + 1) / (12 * n))
    rows = [{"a": labels[i], "b": labels[j], "mean_rank_a": Rbar[i], "mean_rank_b": Rbar[j],
             "p_nemenyi": float(stats.studentized_range.sf(abs(Rbar[i] - Rbar[j]) / se, k, np.inf))}
            for i, j in combinations(range(k), 2)]
    return fr.statistic, fr.pvalue, pd.DataFrame(rows)


def auc_mw(pos, neg):
    """AUC = P(score_pos > score_neg) + 0.5 P(tie)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = stats.rankdata(np.r_[pos, neg])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def roc_curve(pos, neg):
    thr = np.unique(np.r_[pos, neg])[::-1]
    tpr = np.r_[0, [(pos >= t).mean() for t in thr]]
    fpr = np.r_[0, [(neg >= t).mean() for t in thr]]
    return fpr, tpr


def cm_metrics(tp, fn, fp, tn):
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    out = {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "sensitivity": sens, "specificity": spec,
           "balanced_accuracy": (sens + spec) / 2}
    for prev in (0.5, 0.1):
        ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev)) if sens * prev + (1 - spec) * (1 - prev) else np.nan
        npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev) if spec * (1 - prev) + (1 - sens) * prev else np.nan
        out[f"PPV_prev{int(prev * 100)}"], out[f"NPV_prev{int(prev * 100)}"] = ppv, npv
    TP, FN, FP, TN = sens, 1 - sens, 1 - spec, spec   # MCC and F1 on a balanced design
    den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    out["MCC_balanced"] = (TP * TN - FP * FN) / den if den > 0 else np.nan
    out["F1_balanced"] = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else np.nan
    return out


def load_cv_null(summary=None):
    fs = sorted(glob.glob(str(RES / "v3_cv_sim_*.csv")))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(subset=["cond", "map", "model", "scheme"])
    n = d.groupby(["cond", "model", "scheme"])["map"].nunique()
    if n.min() != n.max():
        print("WARNING: unequal numbers of maps per cell", n.min(), n.max())
    if summary is not None:
        summary["cv_maps_per_cell"] = {str(k): int(v) for k, v in n.items()}
    return d


# ======================================================================================================================
# main
# ======================================================================================================================
def main_statistics():
    rng = np.random.default_rng(2424)       # one stream, used by parts A to E in this order
    summary = {}

    # A. map-to-map correlations ----------------------------------------------------------------------------------
    f = RES / "v3_bivariate_sim.csv.gz"
    if f.exists():
        d = pd.read_csv(f)
        tb = d[d["pvalue"] == "enigma_toolbox"].copy()
        tb["pvalue"] = "enigma_toolbox_doubled"
        tb["p"] = np.minimum(1.0, 2 * tb["p"])        # the toolbox's directional P value doubled (two-sided reading)
        d = pd.concat([d, tb], ignore_index=True)
        d["sig"] = d["p"] < 0.05
        rows = []
        for (cond, nt, pv), g in d.groupby(["cond", "null", "pvalue"]):
            per_map = g.groupby("map")["sig"].mean().to_numpy()
            n_maps = len(per_map)
            boots = [per_map[rng.integers(0, n_maps, n_maps)].mean() for _ in range(2000)]
            t = stats.ttest_1samp(per_map, 0.05)
            rows.append({"cond": cond, "null": nt, "pvalue": pv, "n_maps": n_maps, "n_tests": len(g),
                         "fpr": g["sig"].mean(), "lo": np.quantile(boots, 0.025), "hi": np.quantile(boots, 0.975),
                         "t_vs_0.05": t.statistic, "p_vs_0.05": t.pvalue})
        t = pd.DataFrame(rows)
        t["p_holm"] = multipletests(t["p_vs_0.05"], method="holm")[1]
        t["bradley"] = [bradley(f_, lo, hi) for f_, lo, hi in zip(t["fpr"], t["lo"], t["hi"])]
        t.to_csv(RES / "v3_stats_bivariate_fpr.csv", index=False)
        rows = []
        two = d[d["pvalue"] == "two_sided"]
        for cond, g in two.groupby("cond"):
            piv = g.pivot_table(index=["map", "annotation"], columns="null", values="sig")
            maps = piv.index.get_level_values(0).to_numpy()
            um = np.unique(maps)
            per = {nt: pd.Series(piv[nt].to_numpy(), index=maps).groupby(level=0).mean().reindex(um).to_numpy()
                   for nt in piv.columns}
            for nt in ["shuffle", "spin_enigma", "spin_std", "spin_fslr", "spin_aligned"]:
                diff = per[nt] - per["spin_sym"]
                bs = [diff[rng.integers(0, len(um), len(um))].mean() for _ in range(2000)]
                rows.append({"cond": cond, "contrast": f"{nt} - spin_sym", "diff": diff.mean(),
                             "lo": np.quantile(bs, 0.025), "hi": np.quantile(bs, 0.975),
                             "p_wilcoxon": stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0})
        c = pd.DataFrame(rows)
        c["p_holm"] = multipletests(c["p_wilcoxon"], method="holm")[1]
        c.to_csv(RES / "v3_stats_bivariate_contrasts.csv", index=False)
        summary["bivariate"] = {r["cond"] + "|" + r["null"] + "|" + r["pvalue"]: [r["fpr"], r["lo"], r["hi"]]
                                for r in t.to_dict("records")}
    print("A. map-to-map correlations done", flush=True)

    # B. cross-validation null simulation -------------------------------------------------------------------------
    d = load_cv_null(summary)
    if d is not None:
        rows = []
        for (cond, m, s), g in d.groupby(["cond", "model", "scheme"]):
            for nt in NULLS_CV:
                k, n = int((g[f"p_{nt}"] < 0.05).sum()), len(g)
                lo, hi = wilson(k, n)
                rows.append({"cond": cond, "model": m, "scheme": s, "null": nt, "k": k, "n": n, "fpr": k / n,
                             "lo": lo, "hi": hi, "mc_se": np.sqrt(k / n * (1 - k / n) / n),
                             "p_binom_vs_0.05": binom_two_sided(k, n)})
        t = pd.DataFrame(rows)
        t["p_holm"] = t.groupby("cond")["p_binom_vs_0.05"].transform(lambda p: multipletests(p, method="holm")[1])
        t["verdict"] = np.where(t["p_holm"] >= 0.05, "not different from 5%",
                                np.where(t["fpr"] > 0.05, "liberal", "conservative"))
        t["bradley"] = [bradley(f_, lo, hi) for f_, lo, hi in zip(t["fpr"], t["lo"], t["hi"])]
        t.to_csv(RES / "v3_stats_cv_fpr.csv", index=False)
        summary["cv_fpr"] = {f"{r['cond']}|{r['model']}|{r['scheme']}|{r['null']}":
                             [r["fpr"], r["lo"], r["hi"], r["verdict"]] for r in t.to_dict("records")}
        # Cochran's Q across nulls and McNemar contrasts
        qrows, mrows = [], []
        for (cond, m), g in d.groupby(["cond", "model"]):
            piv = {s: gg.set_index("map") for s, gg in g.groupby("scheme")}
            for s, gg in piv.items():
                M = np.column_stack([(gg[f"p_{nt}"] < 0.05).to_numpy() for nt in NULLS_CV])
                Q, pq = cochran_q(M)
                qrows.append({"cond": cond, "model": m, "scheme": s, "Q": Q, "df": len(NULLS_CV) - 1, "p": pq})
                for nt in ["shuffle", "spin_std", "spin_fslr", "spin_aligned"]:
                    n10, n01, p = mcnemar_exact((gg[f"p_{nt}"] < 0.05).to_numpy(), (gg["p_spin_sym"] < 0.05).to_numpy())
                    mrows.append({"cond": cond, "model": m, "family": "nulls", "contrast": f"{s}: {nt} vs spin_sym",
                                  "n_only_first": n10, "n_only_second": n01, "p_exact": p})
            common = sorted(set.intersection(*[set(v.index) for v in piv.values()]))
            for a_s, b_s in [("R", "SH"), ("H", "SH"), ("Sw", "SH"), ("R", "H"), ("DD", "DD-H"), ("DD", "SH"),
                             ("R", "DD-H")]:
                if a_s in piv and b_s in piv:
                    for nt in ["spin_sym", "spin_std"]:
                        n10, n01, p = mcnemar_exact((piv[a_s].loc[common, f"p_{nt}"] < 0.05).to_numpy(),
                                                    (piv[b_s].loc[common, f"p_{nt}"] < 0.05).to_numpy())
                        mrows.append({"cond": cond, "model": m, "family": "schemes", "contrast": f"{nt}: {a_s} vs {b_s}",
                                      "n_only_first": n10, "n_only_second": n01, "p_exact": p})
        pd.DataFrame(qrows).to_csv(RES / "v3_stats_cv_cochranq.csv", index=False)
        mc = pd.DataFrame(mrows)
        mc["p_holm"] = mc.groupby(["cond", "model", "family"])["p_exact"].transform(
            lambda p: multipletests(p, method="holm")[1])
        mc.to_csv(RES / "v3_stats_cv_mcnemar.csv", index=False)
        # null optimism: statistic under the null (R2 for partition schemes; mean test r for DD schemes)
        orows, wrows, nrows = [], [], []
        for (cond, m), g in d.groupby(["cond", "model"]):
            piv = g.pivot_table(index="map", columns="scheme", values="stat")
            piv_sk = g.pivot_table(index="map", columns="scheme", values="skill")
            for s in piv.columns:
                v = piv[s].dropna().to_numpy()
                bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
                orows.append({"cond": cond, "model": m, "scheme": s, "mean_stat": v.mean(), "lo": np.quantile(bs, 0.025),
                              "hi": np.quantile(bs, 0.975), "median_stat": np.median(v),
                              "mean_skill": piv_sk[s].mean(), "p_wilcoxon_vs_0": stats.wilcoxon(v).pvalue})
            for a_s, b_s in [("R", "SH"), ("R", "H"), ("H", "SH"), ("Sw", "SH"), ("R", "Sw"), ("DD", "DD-H")]:
                if a_s in piv and b_s in piv:
                    dd_ = (piv[a_s] - piv[b_s]).dropna().to_numpy()
                    w = stats.wilcoxon(dd_)
                    hl_, lo, hi = hodges_lehmann(dd_)
                    n = len(dd_)
                    rbc = 1 - 2 * min(w.statistic, n * (n + 1) / 2 - w.statistic) / (n * (n + 1) / 2)
                    wrows.append({"cond": cond, "model": m, "contrast": f"{a_s} - {b_s}", "n": n, "mean_diff": dd_.mean(),
                                  "hl": hl_, "hl_lo": lo, "hl_hi": hi, "W": w.statistic, "p": w.pvalue,
                                  "rank_biserial": rbc * np.sign(np.median(dd_))})
            part = [c for c in ["R", "H", "S", "SH", "Sw"] if c in piv.columns]
            chi2, pf, nem = nemenyi(piv[part].dropna().to_numpy(), part)
            nem.insert(0, "model", m)
            nem.insert(0, "cond", cond)
            nem["friedman_chi2"], nem["friedman_p"] = chi2, pf
            nrows.append(nem)
        pd.DataFrame(orows).to_csv(RES / "v3_stats_cv_optimism.csv", index=False)
        w = pd.DataFrame(wrows)
        w["p_holm"] = w.groupby(["cond", "model"])["p"].transform(lambda p: multipletests(p, method="holm")[1])
        w.to_csv(RES / "v3_stats_cv_wilcoxon.csv", index=False)
        pd.concat(nrows).to_csv(RES / "v3_stats_cv_friedman_nemenyi.csv", index=False)
    print("B. cross-validation null simulation done", flush=True)

    # C. positive control -----------------------------------------------------------------------------------------
    fs = sorted(glob.glob(str(RES / "v3_power_s*.csv")))
    d0 = load_cv_null()
    if fs and d0 is not None:
        pw = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(subset=["s", "map", "model", "scheme"])
        nul = d0[d0["cond"] == "lam77"]
        arow, crow, droc, prow = [], [], [], []
        for (m, s), g in pw.groupby(["model", "scheme"]):
            gn = nul[(nul["model"] == m) & (nul["scheme"] == s)].set_index("map")
            for nt in ["spin_std", "spin_fslr", "spin_sym"]:
                zneg = ((gn["stat"] - gn[f"mu_{nt}"]) / gn[f"sd_{nt}"]).to_numpy()
                pneg = gn[f"p_{nt}"].to_numpy()
                # size-adjusted threshold: 95th percentile of the z-scores of the signal-free maps, so that exactly 5%
                # of them are declared positive. Monte Carlo P values (500 draws) are discrete and tie at their lower
                # end, so a percentile of the null P values cannot fix the false-positive rate.
                zcrit = float(np.quantile(zneg, 0.95))
                for lvl, gp in g.groupby("s"):
                    gp = gp.set_index("map")
                    zpos = ((gp["stat"] - gp[f"mu_{nt}"]) / gp[f"sd_{nt}"]).to_numpy()
                    ppos = gp[f"p_{nt}"].to_numpy()
                    a = auc_mw(zpos, zneg)
                    bs = [auc_mw(zpos[rng.integers(0, len(zpos), len(zpos))], zneg[rng.integers(0, len(zneg), len(zneg))])
                          for _ in range(1000)]
                    arow.append({"model": m, "scheme": s, "null": nt, "s": lvl, "auc": a, "lo": np.quantile(bs, 0.025),
                                 "hi": np.quantile(bs, 0.975), "auc_from_p": auc_mw(-ppos, -pneg), "n_pos": len(zpos),
                                 "n_neg": len(zneg)})
                    fpr_, tpr_ = roc_curve(zpos, zneg)
                    grid = np.linspace(0, 1, 101)
                    droc.append(pd.DataFrame({"model": m, "scheme": s, "null": nt, "s": lvl, "fpr": grid,
                                              "tpr": np.interp(grid, fpr_, tpr_)}))
                    for thr_name, hit_pos, hit_neg, pc, zc in [
                            ("nominal_0.05", ppos <= 0.05, pneg <= 0.05, 0.05, np.nan),
                            ("size_adjusted", zpos > zcrit, zneg > zcrit, np.nan, zcrit)]:
                        tp, fn = int(hit_pos.sum()), int((~hit_pos).sum())
                        fp, tn = int(hit_neg.sum()), int((~hit_neg).sum())
                        crow.append({"model": m, "scheme": s, "null": nt, "s": lvl, "threshold": thr_name, "p_crit": pc,
                                     "z_crit": zc, **cm_metrics(tp, fn, fp, tn)})
                est = g.groupby("s").agg(true_share=("realised_share", "median"), skill_median=("skill", "median"),
                                         skill_q25=("skill", lambda v: np.quantile(v, 0.25)),
                                         skill_q75=("skill", lambda v: np.quantile(v, 0.75)),
                                         stat_median=("stat", "median")).reset_index()
                if nt == "spin_sym":
                    est.insert(0, "scheme", s)
                    est.insert(0, "model", m)
                    prow.append(est)
        au = pd.DataFrame(arow)
        au.to_csv(RES / "v3_stats_auc.csv", index=False)
        pd.concat(droc).to_csv(RES / "v3_stats_roc_curves.csv", index=False)
        pd.DataFrame(crow).to_csv(RES / "v3_stats_confusion.csv", index=False)
        pd.concat(prow).to_csv(RES / "v3_stats_estimation.csv", index=False)
        rows = []    # paired bootstrap AUC differences between schemes (same null maps, same positive maps)
        for (m, nt, lvl), g in au.groupby(["model", "null", "s"]):
            Z = {}
            for s in g["scheme"]:
                gn = nul[(nul["model"] == m) & (nul["scheme"] == s)].set_index("map").sort_index()
                gp = pw[(pw["model"] == m) & (pw["scheme"] == s) & (pw["s"] == lvl)].set_index("map").sort_index()
                Z[s] = (((gp["stat"] - gp[f"mu_{nt}"]) / gp[f"sd_{nt}"]).to_numpy(),
                        ((gn["stat"] - gn[f"mu_{nt}"]) / gn[f"sd_{nt}"]).to_numpy())
            for a_s, b_s in [("SH", "R"), ("SH", "H"), ("SH", "Sw"), ("DD-H", "DD"), ("SH", "DD-H"), ("H", "R")]:
                if a_s in Z and b_s in Z:
                    obs = auc_mw(*Z[a_s]) - auc_mw(*Z[b_s])
                    npos, nneg = len(Z[a_s][0]), len(Z[a_s][1])
                    bs = []
                    for _ in range(1000):
                        ip, ineg = rng.integers(0, npos, npos), rng.integers(0, nneg, nneg)
                        bs.append(auc_mw(Z[a_s][0][ip], Z[a_s][1][ineg]) - auc_mw(Z[b_s][0][ip], Z[b_s][1][ineg]))
                    bs = np.array(bs)
                    p = min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)) + 1 / len(bs))
                    rows.append({"model": m, "null": nt, "s": lvl, "contrast": f"AUC({a_s}) - AUC({b_s})", "diff": obs,
                                 "lo": np.quantile(bs, 0.025), "hi": np.quantile(bs, 0.975), "p_boot": p})
        ad = pd.DataFrame(rows)
        ad["p_holm"] = ad.groupby(["model", "null"])["p_boot"].transform(lambda p: multipletests(p, method="holm")[1])
        ad.to_csv(RES / "v3_stats_auc_contrasts.csv", index=False)
    print("C. positive control done", flush=True)

    # D. ENIGMA maps ----------------------------------------------------------------------------------------------
    f = RES / "v3_real_cv.csv"
    if f.exists():
        d = pd.read_csv(f)
        nulls = [c[2:] for c in d.columns if c.startswith("p_")]
        rows = []
        for (st, m, s), g in d.groupby(["set", "model", "scheme"]):
            for nt in nulls:
                rows.append({"set": st, "model": m, "scheme": s, "null": nt, "n_p05": int((g[f"p_{nt}"] < 0.05).sum()),
                             "n_bh05": int((g[f"q_{nt}"] < 0.05).sum()), "n_holm05": int((g[f"holm_{nt}"] < 0.05).sum()),
                             "maps_bh": ",".join(g.loc[g[f"q_{nt}"] < 0.05, "map"])})
        c = pd.DataFrame(rows)
        c.to_csv(RES / "v3_stats_real_counts.csv", index=False)
        wrows = []
        for (st, m), g in d.groupby(["set", "model"]):
            piv = g.pivot_table(index="map", columns="scheme", values="stat")
            for a_s, b_s in [("R", "SH"), ("R", "H"), ("Sw", "SH"), ("DD", "DD-H")]:
                dd_ = (piv[a_s] - piv[b_s]).to_numpy()
                w = stats.wilcoxon(dd_)
                hl_, lo, hi = hodges_lehmann(dd_)
                wrows.append({"set": st, "model": m, "contrast": f"{a_s} - {b_s}", "median_diff": np.median(dd_),
                              "hl": hl_, "hl_lo": lo, "hl_hi": hi, "W": w.statistic, "p_exact": w.pvalue})
        w = pd.DataFrame(wrows)
        w["p_holm"] = multipletests(w["p_exact"], method="holm")[1]
        w.to_csv(RES / "v3_stats_real_wilcoxon.csv", index=False)
        summary["real_counts"] = {f"{r['set']}|{r['model']}|{r['scheme']}|{r['null']}": [r["n_bh05"], r["n_holm05"]]
                                  for r in c.to_dict("records")}
    print("D. ENIGMA maps done", flush=True)

    # E. least-squares distance-dependent analysis; approximate correspondence ------------------------------------
    f = RES / "v3_dd_ols_sim.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows = []
        for s_, g in d.groupby("scheme"):
            boot = [g["mean_test_r"].sample(frac=1, replace=True, random_state=b).mean() for b in range(1000)]
            r = {"scheme": s_, "n_maps": len(g), "mean_test_r": g["mean_test_r"].mean(),
                 "mean_test_r_lo": np.quantile(boot, 0.025), "mean_test_r_hi": np.quantile(boot, 0.975)}
            for nt in ["shuffle", "spin_std", "spin_sym"]:
                k, n = int((g[f"p_{nt}"] < 0.05).sum()), len(g)
                lo, hi = wilson(k, n)
                r.update({f"fpr_{nt}": k / n, f"lo_{nt}": lo, f"hi_{nt}": hi})
            rows.append(r)
        pd.DataFrame(rows).to_csv(RES / "v3_stats_dd_ols.csv", index=False)
    f = RES / "v3_approx_coupling_cv.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows = []
        for (m, s_), g in d.groupby(["model", "scheme"]):
            for nt in NULLS_CV:
                k, n = int((g[f"p_{nt}"] < 0.05).sum()), len(g)
                lo, hi = wilson(k, n)
                rows.append({"model": m, "scheme": s_, "null": nt, "fpr": k / n, "lo": lo, "hi": hi, "n": n,
                             "mean_stat": g["stat"].mean(), "bradley": bradley(k / n, lo, hi)})
        pd.DataFrame(rows).to_csv(RES / "v3_stats_approx_coupling_cv.csv", index=False)
    print("E. distance-dependent least squares and approximate correspondence done", flush=True)
    with open(RES / "v3_stats_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=float)


# ======================================================================================================================
# generative null models
# ======================================================================================================================
def generative_statistics():
    rng = np.random.default_rng(2929)

    def parts(pattern):
        fs = sorted(RES.glob(pattern))
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None

    # map-to-map correlations: generative nulls (500 draws) and, on the same maps, the spins of the main analysis
    conds = {"lam77_rho0.9": 500, "lam77_rho0.0": 250}                    # maps used per condition
    g = parts("v3_genpart_biv_*.csv").drop_duplicates(["cond", "null", "annotation", "map"])
    main = pd.read_csv(RES / "v3_bivariate_sim.csv.gz")
    main = main[(main["pvalue"] == "two_sided") & main["cond"].isin(list(conds))]
    keep = [main[(main["cond"] == cond) & (main["map"] < n_used) &
                 main["null"].isin(["shuffle", "spin_std", "spin_enigma", "spin_sym"])] for cond, n_used in conds.items()]
    d = pd.concat([g, pd.concat(keep)[["cond", "null", "annotation", "map", "r", "p"]]], ignore_index=True)
    d["sig"] = d["p"] < 0.05
    rows = []
    for (cond, nt), gg in d.groupby(["cond", "null"]):
        per_map = gg.groupby("map")["sig"].mean().to_numpy()
        n = len(per_map)
        boots = [per_map[rng.integers(0, n, n)].mean() for _ in range(2000)]
        rows.append({"cond": cond, "null": nt, "n_maps": n, "n_tests": len(gg), "fpr": gg["sig"].mean(),
                     "lo": np.quantile(boots, 0.025), "hi": np.quantile(boots, 0.975),
                     "p_vs_0.05": stats.ttest_1samp(per_map, 0.05).pvalue})
    t = pd.DataFrame(rows)
    t["p_holm"] = multipletests(t["p_vs_0.05"], method="holm")[1]
    t["bradley"] = [bradley(f_, lo, hi) for f_, lo, hi in zip(t["fpr"], t["lo"], t["hi"])]
    t.to_csv(RES / "v3_stats_gen_biv_fpr.csv", index=False)
    rows = []    # paired contrasts with the symmetric and the conventional spin on the same maps
    for cond, gg in d.groupby("cond"):
        per = gg.groupby(["null", "map"])["sig"].mean().unstack("null")
        for nt in GEN_VARIANTS:
            for ref in ["spin_sym", "spin_std"]:
                diff = (per[nt] - per[ref]).to_numpy()
                bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
                rows.append({"cond": cond, "contrast": f"{nt} - {ref}", "diff": diff.mean(),
                             "lo": np.quantile(bs, 0.025), "hi": np.quantile(bs, 0.975),
                             "p_wilcoxon": stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0})
    c = pd.DataFrame(rows)
    c["p_holm"] = multipletests(c["p_wilcoxon"], method="holm")[1]
    c.to_csv(RES / "v3_stats_gen_biv_contrasts.csv", index=False)
    b = parts("v3_genpart_bivbil_*.csv").drop_duplicates(["cond", "map", "null"])
    b.groupby(["cond", "null"])[["bil_obs", "bil_null_median"]].median().reset_index().to_csv(
        RES / "v3_stats_gen_biv_bilateral.csv", index=False)

    # cross-validation: generative nulls and, on the same 200 maps, the spins of the main simulation
    cvd = parts("v3_genpart_cv_*.csv").drop_duplicates(["map", "model", "scheme"])
    ms = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("v3_cv_sim_lam77_*.csv"))])
    ms = ms[(ms["map"] < 200) & ms["scheme"].isin(["R", "Sw", "SH", "DD-H"])].drop_duplicates(["map", "model", "scheme"])
    rows = []
    for (model, s), gg in cvd.groupby(["model", "scheme"]):
        ref = ms[(ms["model"] == model) & (ms["scheme"] == s)]
        cols = [(v, gg[f"p_{v}"]) for v in GEN_CV_VARIANTS] + [(nt, ref[f"p_{nt}"]) for nt in ["shuffle", "spin_std", "spin_sym"]]
        for nt, p in cols:
            k, n = int((p < 0.05).sum()), int(p.notna().sum())
            lo, hi = wilson(k, n)
            rows.append({"model": model, "scheme": s, "null": nt, "k": k, "n": n, "fpr": k / n, "lo": lo, "hi": hi,
                         "p_binom": stats.binomtest(k, n, 0.05).pvalue, "bradley": bradley(k / n, lo, hi)})
    t = pd.DataFrame(rows)
    t["p_holm"] = multipletests(t["p_binom"], method="holm")[1]
    t.to_csv(RES / "v3_stats_gen_cv_fpr.csv", index=False)
    if (RES / "v3_gen_bilateral.csv").exists():
        bb = pd.read_csv(RES / "v3_gen_bilateral.csv")
        bb.groupby("variant")[["bil_obs", "bil_null_median"]].median().reset_index().to_csv(
            RES / "v3_stats_gen_bilateral.csv", index=False)
    print(pd.read_csv(RES / "v3_stats_gen_biv_fpr.csv").round(3).to_string())
    print(t.round(3).to_string())


# ======================================================================================================================
# extensions: one value per homologous pair (34 regions) and a fine parcellation (Glasser, 360 regions)
# ======================================================================================================================
def _biv_fpr(d, keys, rng):
    """False-positive rates of map-to-map correlation tests as in part A of main: map-level bootstrap CIs (2,000
    resamples), one-sample t-tests of the per-map rejection proportions against 5% (Holm) and the Bradley class."""
    d = d.assign(sig=d["p"] < 0.05)
    rows = []
    for k, g in d.groupby(keys, sort=False):
        per_map = g.groupby("map")["sig"].mean().to_numpy()
        n = len(per_map)
        boots = [per_map[rng.integers(0, n, n)].mean() for _ in range(2000)]
        rows.append({**dict(zip(keys, k)), "n_maps": n, "n_tests": len(g), "fpr": g["sig"].mean(),
                     "lo": np.quantile(boots, 0.025), "hi": np.quantile(boots, 0.975),
                     "p_vs_0.05": stats.ttest_1samp(per_map, 0.05).pvalue})
    t = pd.DataFrame(rows)
    t["p_holm"] = multipletests(t["p_vs_0.05"], method="holm")[1]
    t["bradley"] = [bradley(f_, lo, hi) for f_, lo, hi in zip(t["fpr"], t["lo"], t["hi"])]
    return t


def _cv_fpr(d, keys, nulls, holm_by=None):
    """Cross-validation false-positive rates as in part B of main: Wilson CIs, exact binomial tests against 5% (Holm
    within holm_by, or across all rows) and the Bradley class; mean statistic of the signal-free maps and mean of their
    null distributions (equal if the null reproduces the dependence that the scheme lets leak)."""
    rows = []
    for k, g in d.groupby(keys, sort=False):
        for nt in nulls:
            kk, n = int((g[f"p_{nt}"] < 0.05).sum()), len(g)
            lo, hi = wilson(kk, n)
            rows.append({**dict(zip(keys, k)), "null": nt, "k": kk, "n": n, "fpr": kk / n, "lo": lo, "hi": hi,
                         "mean_stat": g["stat"].mean(), "mean_null": g[f"mu_{nt}"].mean(),
                         "p_binom_vs_0.05": binom_two_sided(kk, n)})
    t = pd.DataFrame(rows)
    holm = lambda p: multipletests(p, method="holm")[1]
    t["p_holm"] = t.groupby(holm_by)["p_binom_vs_0.05"].transform(holm) if holm_by else holm(t["p_binom_vs_0.05"])
    t["bradley"] = [bradley(f_, lo, hi) for f_, lo, hi in zip(t["fpr"], t["lo"], t["hi"])]
    return t


def extensions():
    rng = np.random.default_rng(3333)
    spin = {"spin_std": "spin"}
    # one value per homologous pair: map-to-map correlations (simulated maps; ENIGMA maps)
    d = pd.read_csv(RES / "v3_half_biv_sim.csv.gz")
    _biv_fpr(d, ["how", "null"], rng).to_csv(RES / "v3_stats_half_biv.csv", index=False)
    r = pd.read_csv(RES / "v3_half_biv_real.csv")
    r.groupby(["how", "null"], sort=False).agg(
        n_p05=("p", lambda p: int((p < 0.05).sum())), n_q05=("q_bh", lambda q: int((q < 0.05).sum())),
        n_holm05=("p_holm", lambda q: int((q < 0.05).sum()))).reset_index().to_csv(
        RES / "v3_stats_half_biv_real.csv", index=False)
    # cross-validation: false-positive rates, and AUC and power in the positive control (score: z relative to the spin)
    cv = pd.concat([pd.read_csv(RES / f"v3_half_cv_{h}.csv") for h in ("avg", "left")], ignore_index=True)
    null = cv[cv["cond"] == "signal_free"]
    t = _cv_fpr(null, ["how", "model", "scheme"], ["shuffle", "spin_std"], holm_by="how")
    t["null"] = t["null"].replace(spin)
    t.to_csv(RES / "v3_stats_half_cv_fpr.csv", index=False)
    rows = []
    for (how, m, s_), g in cv.groupby(["how", "model", "scheme"], sort=False):
        gn = g[g["cond"] == "signal_free"]
        zneg = ((gn["stat"] - gn["mu_spin_std"]) / gn["sd_spin_std"]).to_numpy()
        zcrit = float(np.quantile(zneg, 0.95))
        for lvl in ["s0.1", "s0.3", "s0.5"]:
            gp = g[g["cond"] == lvl]
            zpos = ((gp["stat"] - gp["mu_spin_std"]) / gp["sd_spin_std"]).to_numpy()
            bs = [auc_mw(zpos[rng.integers(0, len(zpos), len(zpos))], zneg[rng.integers(0, len(zneg), len(zneg))])
                  for _ in range(1000)]
            rows.append({"how": how, "model": m, "scheme": s_, "s": float(lvl[1:]), "auc": auc_mw(zpos, zneg),
                         "lo": np.quantile(bs, 0.025), "hi": np.quantile(bs, 0.975), "z_crit": zcrit,
                         "power_size_adjusted": float((zpos > zcrit).mean()),
                         "power_nominal": float((gp["p_spin_std"] < 0.05).mean()),
                         "fpr_nominal": float((gn["p_spin_std"] < 0.05).mean()), "skill_median": gp["skill"].median(),
                         "n_pos": len(zpos), "n_neg": len(zneg)})
    pd.DataFrame(rows).to_csv(RES / "v3_stats_half_power.csv", index=False)
    # ENIGMA maps: number of significant maps
    h = pd.read_csv(RES / "v3_half_real_cv.csv")
    rows = []
    for (how, m, s_), g in h.groupby(["how", "model", "scheme"], sort=False):
        for nt in ["shuffle", "spin"]:
            rows.append({"how": how, "model": m, "scheme": s_, "null": nt, "median_stat": g["stat"].median(),
                         "n_p05": int((g[f"p_{nt}"] < 0.05).sum()), "n_bh05": int((g[f"q_{nt}"] < 0.05).sum()),
                         "n_holm05": int((g[f"holm_{nt}"] < 0.05).sum()),
                         "maps_bh": ",".join(g.loc[g[f"q_{nt}"] < 0.05, "map"])})
    pd.DataFrame(rows).to_csv(RES / "v3_stats_half_real.csv", index=False)
    print("one value per homologous pair done", flush=True)

    # Glasser parcellation: map-to-map correlations (with the bilateral structure of the null maps) and cross-validation
    g = pd.read_csv(RES / "v3_glasser_biv.csv.gz")
    t = _biv_fpr(g, ["cond", "null"], rng)
    b = pd.read_csv(RES / "v3_glasser_bilateral.csv").groupby("null").agg(
        bilateral_r_maps=("bilateral_r", "median"), bilateral_r_null=("median_null", "median"))
    t.merge(b, left_on="null", right_index=True).to_csv(RES / "v3_stats_glasser_biv.csv", index=False)
    fs = sorted(RES.glob("v3_glasser_cv_*.csv"))
    if not fs:
        print("Glasser cross-validation: no result files yet", flush=True)
        return
    gc = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["map", "model", "scheme"])
    _cv_fpr(gc, ["model", "scheme"], [c[2:] for c in gc.columns if c.startswith("p_")]).to_csv(
        RES / "v3_stats_glasser_cv_fpr.csv", index=False)
    print(f"Glasser parcellation done ({gc['map'].nunique()} maps in the cross-validation simulation)", flush=True)


STEPS = {"main": main_statistics, "generative": generative_statistics, "extensions": extensions}

if __name__ == "__main__":
    for step in (sys.argv[1:] or list(STEPS)):
        if step not in STEPS:
            raise SystemExit(__doc__)
        STEPS[step]()
