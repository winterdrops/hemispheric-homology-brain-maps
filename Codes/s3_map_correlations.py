"""
s3_map_correlations.py - step 3: tests of map-to-map correlations (no machine learning; Figs 2 and 5a,
Supplementary Fig. 6, Supplementary Tables 2a, 4a-b and 8a-c).

  sim        simulated null maps (Gaussian random fields independent of the annotations; lambda and rho varied)
             correlated with the 28 annotations; two-sided P values for shuffle and five spins (1,000 draws); the
             ENIGMA Toolbox procedure exactly as implemented; 28 simulated annotations as a robustness check
  real       the 336 ENIGMA disorder-annotation correlations (10,000 draws per null; the ENIGMA Toolbox procedure with
             1,000 rotations), BH-FDR and Holm, and the joint null for the number of significant correlations
  approx     simulated maps with only approximate hemispheric correspondence (500 maps; 1,000 draws)
  gen_bilateral          bilateral correlation of 1,000 generative surrogates of every ENIGMA map
  gen_sim COND I0 I1     generative nulls for simulated maps I0..I1-1 (COND lam77_rho0.9: first 500 maps of the main
                         simulation; lam77_rho0.0: first 250 maps); 500 surrogates
  gen_real               generative nulls for the 336 ENIGMA correlations (10,000 surrogates), BH-FDR and Holm
  gen_selftest           checks of the generators (reproducibility, bilateral structure)
  half       one value per homologous pair (34 regions; homologue average or left hemisphere): simulated maps and the
             ENIGMA correlations; shuffle and a spin of the left sphere
  glasser    fine parcellation (Glasser, 360 regions): simulated maps and simulated annotations; shuffle and five spins
Generative nulls are pre-specified in plan/prereg_generative_nulls.md. Long jobs resume where they stopped when
HEMIHOM_RESUME=1.

Usage: python s3_map_correlations.py STEP [arguments]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

import homology_lib as hl

X, Y, reg, geo = hl.load_data()
Xv, Yv = X.to_numpy(float), Y.to_numpy(float)
RES = hl.RES
RESUME = os.environ.get("HEMIHOM_RESUME") == "1"
SIX_NULLS = ["shuffle", "spin_std", "spin_enigma", "spin_fslr", "spin_aligned", "spin_sym"]
SIM_CONDS = [  # (tag, lambda, rho, n_maps, seed)
    ("lam77_rho0.9", 77, 0.9, 1000, 1701),
    ("lam25_rho0.9", 25, 0.9, 500, 1702),
    ("lam50_rho0.9", 50, 0.9, 500, 1703),
    ("lam100_rho0.9", 100, 0.9, 500, 1704),
    ("lam77_rho0.0", 77, 0.0, 500, 1705),
    ("lam77_rho0.5", 77, 0.5, 500, 1706),
    ("lam77_rho1.0", 77, 1.0, 500, 1707),
]
GEN_SIM_CONDS = {"lam77_rho0.9": (0.9, 1000, 1701, 500), "lam77_rho0.0": (0.0, 500, 1705, 250)}  # rho, n, seed, used


def six_nulls(n_draws, seed0):
    """The six null models with seeds seed0 + 0 .. seed0 + 5."""
    return hl.null_permutations(reg, n_draws, {nt: seed0 + i for i, nt in enumerate(SIX_NULLS)})


# ======================================================================================================================
# sim: simulated null maps
# ======================================================================================================================
def sim():
    t0 = time.time()
    nulls = six_nulls(1000, 17000)
    out = []
    for tag, lam, rho, n_maps, seed in SIM_CONDS:
        G = hl.grf_maps(reg, geo, lam=lam, rho=rho, n_maps=n_maps, seed=seed)
        for nt, perms in nulls.items():
            r, p = hl.corr_pvalues(Xv, G, perms)
            out.append(pd.DataFrame({"cond": tag, "lam": lam, "rho": rho, "null": nt, "pvalue": "two_sided",
                                     "annotation": np.repeat(X.columns, n_maps),
                                     "map": np.tile(np.arange(n_maps), Xv.shape[1]), "r": r.ravel(), "p": p.ravel()}))
        if tag == "lam77_rho0.9":
            # ENIGMA Toolbox procedure exactly as implemented (directional P, averaged over spinning x and y)
            P = nulls["spin_enigma"]
            pe = np.stack([hl.enigma_pvalue(Xv[:, j], G, P) for j in range(Xv.shape[1])])
            out.append(pd.DataFrame({"cond": tag, "lam": lam, "rho": rho, "null": "spin_enigma",
                                     "pvalue": "enigma_toolbox", "annotation": np.repeat(X.columns, n_maps),
                                     "map": np.tile(np.arange(n_maps), Xv.shape[1]),
                                     "r": hl.corr_xy(Xv, G).ravel(), "p": pe.ravel()}))
            # simulated, independent annotation maps from the same generator (robustness)
            Xsim = hl.grf_maps(reg, geo, lam=lam, rho=rho, n_maps=28, seed=1799)
            for nt, perms in nulls.items():
                r, p = hl.corr_pvalues(Xsim, G, perms)
                out.append(pd.DataFrame({"cond": tag + "_synthX", "lam": lam, "rho": rho, "null": nt,
                                         "pvalue": "two_sided", "annotation": np.repeat([f"GRF{j}" for j in range(28)], n_maps),
                                         "map": np.tile(np.arange(n_maps), 28), "r": r.ravel(), "p": p.ravel()}))
        pd.concat(out).to_csv(RES / "v3_bivariate_sim.csv.gz", index=False,
                              compression={"method": "gzip", "mtime": 0})   # byte-reproducible
        print(tag, f"done at {time.time() - t0:.0f}s", flush=True)
    d = pd.concat(out)
    print(d.assign(sig=d["p"] < 0.05).groupby(["cond", "null", "pvalue"])["sig"].mean().unstack([1, 2]).round(3).to_string())


# ======================================================================================================================
# real: ENIGMA disorder-annotation correlations and their joint null
# ======================================================================================================================
def real(joint=True):
    nulls = six_nulls(10000, 18000)
    rows = []
    for nt, perms in nulls.items():
        r, p = hl.corr_pvalues(Xv, Yv, perms, chunk=4)
        for i, a in enumerate(X.columns):
            for j, m in enumerate(Y.columns):
                rows.append({"null": nt, "pvalue": "two_sided", "annotation": a, "map": m, "r": r[i, j], "p": p[i, j]})
    P1000 = nulls["spin_enigma"][:1000]
    for i, a in enumerate(X.columns):
        pe = hl.enigma_pvalue(Xv[:, i], Yv, P1000)
        for j, m in enumerate(Y.columns):
            rows.append({"null": "spin_enigma", "pvalue": "enigma_toolbox", "annotation": a, "map": m,
                         "r": np.corrcoef(Xv[:, i], Yv[:, j])[0, 1], "p": pe[j]})
    d = pd.DataFrame(rows)
    d["q_bh"] = d.groupby(["null", "pvalue"])["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    d["p_holm"] = d.groupby(["null", "pvalue"])["p"].transform(lambda p: multipletests(p, method="holm")[1])
    d.to_csv(RES / "v3_bivariate_real.csv", index=False)
    summ = d.groupby(["null", "pvalue"]).agg(n_p05=("p", lambda p: int((p < 0.05).sum())),
                                              n_q05=("q_bh", lambda q: int((q < 0.05).sum())),
                                              n_holm05=("p_holm", lambda q: int((q < 0.05).sum())))
    summ.to_csv(RES / "v3_bivariate_real_counts.csv")
    print(summ.to_string())
    if joint:
        joint_null(nulls, summ.reset_index())


def joint_null(nulls, counts):
    """Joint null of the number of significant correlations. The 336 tests are dependent, so the count expected by
    chance is not 0.05 x 336 with a binomial spread. All 12 maps are permuted jointly with the same permutation; each
    draw is treated as pseudo-observed data and its 336 two-sided P values are computed against the remaining draws."""
    obs = counts.query("pvalue == 'two_sided'").set_index("null")
    Xc = (Xv - Xv.mean(0)) / Xv.std(0)
    rows = []
    for nt, perms in nulls.items():
        N = len(perms)
        Yp = Yv[perms]                                                  # (N, 68, 12)
        Yp = (Yp - Yp.mean(1, keepdims=True)) / Yp.std(1, keepdims=True)
        rn = np.abs(np.einsum("nq,Nnm->Nqm", Xc, Yp) / 68)              # (N, 28, 12)
        rk = rankdata(rn.reshape(N, -1), axis=0, method="max")          # rank among the N draws
        p_self = (1.0 + N - rk) / N                                     # P value of each draw against the others
        n_p05 = (p_self < 0.05).sum(1)
        n_q05 = np.array([(hl.bh_fdr(p_self[i]) < 0.05).sum() for i in range(0, N, 5)])
        o_p, o_q = int(obs.loc[nt, "n_p05"]), int(obs.loc[nt, "n_q05"])
        rows.append({"null": nt, "observed_p05": o_p, "null_median_p05": float(np.median(n_p05)),
                     "null_q95_p05": float(np.quantile(n_p05, 0.95)), "P_count_p05": float((1 + (n_p05 >= o_p).sum()) / (1 + N)),
                     "observed_q05": o_q, "null_q95_q05": float(np.quantile(n_q05, 0.95)),
                     "P_count_q05": float((1 + (n_q05 >= o_q).sum()) / (1 + len(n_q05)))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "v3_bivariate_joint_null.csv", index=False)


# ======================================================================================================================
# approx: approximate hemispheric correspondence
# ======================================================================================================================
def approx():
    nulls = six_nulls(1000, 27000)
    G = hl.approx_maps(reg, geo, 500, 2701)
    L = np.where(reg["hemi"] == "L")[0]
    hom = reg["homologue_idx"].to_numpy()
    bil = np.array([np.corrcoef(G[L, j], G[hom[L], j])[0, 1] for j in range(G.shape[1])])
    rows = []
    for nt, P in nulls.items():
        r, p = hl.corr_pvalues(Xv, G, P)
        per_map = (p < 0.05).mean(0)
        bs = [per_map[np.random.default_rng(b).integers(0, len(per_map), len(per_map))].mean() for b in range(2000)]
        rows.append({"null": nt, "fpr": per_map.mean(), "lo": np.quantile(bs, 0.025), "hi": np.quantile(bs, 0.975),
                     "median_bilateral_r_of_maps": float(np.median(bil))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "v3_approx_coupling_biv.csv", index=False)


# ======================================================================================================================
# generative null models
# ======================================================================================================================
def _resume(path, key="map"):
    if RESUME and path.exists():
        d = pd.read_csv(path)
        return d, set(d[key].unique())
    return None, set()


def gen_bilateral():
    gn = hl.GenerativeNulls(reg, geo)
    rows = []
    for j, m in enumerate(Y.columns):
        y = Yv[:, j]
        obs = np.corrcoef(y[gn.L], y[gn.R])[0, 1]
        for k, v in enumerate(gn.VARIANTS):
            b = gn.bilateral_r(gn.surrogates(y, v, 1000, 29300000 + 100 * j + 10 * k))
            rows.append({"map": m, "variant": v, "bil_obs": obs, "bil_null_median": float(np.median(b)),
                         "bil_null_q05": float(np.quantile(b, 0.05)), "bil_null_q95": float(np.quantile(b, 0.95)),
                         # statistic of hypothesis H6a as written in plan/prereg_generative_nulls.md
                         "bil_null_median_abs": float(np.median(np.abs(b)))})
        print("bilateral", m, flush=True)
    pd.DataFrame(rows).to_csv(RES / "v3_gen_bilateral.csv", index=False)


def gen_sim(cond, i0, i1):
    gn = hl.GenerativeNulls(reg, geo)
    rho, n_all, seed, n_used = GEN_SIM_CONDS[cond]
    G = hl.grf_maps(reg, geo, lam=77, rho=rho, n_maps=n_all, seed=seed)[:, :n_used]
    i1 = min(i1, n_used)
    out = RES / f"v3_genpart_biv_{cond}_{i0}_{i1}.csv"
    outb = RES / f"v3_genpart_bivbil_{cond}_{i0}_{i1}.csv"
    d0, done = _resume(out)
    rows = d0.to_dict("records") if d0 is not None else []
    b0, _ = _resume(outb)
    brows = b0.to_dict("records") if b0 is not None else []
    base = 29100000 if cond == "lam77_rho0.9" else 29150000
    t0 = time.time()
    for j in range(i0, i1):
        if j in done:
            continue
        y = G[:, j]
        for k, v in enumerate(gn.VARIANTS):
            S = gn.surrogates(y, v, 500, base + 100 * j + 10 * k)
            r, p = hl.corr_pvalues_surrogates(Xv, y, S)
            rows += [{"cond": cond, "null": v, "annotation": a, "map": j, "r": r[i], "p": p[i]}
                     for i, a in enumerate(X.columns)]
            brows.append({"cond": cond, "map": j, "null": v, "bil_obs": float(np.corrcoef(y[gn.L], y[gn.R])[0, 1]),
                          "bil_null_median": float(np.median(gn.bilateral_r(S)))})
        if (j - i0) % 10 == 9 or j == i1 - 1:
            pd.DataFrame(rows).to_csv(out, index=False)
            pd.DataFrame(brows).to_csv(outb, index=False)
            print(f"{cond}: map {j} at {time.time() - t0:.0f}s", flush=True)


def gen_real():
    gn = hl.GenerativeNulls(reg, geo)
    rows = []
    for j, m in enumerate(Y.columns):
        y = Yv[:, j]
        for k, v in enumerate(gn.VARIANTS):
            r, p = hl.corr_pvalues_surrogates(Xv, y, gn.surrogates(y, v, 10000, 29200000 + 100 * j + 10 * k))
            rows += [{"null": v, "annotation": a, "map": m, "r": r[i], "p": p[i]} for i, a in enumerate(X.columns)]
        print("real", m, flush=True)
    d = pd.DataFrame(rows)
    d["q_bh"] = d.groupby("null")["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    d["p_holm"] = d.groupby("null")["p"].transform(lambda p: multipletests(p, method="holm")[1])
    d.to_csv(RES / "v3_gen_real.csv", index=False)
    s = d.groupby("null").agg(n_p05=("p", lambda p: int((p < 0.05).sum())),
                              n_q05=("q_bh", lambda q: int((q < 0.05).sum())),
                              n_holm05=("p_holm", lambda q: int((q < 0.05).sum())))
    s.to_csv(RES / "v3_gen_real_counts.csv")
    print(s.to_string())


def gen_selftest():
    gn = hl.GenerativeNulls(reg, geo)
    y = Yv[:, list(Y.columns).index("SCZ")]
    for v in gn.VARIANTS:
        S1, S2 = gn.surrogates(y, v, 200, 7), gn.surrogates(y, v, 200, 7)
        assert S1.shape == (68, 200) and np.allclose(S1, S2), v                      # shape, reproducibility
    b = {v: np.median(gn.bilateral_r(gn.surrogates(y, v, 500, 11))) for v in gn.VARIANTS}
    assert abs(b["bs_indep"]) < 0.2 and abs(b["msr_indep"]) < 0.2, b                   # independent: no symmetry
    obs = np.corrcoef(y[gn.L], y[gn.R])[0, 1]
    assert abs(b["msr_sym"] - obs) < 0.1 and abs(b["bs_sym"] - obs) < 0.15, (b, obs)  # symmetric: preserved
    Ssym = gn.surrogates(Yv[:, list(Y.columns).index("ADHD")], "msr_sym", 50, 3)      # hemisphere-averaged map
    assert np.allclose(Ssym[gn.L], Ssym[gn.R])
    print("selftest passed", {k: round(float(x), 3) for k, x in b.items()}, "observed", round(obs, 3))


# ======================================================================================================================
# half: one value per homologous pair (34 regions)
# ======================================================================================================================
def half():
    """Map-to-map correlations when every map is reduced to one value per homologous pair (homologue average or left
    hemisphere only): the first 500 maps of the main simulation (lambda = 77 mm, rho = 0.9) against the 28 annotations
    (1,000 draws) and the 336 ENIGMA disorder-annotation correlations (10,000 draws; BH-FDR and Holm); shuffle and a
    spin of the left sphere."""
    G = hl.grf_maps(reg, geo, lam=77, rho=0.9, n_maps=1000, seed=1701)[:, :500]
    sim_rows, real_rows = [], []
    for k, how in enumerate(["avg", "left"]):
        r34, _, f = hl.half_brain(reg, geo, how)
        for n_draws, seed, Ymat, names, out in [(1000, 30000 + 2 * k, f(G), np.arange(G.shape[1]), sim_rows),
                                                 (10000, 30010 + 2 * k, f(Yv), np.array(Y.columns), real_rows)]:
            nulls = hl.null_permutations(r34, n_draws, {"shuffle": seed, "spin_std": seed + 1})
            for nt, P in nulls.items():
                r, p = hl.corr_pvalues(f(Xv), Ymat, P, chunk=50 if len(names) > 12 else 4)
                out.append(pd.DataFrame({"how": how, "null": "spin" if nt == "spin_std" else nt,
                                         "annotation": np.repeat(X.columns, len(names)),
                                         "map": np.tile(names, Xv.shape[1]), "r": r.ravel(), "p": p.ravel()}))
        print(how, "done", flush=True)
    sim_d = pd.concat(sim_rows)
    sim_d.to_csv(RES / "v3_half_biv_sim.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    real_d = pd.concat(real_rows)
    real_d["q_bh"] = real_d.groupby(["how", "null"])["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    real_d["p_holm"] = real_d.groupby(["how", "null"])["p"].transform(lambda p: multipletests(p, method="holm")[1])
    real_d.to_csv(RES / "v3_half_biv_real.csv", index=False)
    print(sim_d.assign(sig=sim_d["p"] < 0.05).groupby(["how", "null"])["sig"].mean().round(3).to_string())
    print(real_d.groupby(["how", "null"]).agg(n_p05=("p", lambda p: int((p < 0.05).sum())),
                                              n_q05=("q_bh", lambda q: int((q < 0.05).sum()))).to_string())


# ======================================================================================================================
# glasser: fine parcellation (360 regions)
# ======================================================================================================================
def glasser():
    """Map-to-map correlations on the Glasser parcellation (360 regions, 180 homologous pairs): 500 null maps against
    28 simulated annotation maps (lambda = 77 mm, rho = 0.9), with bilaterally coupled null maps (rho = 0.9, the maps of
    the cross-validation simulation) and without coupling (rho = 0); shuffle and the five spins (1,000 draws); bilateral
    correlation of the null versions of the annotation maps."""
    rg, gg, Xg, G = hl.glasser_simulation()
    conds = {"lam77_rho0.9": G, "lam77_rho0.0": hl.grf_maps(rg, gg, lam=77, rho=0.0, n_maps=500, seed=31003)}
    nulls = hl.null_permutations(rg, 1000, {nt: 31010 + i for i, nt in enumerate(SIX_NULLS)})
    names = [f"GRF{j}" for j in range(Xg.shape[1])]
    L = np.where(rg["hemi"] == "L")[0]
    hom = rg["homologue_idx"].to_numpy()
    rows, bil = [], []
    for nt, P in nulls.items():
        for cond, M in conds.items():
            r, p = hl.corr_pvalues(Xg, M, P)
            rows.append(pd.DataFrame({"cond": cond, "null": nt, "annotation": np.repeat(names, M.shape[1]),
                                      "map": np.tile(np.arange(M.shape[1]), len(names)), "r": r.ravel(), "p": p.ravel()}))
            print(cond, nt, f"FPR {(p < 0.05).mean():.3f}", flush=True)
        for j, a in enumerate(names):
            b = hl.spin_bilateral_r(Xg[:, j], P, rg)
            bil.append({"null": nt, "annotation": a, "bilateral_r": np.corrcoef(Xg[L, j], Xg[hom[L], j])[0, 1],
                        "median_null": float(np.median(b)), "q05_null": float(np.quantile(b, 0.05))})
    pd.concat(rows).to_csv(RES / "v3_glasser_biv.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(bil).to_csv(RES / "v3_glasser_bilateral.csv", index=False)


STEPS = {"sim": sim, "real": real, "approx": approx, "gen_bilateral": gen_bilateral,
         "gen_sim": lambda cond, i0, i1: gen_sim(cond, int(i0), int(i1)), "gen_real": gen_real,
         "gen_selftest": gen_selftest, "half": half, "glasser": glasser}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        raise SystemExit(__doc__)
    hl.ensure_dirs()
    STEPS[sys.argv[1]](*sys.argv[2:])
