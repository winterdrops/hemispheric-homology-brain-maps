"""
s1_data.py - step 1: data, geometry and descriptive analyses (no outcome statistics).

  prepare   analysis-ready matrices from the public raw files: 12 ENIGMA case-control Cohen's d maps (targets), 28
            annotations (predictors), region geometry (sphere and surface centroids, geodesic distances), descriptive
            spatial statistics and the parity check against Hansen et al. (2022)
  lambda    correlation length: empirical correlogram fits and calibration of the generating lambda by simulation
            (Supplementary Fig. 1)
  geometry  sphere-frame misalignment on fsaverage5 and fs_LR (Fig. 1b, Supplementary Table 1), bilateral structure of
            the null maps (Fig. 1c) and homologue leakage of every cross-validation scheme (Fig. 1e)
  se        region-wise standard errors of the ENIGMA effect sizes (Supplementary Note 4, Supplementary Table 10)
  meff      effective number of independent target maps (Li & Ji 2005; Nyholt 2004)
  glasser   geometry of the Glasser parcellation (360 regions) for the fine-parcellation simulations

Usage: python s1_data.py [prepare|lambda|geometry|se|meff|glasser ...]   (default: all, in this order)
Outputs: data/derived/*, results/data_checks.json, descriptives_spatial.csv, rev_lambda_*.csv, v3_misalignment.csv,
         v3_bilateral_null.csv, v3_null_perms_geometry.npz, v3_scheme_structure.csv, v3_geometry.json,
         v3_target_se*.{csv,json}, v3_meff.json, v3_target_correlations.csv
"""
import json
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

import homology_lib as hl


# ======================================================================================================================
# prepare
# ======================================================================================================================
def prepare():
    raw = hl.RAW
    # 1. region order (ENIGMA / FreeSurfer aparc colour-table order; LH then RH)
    names = pd.read_csv(raw / "enigma/strucLabels_ctx.csv", header=None).iloc[0].str.strip().tolist()
    assert len(names) == 68 and names[0] == "L_bankssts" and names[34] == "R_bankssts"
    hemi = np.array([n[0] for n in names])
    label = np.array([n[2:] for n in names])

    # 2. targets: ENIGMA case-control cortical-thickness Cohen's d maps
    def load_enigma_map(stem):
        df = pd.read_csv(raw / f"enigma/{stem}.csv")
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df["Structure"] = df["Structure"].astype(str).str.strip()
        s = df.set_index("Structure")["d_icv"].astype(float)
        missing = [n for n in names if n not in s.index]
        assert not missing, f"{stem}: missing regions {missing}"
        return s.loc[names].to_numpy()

    targets = pd.DataFrame({k: load_enigma_map(v) for k, v in hl.TARGETS_PRIMARY.items()}, index=names)
    targets_sec = pd.DataFrame({k: load_enigma_map(v) for k, v in hl.TARGETS_SECONDARY.items()}, index=names)

    # 3. predictors
    info = pd.read_excel(raw / "hansen/region_info_scale033.xlsx")
    ctx = info.query("structure == 'cortex'")
    laus_names = (ctx["hemisphere"].str.strip() + "_" + ctx["label"].str.strip()).tolist()
    assert len(laus_names) == 68 and set(laus_names) == set(names), "Lausanne scale033 cortex != DK68"
    laus_to_dk = np.array([laus_names.index(n) for n in names])  # reorder Lausanne -> ENIGMA order

    rec = np.loadtxt(raw / "hansen/receptor_data_scale033.csv", delimiter=",")[laus_to_dk]
    rec_names = [str(r) for r in np.load(raw / "hansen/receptor_names_pet.npy", allow_pickle=True)]
    meg = np.loadtxt(raw / "hansen/power_scale033.csv", delimiter=",")[laus_to_dk]
    meg_names = ["MEG_delta", "MEG_theta", "MEG_alpha", "MEG_beta", "MEG_lowgamma", "MEG_highgamma"]
    mesulam = np.loadtxt(raw / "hansen/mesulam_scale033.csv", delimiter=",")[laus_to_dk]

    sc = pd.read_csv(raw / "enigma/strucMatrix_ctx.csv", header=None).to_numpy(float)
    fc = pd.read_csv(raw / "enigma/funcMatrix_ctx.csv", header=None).to_numpy(float)
    fc_names = pd.read_csv(raw / "enigma/funcLabels_ctx.csv", header=None).iloc[0].str.strip().tolist()
    assert fc_names == names
    np.fill_diagonal(sc, 0)
    np.fill_diagonal(fc, 0)

    predictors = pd.DataFrame(rec, index=names, columns=[f"PET_{r}" for r in rec_names])
    predictors[meg_names] = meg
    predictors["SC_strength"] = sc.sum(1)
    predictors["FC_strength"] = np.where(fc > 0, fc, 0).sum(1)
    predictors["Mesulam_class"] = mesulam
    assert predictors.notna().all().all()

    # 4. parity check: Hansen et al. (2022) ENIGMA maps (Lausanne order) vs our re-extraction
    hansen_ct = np.loadtxt(raw / "hansen/enigma_atrophy.csv", delimiter=",")[laus_to_dk]
    hansen_cols = ['22q', 'adhd', 'asd', 'epilepsy_gge', 'epilepsy_rtle', 'epilepsy_ltle', 'depression',
                   'ocd', 'schizophrenia', 'bipolar', 'obesity', 'schizotypy', 'park']
    ours_for = {'22q': targets["22q11.2DS"], 'adhd': targets["ADHD"], 'asd': targets["ASD"],
                'epilepsy_gge': targets_sec["GGE"], 'epilepsy_rtle': targets_sec["TLE_R"],
                'epilepsy_ltle': targets_sec["TLE_L"], 'depression': targets["MDD"], 'ocd': targets["OCD"],
                'schizophrenia': targets["SCZ"], 'bipolar': targets["BD"], 'schizotypy': targets_sec["Schizotypy"],
                'park': targets["PD"]}
    parity = {}
    for j, c in enumerate(hansen_cols):
        if c in ours_for:
            a, b = hansen_ct[:, j], np.asarray(ours_for[c])
            parity[c] = {"pearson_r": float(np.corrcoef(a, b)[0, 1]), "max_abs_diff": float(np.max(np.abs(a - b)))}

    # 5. geometry on fsaverage5
    aparc = np.loadtxt(raw / "enigma/aparc_fsa5.csv").astype(int)
    nv = 10242
    surf = {h: nib.load(raw / f"enigma/fsa5_{h.lower()}h.gii") for h in "LR"}
    sphr = {h: nib.load(raw / f"enigma/fsa5_sphere_{h.lower()}h.gii") for h in "LR"}
    sph_c, srf_c, medoid, nvert = np.zeros((68, 3)), np.zeros((68, 3)), np.zeros(68, int), np.zeros(68, int)
    for r in range(68):
        h = hemi[r]
        lab = aparc[:nv] if h == "L" else aparc[nv:]
        v = np.where(lab == hl.A_IDX[r])[0]
        assert len(v) > 0
        c = sphr[h].darrays[0].data[v].mean(0)
        xyz = surf[h].darrays[0].data[v]
        sph_c[r] = c / np.linalg.norm(c)
        srf_c[r] = xyz.mean(0)
        medoid[r] = v[np.argmin(np.linalg.norm(xyz - srf_c[r], axis=1))]  # vertex closest to the surface centroid
        nvert[r] = len(v)

    # geodesic distances between these vertices (Dijkstra along mesh edges, within hemisphere)
    geo = np.full((68, 68), np.nan)
    for h in "LR":
        xyz = surf[h].darrays[0].data.astype(float)
        faces = surf[h].darrays[1].data.astype(int)
        e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
        w = np.linalg.norm(xyz[e[:, 0]] - xyz[e[:, 1]], axis=1)
        G = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nv, nv)).tocsr()
        idx = np.where(hemi == h)[0]
        D = dijkstra(G, directed=False, indices=medoid[idx])
        geo[np.ix_(idx, idx)] = D[:, medoid[idx]]
    euc = np.linalg.norm(srf_c[:, None, :] - srf_c[None, :, :], axis=2)

    regions = pd.DataFrame({"name": names, "hemi": hemi, "label": label, "n_vertices": nvert,
                            "sph_x": sph_c[:, 0], "sph_y": sph_c[:, 1], "sph_z": sph_c[:, 2],
                            "x": srf_c[:, 0], "y": srf_c[:, 1], "z": srf_c[:, 2]})
    homologue = np.array([names.index(("R_" if n[0] == "L" else "L_") + n[2:]) for n in names])
    regions["homologue_idx"] = homologue

    # 6. descriptive spatial statistics
    def morans_i(x, W):
        z = x - x.mean()
        return float(len(x) / W.sum() * (z @ W @ z) / (z @ z))

    W = np.zeros((68, 68))
    for h in "LR":
        idx = np.where(hemi == h)[0]
        d = geo[np.ix_(idx, idx)]
        with np.errstate(divide="ignore"):
            W[np.ix_(idx, idx)] = np.where(d > 0, 1.0 / d, 0.0)
    rows = []
    for df, kind in [(targets, "target"), (targets_sec, "target_secondary"), (predictors, "predictor")]:
        for c in df.columns:
            x = df[c].to_numpy(float)
            rows.append({"map": c, "kind": kind, "morans_I": morans_i(x, W),
                         "bilateral_r": float(np.corrcoef(x[:34], x[homologue[:34]])[0, 1]),
                         "mean": float(x.mean()), "sd": float(x.std(ddof=1))})
    desc = pd.DataFrame(rows)

    # 7. save
    targets.to_csv(hl.DER / "targets_primary.csv")
    targets_sec.to_csv(hl.DER / "targets_secondary.csv")
    predictors.to_csv(hl.DER / "predictors.csv")
    regions.to_csv(hl.DER / "regions.csv", index=False)
    np.save(hl.DER / "geodesic.npy", geo)
    np.save(hl.DER / "euclidean.npy", euc)
    desc.to_csv(hl.RES / "descriptives_spatial.csv", index=False)
    checks = {"n_regions": 68, "n_targets_primary": targets.shape[1], "n_predictors": predictors.shape[1],
              "targets_nan": int(targets.isna().sum().sum()), "parity_vs_hansen2022": parity,
              "vertices_per_region_min": int(nvert.min()), "vertices_per_region_max": int(nvert.max())}
    (hl.RES / "data_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    print(desc.round(3).to_string())


# ======================================================================================================================
# lambda: correlation length
# ======================================================================================================================
def calibrate_lambda():
    """(a) Empirical within-hemisphere correlogram fits (targets 51.5 mm, predictors 69.3 mm in the analysis plan);
    (b) the generating lambda whose simulated maps give the same plug-in estimate as the targets (the plug-in estimate
    is biased low because of mean-centring)."""
    X, Y, reg, geo = hl.load_data()
    pairs = []
    for h in "LR":
        idx = np.where(reg["hemi"] == h)[0]
        a, b = np.triu_indices(len(idx), 1)
        pairs.append((idx[a], idx[b]))
    I = np.concatenate([p[0] for p in pairs])
    J = np.concatenate([p[1] for p in pairs])
    D = geo[I, J]

    def corr_length(M):
        Z = (M - M.mean(0)) / M.std(0)
        prod = (Z[I] * Z[J]).mean(1)
        (lam, s), _ = curve_fit(lambda d, lam, s: s * np.exp(-d / lam), D, prod, p0=[30, 0.8], bounds=([1, 0], [500, 1]))
        return lam

    emp_t = corr_length(Y.to_numpy(float))
    emp_p = corr_length(X.to_numpy(float))
    rows = []
    for lam in [30, 40, 50, 60, 70, 80, 90, 100, 120]:
        est = [corr_length(hl.grf_maps(reg, geo, lam=lam, rho=0.9, n_maps=12, seed=1000 + k)) for k in range(40)]
        rows.append({"generator_lambda": lam, "estimated_median": float(np.median(est)),
                     "estimated_iqr_lo": float(np.percentile(est, 25)), "estimated_iqr_hi": float(np.percentile(est, 75))})
    cal = pd.DataFrame(rows)
    cal_lam = float(np.interp(emp_t, cal["estimated_median"], cal["generator_lambda"]))
    cal.to_csv(hl.RES / "rev_lambda_calibration.csv", index=False)
    pd.Series({"empirical_targets_mm": emp_t, "empirical_predictors_mm": emp_p, "calibrated_generator_lambda_mm": cal_lam}) \
        .to_csv(hl.RES / "rev_lambda_summary.csv")
    print(cal.round(1).to_string())
    print(f"empirical targets {emp_t:.1f} mm, predictors {emp_p:.1f} mm; simulation-calibrated lambda {cal_lam:.0f} mm")


# ======================================================================================================================
# geometry: sphere frames, bilateral structure of null maps, structure of the cross-validation schemes
# ======================================================================================================================
def geometry():
    X, Y, reg, geo = hl.load_data()
    raw = hl.RAW_ENIGMA
    F = np.array([-1.0, 1.0, 1.0])

    # (1) misalignment of the hemispheric sphere frames
    def angles(cl, cr):
        a = cl / np.linalg.norm(cl, axis=1, keepdims=True)
        b = (cr * F) / np.linalg.norm(cr, axis=1, keepdims=True)
        return np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))

    rows = []
    for surf in ["fsa5", "conte69"]:
        if surf == "fsa5":
            S = {h: nib.load(raw / f"fsa5_sphere_{h}h.gii").darrays[0].data.astype(float) for h in "lr"}
        else:
            S = {h: nib.load(raw / f"conte69_32k_{h}h_sphere.gii").darrays[0].data.astype(float) for h in "lr"}
        nv = len(S["l"])
        for parc in ["aparc", "glasser_360"]:
            lab = np.loadtxt(raw / f"{parc}_{surf}.csv").astype(int)
            L, R = lab[:nv], lab[nv:]
            if parc == "aparc":
                lk = list(range(1, 4)) + list(range(5, 36))
                rk = {k: k + 35 for k in lk}
            else:
                lk = list(range(1, 181))
                rk = {k: k + 180 for k in lk}
            cl = np.array([S["l"][L == k].mean(0) for k in lk])
            cr = np.array([S["r"][R == rk[k]].mean(0) for k in lk])
            a = angles(cl, cr)
            A = cr / np.linalg.norm(cr, axis=1, keepdims=True)
            B = (cl * F) / np.linalg.norm(cl, axis=1, keepdims=True)
            U, _, Vt = np.linalg.svd(A.T @ B)
            Rk = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            ak = np.degrees(np.arccos(np.clip(((A @ Rk) * B).sum(1), -1, 1)))
            rows.append({"sphere": "fsaverage5" if surf == "fsa5" else "fs_LR 32k (Conte69)",
                         "parcellation": "Desikan-Killiany (68)" if parc == "aparc" else "Glasser (360)",
                         "n_pairs": len(lk), "mean_deg": a.mean(), "median_deg": np.median(a), "max_deg": a.max(),
                         "kabsch_rotation_deg": np.degrees(np.arccos(np.clip((np.trace(Rk) - 1) / 2, -1, 1))),
                         "after_alignment_mean_deg": ak.mean(), "vertex_frames_identical": bool(np.allclose(S["l"], S["r"])),
                         "vertex_frames_mirrored": bool(np.allclose(S["l"] * F, S["r"]))})
    mis = pd.DataFrame(rows)
    mis.to_csv(hl.RES / "v3_misalignment.csv", index=False)
    print(mis.round(2).to_string())

    # (2) bilateral correlation of the null maps of every target and predictor
    nulls = hl.null_permutations(reg, 1000, {"shuffle": 16000, "spin_std": 16001, "spin_enigma": 16002,
                                             "spin_fslr": 16003, "spin_aligned": 16004, "spin_sym": 16005})
    np.savez_compressed(hl.RES / "v3_null_perms_geometry.npz", **nulls)
    Lh = np.where(reg["hemi"].to_numpy() == "L")[0]
    hom = reg["homologue_idx"].to_numpy()
    rows = []
    for kind, df in [("target", Y), ("predictor", X)]:
        for c in df.columns:
            y = df[c].to_numpy(float)
            row = {"map": c, "kind": kind, "bilateral_r": np.corrcoef(y[Lh], y[hom[Lh]])[0, 1]}
            for nt, P in nulls.items():
                b = hl.spin_bilateral_r(y, P, reg)
                row[f"median_{nt}"] = np.median(b)
                row[f"q05_{nt}"] = np.quantile(b, 0.05)
            rows.append(row)
    bil = pd.DataFrame(rows)
    bil.to_csv(hl.RES / "v3_bilateral_null.csv", index=False)
    print(bil.groupby("kind")[[c for c in bil.columns if c.startswith("median") or c == "bilateral_r"]].median().round(3))

    # (3) homologue leakage and distance to the nearest training region of every scheme
    hemi = reg["hemi"].to_numpy()

    def nearest_train_distance(splits):
        d = []
        for tr, te in splits:
            for t in te:
                same = [u for u in tr if hemi[u] == hemi[t]]
                d.append(np.min(geo[t, same]) if same else np.nan)
        return float(np.nanmean(d))

    schemes = {}
    for k in range(100):
        schemes[f"R_seed{k}"] = hl.folds_to_splits(hl.make_folds("R", reg, seed=k))
    for k in range(10):
        schemes[f"H_seed{k}"] = hl.folds_to_splits(hl.make_folds("H", reg, seed=k))
    schemes["R_prereg2026"] = hl.folds_to_splits(hl.make_folds("R", reg))
    schemes["S"] = hl.folds_to_splits(hl.make_folds("S", reg))
    schemes["SH"] = hl.folds_to_splits(hl.make_folds("SH", reg))
    schemes["Sw"] = hl.folds_to_splits(hl.make_folds_within(reg))
    schemes["DD"] = hl.dd_splits(reg)
    schemes["DD-H"] = hl.ddh_splits(reg)
    rows = [{"scheme": k, "homologue_in_training": hl.homologue_in_training(reg, s),
             "nearest_train_geodesic_mm": nearest_train_distance(s), "n_splits": len(s)} for k, s in schemes.items()]
    sch = pd.DataFrame(rows)
    sch.to_csv(hl.RES / "v3_scheme_structure.csv", index=False)
    summ = sch.assign(base=sch["scheme"].str.replace(r"_seed\d+", "", regex=True)).groupby("base").agg(
        hom_mean=("homologue_in_training", "mean"), hom_min=("homologue_in_training", "min"),
        hom_max=("homologue_in_training", "max"), dist_mean=("nearest_train_geodesic_mm", "mean"))
    print(summ.round(3))
    with open(hl.RES / "v3_geometry.json", "w") as f:
        json.dump({"misalignment": mis.to_dict("records"), "scheme_summary": summ.reset_index().to_dict("records")},
                  f, indent=2, default=float)


# ======================================================================================================================
# se: standard errors of the ENIGMA effect sizes
# ======================================================================================================================
def standard_errors():
    X, Y, reg, geo = hl.load_data()
    se, used = hl.target_standard_errors(reg, Y)
    se.to_csv(hl.RES / "v3_target_se.csv")
    with open(hl.RES / "v3_target_se_sources.json", "w") as f:
        json.dump(used, f, indent=1)
    print(json.dumps(used, indent=1))


# ======================================================================================================================
# meff: effective number of independent target maps
# ======================================================================================================================
def meff():
    X, Y, reg, geo = hl.load_data()
    R = np.corrcoef(Y.to_numpy(float).T)
    out = {"li_ji": hl.effective_number_of_tests(R), "nyholt": hl.nyholt_meff(R),
           "max_abs_offdiag_r": float(np.max(np.abs(R - np.eye(len(R))))),
           "pairs_abs_r_gt_0.7": int(((np.abs(R) > 0.7).sum() - len(R)) / 2)}
    pd.DataFrame(R, index=Y.columns, columns=Y.columns).to_csv(hl.RES / "v3_target_correlations.csv")
    with open(hl.RES / "v3_meff.json", "w") as f:
        json.dump(out, f, indent=1)
    print(out)


# ======================================================================================================================
# glasser: geometry of the Glasser multimodal parcellation (fine-parcellation simulations)
# ======================================================================================================================
def glasser():
    """Region table and geodesic distances of the Glasser parcellation (360 regions, 180 homologous pairs), built as for
    the 68 Desikan-Killiany regions in `prepare`: fsaverage5 sphere and surface centroids and geodesic distances between
    the vertices closest to the surface centroids (Dijkstra along mesh edges, within hemisphere); in addition raw
    fsaverage5 sphere centroids (ENIGMA Toolbox procedure) and fs_LR sphere centroids. Labels 1-180 are left and
    181-360 right; label k + 180 is the homologue of label k."""
    raw = hl.RAW_ENIGMA
    nv = 10242
    lab = np.loadtxt(raw / "glasser_360_fsa5.csv").astype(int)
    lab_lr = np.loadtxt(raw / "glasser_360_conte69.csv").astype(int)
    surf = {h: nib.load(raw / f"fsa5_{h.lower()}h.gii") for h in "LR"}
    sphr = {h: nib.load(raw / f"fsa5_sphere_{h.lower()}h.gii").darrays[0].data.astype(float) for h in "LR"}
    sph_lr = {h: nib.load(raw / f"conte69_32k_{h.lower()}h_sphere.gii").darrays[0].data.astype(float) for h in "LR"}
    nv_lr = len(sph_lr["L"])
    rows, medoid = [], []
    for k in range(1, 361):
        h = "L" if k <= 180 else "R"
        v = np.where((lab[:nv] if h == "L" else lab[nv:]) == k)[0]
        v_lr = np.where((lab_lr[:nv_lr] if h == "L" else lab_lr[nv_lr:]) == k)[0]
        assert len(v) > 0 and len(v_lr) > 0, k
        xyz = surf[h].darrays[0].data[v].astype(float)
        c_raw, c_lr, c_srf = sphr[h][v].mean(0), sph_lr[h][v_lr].mean(0), xyz.mean(0)
        c_sph, c_lr = c_raw / np.linalg.norm(c_raw), c_lr / np.linalg.norm(c_lr)
        medoid.append(v[np.argmin(np.linalg.norm(xyz - c_srf, axis=1))])
        rows.append({"name": f"{h}_{(k - 1) % 180 + 1}", "hemi": h, "label": k, "n_vertices": len(v),
                     "sph_x": c_sph[0], "sph_y": c_sph[1], "sph_z": c_sph[2],
                     "sphraw_x": c_raw[0], "sphraw_y": c_raw[1], "sphraw_z": c_raw[2],
                     "fslr_x": c_lr[0], "fslr_y": c_lr[1], "fslr_z": c_lr[2],
                     "x": c_srf[0], "y": c_srf[1], "z": c_srf[2]})
    reg = pd.DataFrame(rows)
    reg["homologue_idx"] = np.r_[np.arange(180, 360), np.arange(180)]
    medoid = np.array(medoid)
    hemi = reg["hemi"].to_numpy()
    geo = np.full((360, 360), np.nan)
    for h in "LR":
        xyz = surf[h].darrays[0].data.astype(float)
        faces = surf[h].darrays[1].data.astype(int)
        e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
        w = np.linalg.norm(xyz[e[:, 0]] - xyz[e[:, 1]], axis=1)
        G = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nv, nv)).tocsr()
        idx = np.where(hemi == h)[0]
        geo[np.ix_(idx, idx)] = dijkstra(G, directed=False, indices=medoid[idx])[:, medoid[idx]]
    assert np.isfinite(geo[:180, :180]).all() and np.isfinite(geo[180:, 180:]).all()
    reg.to_csv(hl.DER / "glasser_regions.csv", index=False)
    np.save(hl.DER / "glasser_geodesic.npy", geo)
    a = hl.homologue_angles(reg, reg[["sph_x", "sph_y", "sph_z"]].to_numpy())
    print(f"Glasser: 360 regions, vertices per region {reg['n_vertices'].min()}-{reg['n_vertices'].max()}; "
          f"homologue misalignment on fsaverage5 {a.mean():.2f} deg (mean)")


STEPS = {"prepare": prepare, "lambda": calibrate_lambda, "geometry": geometry, "se": standard_errors, "meff": meff,
         "glasser": glasser}

if __name__ == "__main__":
    hl.ensure_dirs()
    for step in (sys.argv[1:] or list(STEPS)):
        if step not in STEPS:
            raise SystemExit(__doc__)
        STEPS[step]()
