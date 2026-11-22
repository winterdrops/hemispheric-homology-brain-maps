"""
homology_lib.py - shared machinery for the analyses of
"Ignoring hemispheric homology inflates false positives in spatial null models and cross-validation of brain maps".

Sections
  paths and constants ........ project folders, model settings, the selection of ENIGMA maps
  data ....................... derived matrices, region-wise standard errors of the ENIGMA effect sizes
  cross-validation schemes ... R, H, S, SH (10-fold), Sw (within-hemisphere blocks), DD and DD-H (distance-dependent)
  models ..................... ridge with exact leave-one-(group-)out tuning, kNN, random forests
  metrics .................... R2_cv, skill, Pearson r, per-split statistics, regression metrics, Monte Carlo P, BH-FDR
  null models ................ shuffle; spins: conventional, symmetric, aligned, fs_LR, ENIGMA Toolbox procedure
  correlation tests .......... vectorised two-sided tests of map-to-map correlations
  generative nulls ........... BrainSMASH and Moran spectral randomization (independent, shared seed, whole brain,
                               symmetric = applied to the homologous sum and difference maps)
  simulators ................. Gaussian random fields with exact or approximate hemispheric coupling; positive control

Design notes
* Regions (n = 68, Desikan-Killiany) are the samples; each brain map is one "dataset".
* Ridge and kNN predictions are linear in y for a fixed design and fold structure (ridge up to the per-column choice of
  the penalty), so out-of-fold predictions are computed for many target columns at once. This makes thousands of null
  draws cheap. Random forests are fitted column by column.
* All preprocessing (feature standardisation, penalty selection) happens inside each training set.
* Every random quantity has a fixed seed, so a rerun reproduces the archived results.
"""
import os
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import norm, rankdata
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# ======================================================================================================================
# paths and constants
# ======================================================================================================================
ROOT = Path(os.environ.get("HEMIHOM_ROOT", Path(__file__).resolve().parents[1]))   # project folder (data/, results/)
RAW = ROOT / "data/raw"
RAW_ENIGMA, RAW_HANSEN = RAW / "enigma", RAW / "hansen"
DER = ROOT / "data/derived"
RES = ROOT / "results"
FIG = ROOT / "figures"
LOGS = ROOT / "logs"

ALPHAS = np.logspace(-2, 4, 25)            # ridge penalty grid (main analysis)
ALPHAS_WIDE = np.logspace(-3, 6, 37)       # sensitivity analysis
K_NEIGHBOURS = 5
N_FOLDS = 10
RF_PARAMS = dict(n_estimators=100, max_features=1 / 3, min_samples_leaf=3, bootstrap=True, n_jobs=1)
RF_PARAMS_ALT = dict(n_estimators=100, max_features="sqrt", min_samples_leaf=1, bootstrap=True, n_jobs=1)
TIE_TOL = 1e-12
A_IDX = list(range(1, 4)) + list(range(5, 39)) + list(range(40, 71))  # ENIGMA aparc label ids in DK-68 order
F_REFLECT = np.diag([-1.0, 1.0, 1.0])

# One case-control cortical-thickness map per condition (rule fixed in the analysis plan)
TARGETS_PRIMARY = {
    "22q11.2DS": "22q_case-controls_CortThick",
    "ADHD": "adhdallages_case-controls_CortThick",
    "ASD": "asd_meta-analysis_case-controls_CortThick",
    "Epilepsy": "allepi_case-controls_CortThick",
    "MDD": "mddadult_case-controls_CortThick",
    "OCD": "ocdadults_case-controls_CortThick",
    "SCZ": "scz_case-controls_CortThick",
    "BD": "bd_case-controls_CortThick_adult",
    "PD": "parkinsons_case-controls_CortThick_PDvsCN",
    "AN": "anorexia_case-controls_CortThick",
    "CD": "Antisocial_case-controls_CortThick",
    "CHR": "psychosis_case-controls_CortThick_postComBatmegaanalysis_ICVex",
}
TARGETS_SECONDARY = {  # only for the parity check against Hansen et al. (2022)
    "ASD_mega": "asd_mega-analysis_case-controls_CortThick",
    "GGE": "gge_case-controls_CortThick",
    "TLE_L": "tlemtsl_case-controls_CortThick",
    "TLE_R": "tlemtsr_case-controls_CortThick",
    "Schizotypy": "schizotypy_CortThick_ThicknessCovariate",
}


def ensure_dirs():
    for d in (DER, RES, FIG, LOGS):
        d.mkdir(parents=True, exist_ok=True)


# ======================================================================================================================
# data
# ======================================================================================================================
def load_data():
    """Predictors (68 x 28), targets (68 x 12), region table and geodesic distances (from data/derived)."""
    X = pd.read_csv(DER / "predictors.csv", index_col=0)
    Y = pd.read_csv(DER / "targets_primary.csv", index_col=0)
    reg = pd.read_csv(DER / "regions.csv")
    geo = np.load(DER / "geodesic.npy")
    assert list(X.index) == list(Y.index) == reg["name"].tolist()
    return X, Y, reg, geo


def load_glasser():
    """Region table and geodesic distances of the Glasser parcellation (360 regions; s1_data.py glasser)."""
    return pd.read_csv(DER / "glasser_regions.csv"), np.load(DER / "glasser_geodesic.npy")


def glasser_simulation(n_maps=500):
    """Fine-parcellation simulations: Glasser region table, geodesic distances, 28 simulated annotation maps and n_maps
    null maps, all Gaussian random fields at lambda = 77 mm and rho = 0.9 (independent of each other)."""
    rg, gg = load_glasser()
    return (rg, gg, grf_maps(rg, gg, lam=77, rho=0.9, n_maps=28, seed=31002),
            grf_maps(rg, gg, lam=77, rho=0.9, n_maps=n_maps, seed=31001))


def half_brain(reg, geo, how):
    """One value per homologous pair (34 regions): how = 'avg' averages every map over the two homologues, 'left' keeps
    the left hemisphere. The geometry is that of the left hemisphere (centroids, sphere coordinates, geodesic distances).
    Returns (region table, geodesic distances, transform), where transform maps a 68-row array to 34 rows."""
    L = np.where(reg["hemi"].to_numpy() == "L")[0]
    hom = reg["homologue_idx"].to_numpy()[L]
    reg34 = reg.iloc[L].reset_index(drop=True).copy()
    reg34["homologue_idx"] = np.arange(len(L))
    if how == "avg":
        def transform(A):
            A = np.asarray(A, float)
            return (A[L] + A[hom]) / 2
    elif how == "left":
        def transform(A):
            return np.asarray(A, float)[L]
    else:
        raise ValueError(how)
    return reg34, geo[np.ix_(L, L)], transform


def target_standard_errors(reg, Y):
    """Region-wise standard errors of the ENIGMA Cohen's d values (Supplementary Note 4). The released files report
    uncertainty inconsistently (some 'se' columns are not standard errors of d), so the source is chosen per map:
    (1) confidence-interval width / 3.92 when a CI containing d is reported; (2) the reported SE when its median agrees
    (within 25%) with the SE implied by the reported P values; (3) otherwise the large-sample formula
    se(d) = sqrt((n1 + n2) / (n1 n2) + d^2 / (2 (n1 + n2))) from the reported sample sizes.
    Returns (DataFrame of SEs in the order of reg, dict with the source used for each map)."""
    names = reg["name"].tolist()
    se, used = {}, {}
    for k, stem in TARGETS_PRIMARY.items():
        df = pd.read_csv(RAW_ENIGMA / f"{stem}.csv")
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df["Structure"] = df["Structure"].astype(str).str.strip()
        s = df.set_index("Structure").loc[names]
        num = {c: pd.to_numeric(s[c], errors="coerce").to_numpy() for c in s.columns if c not in ("CorticalThickness",)}
        d = num["d_icv"]
        assert np.allclose(d, Y[k].to_numpy())
        lo, hi, sec = num["low_ci_icv"], num["up_ci_icv"], num["se_icv"]
        pobs = num.get("pobs", np.full(len(d), np.nan))
        ok_p = (pobs > 1e-6) & (pobs < 0.5)
        se_p = np.nanmedian(np.abs(d[ok_p]) / norm.isf(pobs[ok_p] / 2)) if ok_p.any() else np.nan
        if np.isfinite(lo).all() and np.isfinite(hi).all() and np.all((d >= lo - 1e-9) & (d <= hi + 1e-9)):
            se[k], used[k] = (hi - lo) / 3.92, "confidence interval"
        elif np.isfinite(sec).all() and (sec > 0).all() and np.isfinite(se_p) and abs(np.median(sec) / se_p - 1) < 0.25:
            se[k], used[k] = sec, "reported SE (agrees with p-values)"
        else:
            n1, n2 = num["n_controls"], num["n_patients"]
            se[k], used[k] = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2))), "sample-size formula"
        used[k] += f"; median SE {np.median(se[k]):.3f}; p-implied {se_p:.3f}" if np.isfinite(se_p) else \
            f"; median SE {np.median(se[k]):.3f}; no p-values"
    return pd.DataFrame(se, index=names), used


# ======================================================================================================================
# cross-validation schemes
# ======================================================================================================================
def make_folds(scheme, reg, seed=2026, n_folds=N_FOLDS):
    """Integer fold label per region.
    R  : random K-fold over regions
    H  : random K-fold over homologous pairs (left/right homologues always share a fold)
    S  : spatial K-fold; k-means on surface centroids (x, y, z)
    SH : bilateral spatial blocks; k-means on mirrored pair centroids (|x|, y, z); pairs share a fold"""
    n = len(reg)
    rng = np.random.default_rng(seed)
    hom = reg["homologue_idx"].to_numpy()
    left = np.where(reg["hemi"].to_numpy() == "L")[0]
    xyz = reg[["x", "y", "z"]].to_numpy()
    if scheme == "R":
        perm = rng.permutation(n)
        folds = np.empty(n, int)
        folds[perm] = np.arange(n) % n_folds
    elif scheme == "H":
        perm = rng.permutation(len(left))
        folds = np.empty(n, int)
        pair_fold = np.empty(len(left), int)
        pair_fold[perm] = np.arange(len(left)) % n_folds
        folds[left] = pair_fold
        folds[hom[left]] = pair_fold
    elif scheme == "S":
        km = KMeans(n_clusters=n_folds, n_init=100, random_state=seed).fit(xyz)
        folds = km.labels_
    elif scheme == "SH":
        mirrored = np.c_[np.abs(xyz[left, 0]), xyz[left, 1:]]
        mirrored = (mirrored + np.c_[np.abs(xyz[hom[left], 0]), xyz[hom[left], 1:]]) / 2
        km = KMeans(n_clusters=n_folds, n_init=100, random_state=seed).fit(mirrored)
        folds = np.empty(n, int)
        folds[left] = km.labels_
        folds[hom[left]] = km.labels_
    else:
        raise ValueError(scheme)
    return folds


def make_folds_within(reg, seed=2026, k_per_hemi=5):
    """Sw: spatial blocks built separately within each hemisphere (k-means on surface centroids); a region and its
    homologue are therefore always in different folds."""
    folds = np.empty(len(reg), int)
    xyz = reg[["x", "y", "z"]].to_numpy()
    for hi, h in enumerate("LR"):
        idx = np.where(reg["hemi"].to_numpy() == h)[0]
        km = KMeans(n_clusters=k_per_hemi, n_init=100, random_state=seed).fit(xyz[idx])
        folds[idx] = km.labels_ + hi * k_per_hemi
    return folds


def partitions(scheme, reg, n_part):
    """Fold vectors of a partition scheme: n_part random partitions for R and H (seeds 0 .. n_part - 1), one partition
    otherwise."""
    if scheme in ("R", "H"):
        return [make_folds(scheme, reg, seed=k) for k in range(n_part)]
    if scheme == "Sw":
        return [make_folds_within(reg)]
    return [make_folds(scheme, reg)]


def inner_groups(scheme, reg, folds):
    """Group labels for inner (tuning) cross-validation that mirror the outer scheme: leave-one-out for R,
    leave-one-pair-out for H, leave-one-block-out (the remaining outer folds) for S, SH and Sw."""
    n = len(reg)
    if scheme == "R":
        return np.arange(n)
    if scheme == "H":
        return pair_ids(reg)
    return np.asarray(folds)


def pair_ids(reg):
    """Homologous-pair label of every region (the smaller of the two region indices)."""
    return np.minimum(np.arange(len(reg)), reg["homologue_idx"].to_numpy())


def dd_splits(reg, train_pct=0.75):
    """Distance-dependent CV (Hansen et al. 2022): for every source region, the floor(train_pct * n) regions closest
    to it (Euclidean distance between surface centroids, source included) are the training set; the rest the test set."""
    xyz = reg[["x", "y", "z"]].to_numpy()
    D = cdist(xyz, xyz)
    n = len(reg)
    ntr = int(np.floor(train_pct * n))
    out = []
    for i in range(n):
        idx = np.argsort(D[i], kind="stable")
        out.append((np.sort(idx[:ntr]), np.sort(idx[ntr:])))
    return out


def ddh_splits(reg, train_pct=0.75):
    """Homologue-aware distance-dependent CV: the construction of dd_splits applied to the 34 homologous pairs, with
    distances between mirrored pair centroids (|x|, y, z); both members of a pair are always on the same side."""
    hemi = reg["hemi"].to_numpy()
    hom = reg["homologue_idx"].to_numpy()
    L = np.where(hemi == "L")[0]
    xyz = reg[["x", "y", "z"]].to_numpy()
    m = (np.c_[np.abs(xyz[L, 0]), xyz[L, 1:]] + np.c_[np.abs(xyz[hom[L], 0]), xyz[hom[L], 1:]]) / 2
    D = cdist(m, m)
    npair = len(L)
    ntr = int(np.floor(train_pct * npair))
    out = []
    for i in range(npair):
        idx = np.argsort(D[i], kind="stable")
        trp, tep = idx[:ntr], idx[ntr:]
        out.append((np.sort(np.r_[L[trp], hom[L[trp]]]), np.sort(np.r_[L[tep], hom[L[tep]]])))
    return out


def folds_to_splits(folds):
    return [(np.where(folds != f)[0], np.where(folds == f)[0]) for f in np.unique(folds)]


def homologue_in_training(reg, splits):
    """Homologue leakage: fraction of test regions (over all splits) whose homologue is in the training set."""
    hom = reg["homologue_idx"].to_numpy()
    num = den = 0
    for tr, te in splits:
        s = set(tr.tolist())
        num += sum(hom[t] in s for t in te)
        den += len(te)
    return num / den


# ======================================================================================================================
# models
# ======================================================================================================================
def _standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0, ddof=0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _groups_by_size(groups):
    """Group index arrays batched by group size (for vectorised solves)."""
    by_size = {}
    for g in groups:
        by_size.setdefault(len(g), []).append(g)
    return {k: np.array(v) for k, v in by_size.items()}


def _loo_errors(U, d, resid, ntr, by_size):
    """Exact leave-one-out (by_size None) or leave-one-group-out residuals of ridge with an unpenalised intercept:
    e_G = (I - H_GG)^-1 (y_G - yhat_G), with the training-set standardisation held fixed."""
    if by_size is None:
        h = (U**2) @ d + 1.0 / ntr
        return resid / (1.0 - h)[:, None]
    H = (U * d) @ U.T + 1.0 / ntr
    loo = np.empty_like(resid)
    for size, gidx in by_size.items():  # batched solves for groups of equal size
        M = np.eye(size)[None] - H[gidx[:, :, None], gidx[:, None, :]]
        loo[gidx.ravel()] = np.linalg.solve(M, resid[gidx]).reshape(-1, resid.shape[1])
    return loo


def ridge_oof(X, Y, folds, alphas=ALPHAS, return_alpha=False, inner=None):
    """Out-of-fold ridge predictions for every column of Y (n x m). The penalty is chosen per column and per training
    fold by exact leave-one-out error, or - if `inner` (group labels, length n) is given - by exact
    leave-one-group-out error inside the training fold, so that the inner tuning respects the same dependence
    structure as the outer cross-validation."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, m = Y.shape
    P = np.zeros_like(Y)
    chosen = np.zeros((len(np.unique(folds)), m))
    for fi, f in enumerate(np.unique(folds)):
        te, tr = folds == f, folds != f
        Xtr, Xte = _standardise(X[tr], X[te])
        ytr = Y[tr]
        ybar = ytr.mean(0)
        yc = ytr - ybar
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
        UtY = U.T @ yc
        ntr = tr.sum()
        best_err = np.full(m, np.inf)
        best_a = np.zeros(m)
        by_size = None
        if inner is not None:
            g_tr = np.asarray(inner)[tr]
            groups = [np.where(g_tr == g)[0] for g in np.unique(g_tr)]
            if not all(len(g) == 1 for g in groups):
                by_size = _groups_by_size(groups)
        for a in alphas:
            d = s**2 / (s**2 + a)
            resid = yc - U @ (d[:, None] * UtY)
            err = (_loo_errors(U, d, resid, ntr, by_size) ** 2).mean(0)
            better = err < best_err
            best_err[better], best_a[better] = err[better], a
        chosen[fi] = best_a
        for a in np.unique(best_a):  # predictions with column-specific penalty
            cols = best_a == a
            coef = Vt.T @ ((s / (s**2 + a))[:, None] * UtY[:, cols])
            P[np.ix_(te, cols)] = ybar[cols] + Xte @ coef
    return (P, chosen) if return_alpha else P


def knn_oof(X, Y, folds, k=K_NEIGHBOURS):
    """Out-of-fold k-nearest-neighbour regression (uniform weights, standardised features)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    P = np.zeros_like(Y)
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        Xtr, Xte = _standardise(X[tr], X[te])
        D = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        nb = np.argsort(D, axis=1)[:, :k]
        P[te] = Y[tr][nb].mean(1)
    return P


def rf_oof(X, y, folds, seed=0, params=RF_PARAMS):
    """Out-of-fold random-forest predictions for a single target vector."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    p = np.zeros_like(y)
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        p[te] = RandomForestRegressor(random_state=seed, **params).fit(X[tr], y[tr]).predict(X[te])
    return p


def rf_oof_many(X, Y, folds, seed=0):
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    return np.column_stack([rf_oof(X, Y[:, j], folds, seed) for j in range(Y.shape[1])])


def oof(model, X, Y, folds, seed=0, inner=None):
    if model == "ridge":
        return ridge_oof(X, Y, folds, inner=inner)
    if model == "knn":
        return knn_oof(X, Y, folds)
    if model == "rf":
        return rf_oof_many(X, Y, folds, seed)
    raise ValueError(model)


def _ridge_train_predict(Xtr, Ytr, Xte, alphas, groups=None):
    """Ridge with unpenalised intercept; penalty per column by exact LOO (groups None) or exact leave-one-group-out
    error within the training set. Returns (predictions for Xte, chosen penalties)."""
    Xtr, Xte = _standardise(Xtr, Xte)
    ybar = Ytr.mean(0)
    yc = Ytr - ybar
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    UtY = U.T @ yc
    ntr = len(Xtr)
    m = Ytr.shape[1]
    best_err, best_a = np.full(m, np.inf), np.zeros(m)
    by_size = None
    if groups is not None:
        gl = [np.where(groups == g)[0] for g in np.unique(groups)]
        if any(len(g) > 1 for g in gl):
            by_size = _groups_by_size(gl)
    for a in alphas:
        d = s**2 / (s**2 + a)
        resid = yc - U @ (d[:, None] * UtY)
        err = (_loo_errors(U, d, resid, ntr, by_size) ** 2).mean(0)
        better = err < best_err
        best_err[better], best_a[better] = err[better], a
    P = np.empty((len(Xte), m))
    for a in np.unique(best_a):
        cols = best_a == a
        coef = Vt.T @ ((s / (s**2 + a))[:, None] * UtY[:, cols])
        P[:, cols] = ybar[cols] + Xte @ coef
    return P, best_a


def ridge_split_predict(X, Y, splits, alphas=ALPHAS, inner=None):
    """Ridge predictions for arbitrary (possibly overlapping) train/test splits. inner: None (LOO tuning) or group
    labels (length n) for leave-one-group-out tuning inside the training set. Returns a list of (n_test x m) arrays."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float).reshape(len(X), -1)
    out = []
    for tr, te in splits:
        g = None if inner is None else np.asarray(inner)[tr]
        P, _ = _ridge_train_predict(X[tr], Y[tr], X[te], alphas, g)
        out.append(P)
    return out


def knn_split_predict(X, Y, splits, k=K_NEIGHBOURS):
    X = np.asarray(X, float)
    Y = np.asarray(Y, float).reshape(len(X), -1)
    out = []
    for tr, te in splits:
        Xtr, Xte = _standardise(X[tr], X[te])
        D = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        nb = np.argsort(D, axis=1)[:, :k]
        out.append(Y[tr][nb].mean(1))
    return out


# ======================================================================================================================
# metrics
# ======================================================================================================================
def r2_cv(Y, P):
    """Out-of-sample R^2 on pooled out-of-fold predictions (column-wise; SST around the full-sample mean). Can be
    negative."""
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    Y = Y.reshape(Y.shape[0], -1)
    P = P.reshape(P.shape[0], -1)
    assert Y.shape == P.shape
    sse = ((Y - P) ** 2).sum(0)
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1.0 - sse / sst


def skill_cv(Y, P, folds):
    """Same-fold skill score: 1 - SSE(model) / SSE(training-mean predictor under the same folds)."""
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    P = np.asarray(P, float).reshape(len(P), -1)
    B = np.zeros_like(Y)
    for f in np.unique(folds):
        B[folds == f] = Y[folds != f].mean(0)
    return 1.0 - ((Y - P) ** 2).sum(0) / ((Y - B) ** 2).sum(0)


def pearson_cols(Y, P):
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    P = np.asarray(P, float).reshape(len(P), -1)
    Yc, Pc = Y - Y.mean(0), P - P.mean(0)
    den = np.sqrt((Yc**2).sum(0) * (Pc**2).sum(0))
    return np.where(den > 0, (Yc * Pc).sum(0) / np.where(den > 0, den, 1), 0.0)


def split_stats(Y, splits, preds):
    """Per-split statistics averaged over splits: mean test-set Pearson r (the distance-dependent statistic of Hansen
    et al. 2022), mean test-set R^2 around the test mean, and mean skill against the training-mean predictor."""
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    rs, r2s, sks = [], [], []
    for (tr, te), P in zip(splits, preds):
        yt = Y[te]
        yc, pc = yt - yt.mean(0), P - P.mean(0)
        den = np.sqrt((yc**2).sum(0) * (pc**2).sum(0))
        rs.append(np.where(den > 0, (yc * pc).sum(0) / np.where(den > 0, den, 1), 0.0))
        sse = ((yt - P) ** 2).sum(0)
        r2s.append(1 - sse / (yc**2).sum(0))
        sks.append(1 - sse / ((yt - Y[tr].mean(0)) ** 2).sum(0))
    return np.mean(rs, 0), np.mean(r2s, 0), np.mean(sks, 0)


def regression_metrics(y, p):
    """Pooled out-of-fold metrics for one map: R^2 (around the full-sample mean), RMSE, MAE, Pearson r, Spearman rho,
    calibration slope and intercept (least-squares regression of observed on predicted values)."""
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    sse = ((y - p) ** 2).sum()
    out = {"r2": 1 - sse / ((y - y.mean()) ** 2).sum(), "rmse": np.sqrt(sse / len(y)), "mae": np.abs(y - p).mean()}
    out["pearson_r"] = np.corrcoef(y, p)[0, 1] if p.std() > 0 else 0.0
    out["spearman_rho"] = np.corrcoef(rankdata(y), rankdata(p))[0, 1] if p.std() > 0 else 0.0
    vp = p.var()
    out["calib_slope"] = np.cov(y, p, bias=True)[0, 1] / vp if vp > 0 else np.nan
    out["calib_intercept"] = y.mean() - out["calib_slope"] * p.mean() if vp > 0 else np.nan
    return out


def perm_pvalue(obs, null):
    """One-sided (greater) Monte Carlo P value (Phipson & Smyth 2010). Null statistics within TIE_TOL of the observed
    one count as ties: a null draw can equal the observed statistic exactly (e.g. an identity permutation), and in
    floating point the two values then differ in the last bits with a sign that depends on the CPU and BLAS build."""
    null = np.asarray(null)
    return (1.0 + np.sum(null >= obs - TIE_TOL)) / (1.0 + null.size)


def bh_fdr(p):
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def effective_number_of_tests(R):
    """Li & Ji (2005) effective number of independent tests from a correlation matrix."""
    ev = np.abs(np.linalg.eigvalsh(np.asarray(R, float)))
    return float(np.sum((ev >= 1).astype(float) + (ev - np.floor(ev))))


def nyholt_meff(R):
    """Nyholt (2004) effective number of tests."""
    ev = np.linalg.eigvalsh(np.asarray(R, float))
    M = len(ev)
    return float(1 + (M - 1) * (1 - np.var(ev, ddof=1) / M))


def cv_stats_fn(Xmat, reg_, parts, splits):
    """Cross-validated statistics of many target maps at once: returns stats(model, Ymat, s) for the design given by
    the predictors Xmat, the region table reg_, the fold partitions and the distance-dependent splits."""
    pair = pair_ids(reg_)

    def stats(model, Ymat, s):
        """(tested statistic, R2_cv or mean test-set R2, skill, Pearson r) for every column of Ymat under scheme s."""
        if s in parts:
            r2s, sks, rs = [], [], []
            for f in parts[s]:
                if model == "ridge":
                    P = ridge_oof(Xmat, Ymat, f, inner=inner_groups(s, reg_, f))
                else:
                    P = knn_oof(Xmat, Ymat, f)
                r2s.append(r2_cv(Ymat, P))
                sks.append(skill_cv(Ymat, P, f))
                rs.append(pearson_cols(Ymat, P))
            r2, sk, r = np.mean(r2s, 0), np.mean(sks, 0), np.mean(rs, 0)
            return r2, r2, sk, r
        sp = splits[s]
        if model == "ridge":
            preds = ridge_split_predict(Xmat, Ymat, sp, inner=None if s == "DD" else pair)
        else:
            preds = knn_split_predict(Xmat, Ymat, sp)
        r, r2, sk = split_stats(Ymat, sp, preds)
        return r, r2, sk, r
    return stats


def cv_design(Xmat, reg_, schemes, n_part=5):
    """cv_stats_fn for a design: R and H over n_part random partitions, one partition for S, SH and Sw; DD and DD-H
    splits from the region table."""
    parts = {s: partitions(s, reg_, n_part) for s in schemes if s in ("R", "H", "S", "SH", "Sw")}
    splits = {s: dd_splits(reg_) if s == "DD" else ddh_splits(reg_) for s in schemes if s in ("DD", "DD-H")}
    return cv_stats_fn(Xmat, reg_, parts, splits)


# ======================================================================================================================
# null models (permutations of region labels; y[perm] is a null map)
# ======================================================================================================================
def random_rotation(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def shuffle_permutations(n_regions, n_perm, seed=4321):
    rng = np.random.default_rng(seed)
    return np.array([rng.permutation(n_regions) for _ in range(n_perm)])


def spin_permutations(reg, n_spins, seed=1234):
    """Conventional parcel spin (rotations of Alexander-Bloch et al. 2018; Hungarian re-assignment of parcel centroids as
    in Markello & Misic 2021) on the sphere centroids in `reg`: the right hemisphere receives the mirror image of the
    left-hemisphere rotation."""
    rng = np.random.default_rng(seed)
    sph = reg[["sph_x", "sph_y", "sph_z"]].to_numpy()
    hemi = reg["hemi"].to_numpy()
    idx = {h: np.where(hemi == h)[0] for h in "LR"}
    out = np.zeros((n_spins, len(reg)), int)
    for s in range(n_spins):
        Rot = random_rotation(rng)
        rots = {"L": Rot, "R": F_REFLECT @ Rot @ F_REFLECT}
        for h in "LR":
            c = sph[idx[h]]
            c_rot = c @ rots[h].T
            cost = np.linalg.norm(c[:, None, :] - c_rot[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            out[s, idx[h][rows]] = idx[h][cols]
    return out


def spin_permutations_sym(reg, n_spins, seed=1234):
    """Symmetric spin: rotate and re-assign the left hemisphere only, then apply the homologous re-assignment to the
    right hemisphere. Preserves each map's bilateral (homologue) structure exactly."""
    rng = np.random.default_rng(seed)
    sph = reg[["sph_x", "sph_y", "sph_z"]].to_numpy()
    hemi = reg["hemi"].to_numpy()
    hom = reg["homologue_idx"].to_numpy()
    L = np.where(hemi == "L")[0]
    R = np.where(hemi == "R")[0]
    out = np.zeros((n_spins, len(reg)), int)
    c = sph[L]
    for s in range(n_spins):
        Rot = random_rotation(rng)
        c_rot = c @ Rot.T
        cost = np.linalg.norm(c[:, None, :] - c_rot[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        out[s, L[rows]] = L[cols]
        out[s, R] = hom[out[s, hom[R]]]
    return out


def aligned_sphere_coords(reg):
    """Rigidly rotate the right-hemisphere sphere centroids (Kabsch) onto the mirror image of their left homologues,
    removing the misalignment between the fsaverage5 left/right sphere frames."""
    sph = reg[["sph_x", "sph_y", "sph_z"]].to_numpy().copy()
    hemi = reg["hemi"].to_numpy()
    hom = reg["homologue_idx"].to_numpy()
    L = np.where(hemi == "L")[0]
    A = sph[hom[L]]                          # right homologues
    B = sph[L] * np.array([-1.0, 1.0, 1.0])  # mirrored left
    U, _, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R_idx = np.where(hemi == "R")[0]
    sph[R_idx] = sph[R_idx] @ (U @ D @ Vt)
    sph[R_idx] /= np.linalg.norm(sph[R_idx], axis=1, keepdims=True)
    return sph


def _with_sphere(reg, sph):
    reg2 = reg.copy()
    reg2[["sph_x", "sph_y", "sph_z"]] = sph
    return reg2


def spin_permutations_aligned(reg, n_spins, seed=1234):
    """Conventional spin computed on aligned sphere coordinates (see aligned_sphere_coords)."""
    return spin_permutations(_with_sphere(reg, aligned_sphere_coords(reg)), n_spins, seed=seed)


def sphere_centroids(reg, surface="fsa5", normalise=True):
    """Desikan-Killiany parcel centroids on the fsaverage5 ('fsa5') or fs_LR 32k ('conte69') sphere, in the region
    order of `reg`. With normalise=False the raw mean of vertex coordinates is returned (as in the ENIGMA Toolbox)."""
    if surface == "fsa5":
        sph = {h: nib.load(RAW_ENIGMA / f"fsa5_sphere_{h.lower()}h.gii").darrays[0].data.astype(float) for h in "LR"}
        lab = np.loadtxt(RAW_ENIGMA / "aparc_fsa5.csv").astype(int)
    elif surface == "conte69":
        sph = {h: nib.load(RAW_ENIGMA / f"conte69_32k_{h.lower()}h_sphere.gii").darrays[0].data.astype(float)
               for h in "LR"}
        lab = np.loadtxt(RAW_ENIGMA / "aparc_conte69.csv").astype(int)
    else:
        raise ValueError(surface)
    nv = len(sph["L"])
    assert len(lab) == 2 * nv
    out = np.zeros((len(reg), 3))
    for r, h in enumerate(reg["hemi"].to_numpy()):
        lh = lab[:nv] if h == "L" else lab[nv:]
        v = np.where(lh == A_IDX[r])[0]
        assert len(v) > 0
        c = sph[h][v].mean(0)
        out[r] = c / np.linalg.norm(c) if normalise else c
    return out


def homologue_angles(reg, sph):
    """Angle (degrees) between each left centroid and the mirror image of its right homologue."""
    L = np.where(reg["hemi"].to_numpy() == "L")[0]
    hom = reg["homologue_idx"].to_numpy()
    a = sph[L] / np.linalg.norm(sph[L], axis=1, keepdims=True)
    b = (sph[hom[L]] @ F_REFLECT) / np.linalg.norm(sph[hom[L]], axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))


def spin_permutations_fslr(reg, n_spins, seed=1234):
    """Conventional spin with Hungarian re-assignment, computed on fs_LR sphere centroids (columns fslr_* of the region
    table when present, as for the Glasser parcellation; otherwise the Desikan-Killiany centroids)."""
    sph = reg[["fslr_x", "fslr_y", "fslr_z"]].to_numpy() if "fslr_x" in reg else sphere_centroids(reg, "conte69")
    return spin_permutations(_with_sphere(reg, sph), n_spins, seed=seed)


# ENIGMA Toolbox procedure, re-implemented line by line from enigmatoolbox.permutation_testing (commit b08974b)
def _greedy_match(dist):
    """ENIGMA/Vasa greedy assignment: repeatedly take the unrotated region whose nearest rotated region is farthest
    ('max-min') and assign it to that nearest rotated region. Returns (ref, rot) index arrays."""
    d = dist.copy()
    n = d.shape[0]
    ref, rot = np.empty(n, int), np.empty(n, int)
    for k in range(n):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mins = np.nanmin(d, axis=1)
        ref_ix = np.argwhere(mins == np.nanmax(mins))[0][0]
        rot_ix = np.argwhere(d[ref_ix, :] == np.nanmin(d[ref_ix, :]))[0][0]
        ref[k], rot[k] = ref_ix, rot_ix
        d[:, rot_ix] = np.nan
        d[ref_ix, :] = np.nan
    return ref, rot


def enigma_rotate_parcellation(coord_l, coord_r, nrot, normal=None):
    """Port of enigmatoolbox rotate_parcellation. `normal(size)` draws standard normals (default: np.random.normal,
    as in the toolbox). Returns perm_id with shape (n_regions, nrot); x[perm_id[:, r]] is spun."""
    normal = np.random.normal if normal is None else normal
    nl, nr = len(coord_l), len(coord_r)
    perm_id = np.zeros((nl + nr, nrot), int)
    r = 0
    while r < nrot:
        A = normal(size=(3, 3))
        TL, temp = np.linalg.qr(A)
        TL = TL @ np.diag(np.sign(np.diag(temp)))
        if np.linalg.det(TL) < 0:
            TL[:, 0] = -TL[:, 0]
        TR = F_REFLECT @ TL @ F_REFLECT
        ref_l, rot_l = _greedy_match(cdist(coord_l, coord_l @ TL))
        ref_r, rot_r = _greedy_match(cdist(coord_r, coord_r @ TR))
        ref_lr = np.append(ref_l, nl + ref_r)
        rot_lr = np.append(rot_l, nl + rot_r)
        rot_sorted = rot_lr[np.argsort(ref_lr)]
        if not np.all(rot_sorted == np.arange(nl + nr)):
            perm_id[:, r] = rot_sorted
            r += 1
    return perm_id


def spin_permutations_enigma(reg, n_spins, seed=1234):
    """ENIGMA Toolbox parcel spin on fsaverage5 sphere centroids (raw vertex means, as in the toolbox: columns sphraw_*
    of the region table when present, otherwise the Desikan-Killiany centroids); returned as (n_spins, n_regions) in
    the convention of this library."""
    hemi = reg["hemi"].to_numpy()
    nl = int((hemi == "L").sum())
    assert (hemi[:nl] == "L").all() and (hemi[nl:] == "R").all()
    c = (reg[["sphraw_x", "sphraw_y", "sphraw_z"]].to_numpy() if "sphraw_x" in reg
         else sphere_centroids(reg, "fsa5", normalise=False))
    rng = np.random.default_rng(seed)
    perm_id = enigma_rotate_parcellation(c[:nl], c[nl:], n_spins, normal=lambda size: rng.normal(size=size))
    return perm_id.T.copy()


def enigma_pvalue(x, y, perms):
    """ENIGMA Toolbox perm_sphere_p for Pearson r (vectorised over the columns of y): one-tailed in the direction of
    the observed correlation, averaged over spinning x and spinning y, without the +1 correction.
    x: (n,), y: (n, m), perms: (N, n). Returns (m,) P values."""
    x = np.asarray(x, float)
    y = np.asarray(y, float).reshape(len(x), -1)
    r_obs = corr_xy(x[:, None], y)[0]
    null_xy = np.stack([corr_xy(x[p][:, None], y)[0] for p in perms])     # spin x
    null_yx = np.stack([corr_xy(x[:, None], y[p])[0] for p in perms])     # spin y
    pos = r_obs >= 0
    # strict inequalities as in the toolbox; null values within TIE_TOL of the observed one are treated as equal (not
    # counted), so that exact ties (e.g. spins that only swap regions with equal values of an ordinal annotation) do
    # not depend on floating-point noise
    p_xy = np.where(pos, (null_xy > r_obs + TIE_TOL).mean(0), (null_xy < r_obs - TIE_TOL).mean(0))
    p_yx = np.where(pos, (null_yx > r_obs + TIE_TOL).mean(0), (null_yx < r_obs - TIE_TOL).mean(0))
    return (p_xy + p_yx) / 2


NULL_MAKERS = {
    "shuffle": lambda reg, n, seed: shuffle_permutations(len(reg), n, seed=seed),
    "spin_std": spin_permutations,
    "spin_enigma": spin_permutations_enigma,
    "spin_fslr": spin_permutations_fslr,
    "spin_aligned": spin_permutations_aligned,
    "spin_sym": spin_permutations_sym,
}


def null_permutations(reg, n_draws, seeds):
    """Permutation arrays (n_draws x 68) for the null models named in `seeds` ({null name: seed}), in that order.
    Null names: shuffle, spin_std (conventional), spin_enigma, spin_fslr, spin_aligned, spin_sym (symmetric)."""
    return {name: NULL_MAKERS[name](reg, n_draws, seed) for name, seed in seeds.items()}


def spin_bilateral_r(y, perms, reg):
    """Bilateral correlation of every null version y[perm] of map y."""
    L = np.where(reg["hemi"].to_numpy() == "L")[0]
    hom = reg["homologue_idx"].to_numpy()
    Z = y[perms]
    a, b = Z[:, L], Z[:, hom[L]]
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a**2).sum(1) * (b**2).sum(1))


# ======================================================================================================================
# correlation tests
# ======================================================================================================================
def corr_xy(X, Y):
    """Pearson correlations between the columns of X (n x q) and Y (n x m) -> (q x m)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    Xc = (X - X.mean(0)) / X.std(0)
    Yc = (Y - Y.mean(0)) / Y.std(0)
    return Xc.T @ Yc / X.shape[0]


def corr_pvalues(X, Y, perms, chunk=50):
    """Two-sided Monte Carlo P values (Phipson-Smyth) for the correlation of every column of X with every column of Y,
    permuting Y. Returns r_obs (q x m) and p (q x m)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    r_obs = corr_xy(X, Y)
    Xc = (X - X.mean(0)) / X.std(0)
    count = np.zeros_like(r_obs)
    N = len(perms)
    for j0 in range(0, Y.shape[1], chunk):
        Yj = Y[:, j0:j0 + chunk]
        Yp = Yj[perms]                                      # (N, n, mj)
        Yp = (Yp - Yp.mean(1, keepdims=True)) / Yp.std(1, keepdims=True)
        rn = np.einsum("nq,Nnm->Nqm", Xc, Yp) / X.shape[0]  # (N, q, mj)
        count[:, j0:j0 + chunk] = (np.abs(rn) >= np.abs(r_obs[:, j0:j0 + chunk])[None] - 1e-12).sum(0)
    return r_obs, (1.0 + count) / (1.0 + N)


def corr_pvalues_surrogates(X, y, S):
    """Two-sided Monte Carlo P values of corr(X[:, i], y) against surrogate maps S (68 x N)."""
    r = corr_xy(X, y[:, None])[:, 0]
    Xc = (X - X.mean(0)) / X.std(0)
    Sc = (S - S.mean(0)) / S.std(0)
    rn = Xc.T @ Sc / X.shape[0]
    cnt = (np.abs(rn) >= np.abs(r)[:, None] - 1e-12).sum(1)
    return r, (1.0 + cnt) / (1.0 + S.shape[1])


# ======================================================================================================================
# generative null models (settings that neuromaps v0.0.7 uses for surface data; pre-specified in
# plan/prereg_generative_nulls.md)
# ======================================================================================================================
class GenerativeNulls:
    """Surrogate maps from BrainSMASH (variogram matching) and Moran spectral randomization (MSR), applied to
    parcellated data (34 regions per hemisphere, homologous order).
      *_indep   each hemisphere separately with independent random numbers (neuromaps with seed=None, the default)
      *_shared  each hemisphere separately with the same seed (neuromaps with a fixed integer seed)
      bs_whole  BrainSMASH on the whole brain with Euclidean distances between parcel centroids
      *_sym     remedy: the generator applied to the homologous sum and difference maps s = (x_L + x_R) / 2 and
                a = (x_L - x_R) / 2 on the mean of the two hemispheric distance matrices; x_L* = s* + a*, x_R* = s* - a*
                preserves the bilateral covariance"""
    VARIANTS = ["bs_indep", "bs_shared", "bs_whole", "bs_sym", "msr_indep", "msr_shared", "msr_sym"]

    def __init__(self, reg, geo):
        hom = reg["homologue_idx"].to_numpy()
        self.L = np.where(reg["hemi"].to_numpy() == "L")[0]
        self.R = hom[self.L]                                   # right hemisphere in homologous order
        self.DL, self.DR = geo[np.ix_(self.L, self.L)], geo[np.ix_(self.R, self.R)]
        self.DSYM = (self.DL + self.DR) / 2
        xyz = reg[["x", "y", "z"]].to_numpy()
        self.DW = cdist(xyz, xyz)
        self.WL, self.WR, self.WSYM = (self._weights(D) for D in (self.DL, self.DR, self.DSYM))

    @staticmethod
    def _weights(D):
        W = D.astype(float).copy()
        np.fill_diagonal(W, 1)
        return W ** -1

    @staticmethod
    def _bs(x, D, n, seed):
        from brainsmash.mapgen.base import Base
        return np.asarray(Base(x, D, seed=int(seed))(n, 50)).reshape(n, -1)

    @staticmethod
    def _msr(x, W, n, seed):
        from brainspace.null_models.moran import MoranRandomization
        m = MoranRandomization(joint=True, tol=1e-6, n_rep=n, random_state=int(seed)).fit(W)
        return np.asarray(m.randomize(x)).reshape(n, -1)

    def surrogates(self, y, variant, n, seed):
        """(68, n) surrogate maps of y in the original region order."""
        L, R = self.L, self.R
        out = np.empty((len(y), n))
        yl, yr = y[L], y[R]
        gen, D, Dh, Ws = ((self._bs, self.DL, self.DR, self.DSYM) if variant.startswith("bs")
                          else (self._msr, self.WL, self.WR, self.WSYM))
        kind = variant.split("_")[1]
        if kind == "whole":
            out[:] = self._bs(y, self.DW, n, seed).T
        elif kind in ("indep", "shared"):
            out[L] = gen(yl, D, n, seed).T
            out[R] = gen(yr, Dh, n, seed if kind == "shared" else seed + 1).T
        elif kind == "sym":
            s, a = (yl + yr) / 2, (yl - yr) / 2
            S = gen(s, Ws, n, seed + 2).T
            A = gen(a, Ws, n, seed + 3).T if np.std(a) > 1e-10 * max(np.std(s), 1e-12) else np.zeros_like(S)
            out[L], out[R] = S + A, S - A
        else:
            raise ValueError(variant)
        return out

    def bilateral_r(self, S):
        """Bilateral correlation of every surrogate map (columns of S)."""
        a, b = S[self.L] - S[self.L].mean(0), S[self.R] - S[self.R].mean(0)
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


# ======================================================================================================================
# simulators
# ======================================================================================================================
def grf_maps(reg, geo, lam, rho, n_maps, seed=0, nugget=1e-6):
    """Gaussian random fields at parcel level, independent of the predictors by construction. Within hemisphere:
    covariance exp(-d_geo / lam). Between hemispheres: right-hemisphere values are rho * (left homologue field) +
    sqrt(1 - rho^2) * (independent right field). Columns are z-scored maps."""
    rng = np.random.default_rng(seed)
    hemi = reg["hemi"].to_numpy()
    hom = reg["homologue_idx"].to_numpy()
    L, R = np.where(hemi == "L")[0], np.where(hemi == "R")[0]
    chol = {}
    for h, idx in (("L", L), ("R", R)):
        C = np.exp(-geo[np.ix_(idx, idx)] / lam) + nugget * np.eye(len(idx))
        chol[h] = np.linalg.cholesky(C)
    out = np.zeros((len(reg), n_maps))
    zl = chol["L"] @ rng.normal(size=(len(L), n_maps))
    zr = chol["R"] @ rng.normal(size=(len(R), n_maps))
    out[L] = zl
    right_of_left = hom[L]          # right-hemisphere region hom[L[i]] receives rho * left value i + noise
    pos_in_R = {r: k for k, r in enumerate(R)}
    zr_aligned = zr[[pos_in_R[r] for r in right_of_left]]
    out[right_of_left] = rho * zl + np.sqrt(max(0.0, 1 - rho**2)) * zr_aligned
    out = (out - out.mean(0)) / out.std(0)
    return out


def approx_maps(reg, geo, n_maps, seed, lam=77.0, rho=0.9, tau=20.0):
    """Generator with approximate hemispheric correspondence: the right-hemisphere field is coupled to a spatially
    smoothed, geometry-based image of the left field, z_R = rho * norm(W z_L) + sqrt(1 - rho^2) * e_R, with
    W(r, l) proportional to exp(-d(mirror(r), l)^2 / (2 tau^2)) (Euclidean distance between the mirrored right
    centroid and the left centroids)."""
    rng = np.random.default_rng(seed)
    hemi = reg["hemi"].to_numpy()
    L, R = np.where(hemi == "L")[0], np.where(hemi == "R")[0]
    xyz = reg[["x", "y", "z"]].to_numpy()
    chol = {h: np.linalg.cholesky(np.exp(-geo[np.ix_(idx, idx)] / lam) + 1e-6 * np.eye(len(idx)))
            for h, idx in (("L", L), ("R", R))}
    W = np.exp(-cdist(xyz[R] * [-1, 1, 1], xyz[L]) ** 2 / (2 * tau**2))
    W /= W.sum(1, keepdims=True)
    out = np.zeros((len(reg), n_maps))
    zl = chol["L"] @ rng.normal(size=(len(L), n_maps))
    zr = chol["R"] @ rng.normal(size=(len(R), n_maps))
    img = W @ zl
    img = (img - img.mean(0)) / img.std(0)
    out[L] = zl
    out[R] = rho * img + np.sqrt(1 - rho**2) * (zr - zr.mean(0)) / zr.std(0)
    return (out - out.mean(0)) / out.std(0)


def signal_maps(Xs, E, s_level, rng):
    """Positive-control maps y = sqrt(s) z(X_k beta) + sqrt(1 - s) e for the noise fields in the columns of E, with X_k
    three randomly drawn standardised predictors and beta ~ N(0, I). Random numbers are drawn map by map (the three
    predictors, then beta), so every map has the same signal whatever subset is analysed.
    Returns (maps (68 x m), signals (68 x m), predictor indices (m x 3))."""
    ys, sigs, cols_all = [], [], []
    for j in range(E.shape[1]):
        cols = rng.choice(Xs.shape[1], size=3, replace=False)
        sig = Xs[:, cols] @ rng.normal(size=3)
        sig = (sig - sig.mean()) / sig.std()
        y = np.sqrt(s_level) * sig + np.sqrt(1 - s_level) * E[:, j]
        ys.append((y - y.mean()) / y.std())
        sigs.append(sig)
        cols_all.append(cols)
    return np.column_stack(ys), np.column_stack(sigs), np.array(cols_all)
