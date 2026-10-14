"""
s7_figures.py - step 7: all figures and the Source Data workbook (reads result files; no model fitting).

  fig1 .. fig5     main figures                          -> figures/Fig1..Fig5 (.pdf, .png, .tif)
  supp             Supplementary Figs 1-7, 12 and 13     -> figures/supp/SuppFig*.{pdf,png,tif}
  prespecified     Supplementary Figs 8-11 (pre-specified analyses) -> figures/supp/SuppFig8..11_*.png
  source_data      Source Data workbook (one sheet per main-figure panel) -> figures/Source_Data.xlsx
  supp_data        Supplementary Data 1: Supplementary Tables 1-10, one sheet per table -> figures/Supplementary_Data_1.xlsx

Usage: python s7_figures.py [fig1 fig2 fig3 fig4 fig5 supp prespecified source_data supp_data]   (default: all)
Style: Nature Communications specifications (Arial-metric sans serif, 5-8 pt text, 180 mm double-column width; vector
PDF, 600 dpi RGB TIFF and 300 dpi PNG). Text is set in Liberation Sans (metrically identical to Arial), bundled in
fonts/ (SIL Open Font License 1.1) so that the figures look the same on every system.
"""
import glob
import io
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

import homology_lib as hl  # noqa: E402

warnings.filterwarnings("ignore")
RES, RAW = hl.RES, hl.RAW_ENIGMA
OUT, OUT_SUPP = hl.FIG, hl.FIG / "supp"
for _ttf in sorted((Path(__file__).resolve().parent / "fonts").glob("*.ttf")):   # bundled Liberation Sans
    font_manager.fontManager.addfont(str(_ttf))


def _font():
    for name in ["Liberation Sans", "Arial"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            pass
    return "DejaVu Sans"


FONT = _font()


class fs:
    """Figure style of the main and Supplementary Figs 1-7 (validated categorical palette)."""
    MM = 1 / 25.4
    W2, W1 = 180 * MM, 88 * MM
    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
    GRID, AXIS = "#e1e0d9", "#c3c2b7"
    GRAY_LIGHT = "#dcdbd5"
    NULL_ORDER = ["shuffle", "spin_enigma", "spin_std", "spin_fslr", "spin_aligned", "spin_sym"]
    NULL_COL = {"shuffle": "#2a78d6", "spin_enigma": "#e34948", "spin_std": "#eda100", "spin_fslr": "#1baf7a",
                "spin_aligned": "#4a3aa7", "spin_sym": "#008300"}
    NULL_SHORT = {"shuffle": "Shuffle", "spin_enigma": "ENIGMA TB", "spin_std": "Conventional", "spin_fslr": "fs_LR",
                  "spin_aligned": "Aligned", "spin_sym": "Symmetric"}
    SCHEME_COL = {"R": "#2a78d6", "Sw": "#eb6834", "DD": "#1baf7a", "H": "#eda100", "SH": "#e87ba4", "DD-H": "#008300",
                  "S": "#4a3aa7"}
    SCHEME_LS = {"R": "-", "Sw": "-", "DD": "-", "S": "-", "H": (0, (4, 1.5)), "SH": (0, (4, 1.5)),
                 "DD-H": (0, (4, 1.5))}
    SCHEME_LAB = {"R": "R, random", "H": "H, homologue pairs", "S": "S, spatial blocks", "SH": "SH, bilateral blocks",
                  "Sw": "Sw, within-hemisphere blocks", "DD": "DD, distance-dependent", "DD-H": "DD-H, homologue-aware DD"}
    MODEL_LAB = {"ridge": "Ridge regression", "knn": "k-nearest neighbours", "rf": "Random forest"}
    SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

    @staticmethod
    def apply():
        mpl.rcdefaults()
        mpl.rcParams.update({
            "font.family": FONT, "font.size": 6.5, "axes.titlesize": 6.8, "axes.labelsize": 6.5,
            "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
            "axes.edgecolor": fs.AXIS, "axes.labelcolor": fs.INK2, "xtick.color": fs.INK2, "ytick.color": fs.INK2,
            "text.color": fs.INK, "axes.titlecolor": fs.INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
            "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "axes.grid.axis": "y", "grid.color": fs.GRID, "grid.linewidth": 0.5,
            "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02, "savefig.facecolor": "white", "figure.facecolor": "white",
            "axes.facecolor": "white", "lines.linewidth": 1.2, "patch.linewidth": 0, "pdf.fonttype": 42,
            "ps.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "custom", "mathtext.rm": FONT,
            "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
        })

    @staticmethod
    def panel(ax, letter, x=-0.16, y=1.04):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left",
                color=fs.INK)

    @staticmethod
    def save(fig, name, outdir):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{name}.pdf")
        fig.savefig(outdir / f"{name}.png", dpi=300)
        fig.savefig(outdir / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
        from PIL import Image   # journals ask for RGB (not RGBA) TIFF files: flatten the alpha channel onto white
        im = Image.open(outdir / f"{name}.tif")
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            bg.save(outdir / f"{name}.tif", compression="tiff_lzw", dpi=im.info.get("dpi", (600, 600)))

    @staticmethod
    def sel(df, **kw):
        """Row selection by equality."""
        m = None
        for k, v in kw.items():
            c = df[k] == v
            m = c if m is None else (m & c)
        return df[m] if m is not None else df


def cv_null_results():
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / "v3_cv_sim_*.csv")))]).drop_duplicates(
        ["cond", "map", "model", "scheme"])


def scheme_leakage():
    st = pd.read_csv(RES / "v3_scheme_structure.csv")
    st["base"] = st["scheme"].str.replace(r"_seed\d+", "", regex=True)
    return st


def sphere_angles(surf, parc, align=False):
    """Angle between each left-hemisphere parcel centroid and the mirrored right homologue (degrees)."""
    F = np.array([-1.0, 1.0, 1.0])
    if surf == "fsa5":
        S = {h: nib.load(RAW / f"fsa5_sphere_{h}h.gii").darrays[0].data.astype(float) for h in "lr"}
    else:
        S = {h: nib.load(RAW / f"conte69_32k_{h}h_sphere.gii").darrays[0].data.astype(float) for h in "lr"}
    nv = len(S["l"])
    lab = np.loadtxt(RAW / f"{parc}_{surf}.csv").astype(int)
    Ll, Rl = lab[:nv], lab[nv:]
    lk, off = (list(range(1, 4)) + list(range(5, 36)), 35) if parc == "aparc" else (list(range(1, 181)), 180)
    cl = np.array([S["l"][Ll == k].mean(0) for k in lk])
    cr = np.array([S["r"][Rl == k + off].mean(0) for k in lk])
    cl /= np.linalg.norm(cl, axis=1, keepdims=True)
    cr /= np.linalg.norm(cr, axis=1, keepdims=True)
    if align:
        U, _, Vt = np.linalg.svd(cr.T @ (cl * F))
        cr = cr @ (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt)
    return lk, np.degrees(np.arccos(np.clip((cl * (cr * F)).sum(1), -1, 1)))


def example_test_sets(reg):
    """One test set of each scheme, for the source region left superior frontal gyrus (Fig. 1d)."""
    L = np.where(reg["hemi"] == "L")[0]
    src = reg["name"].tolist().index("L_superiorfrontal")
    tests = {}
    for s in ["R", "Sw", "DD", "H", "SH", "DD-H"]:
        if s in ("R", "H", "SH", "Sw"):
            f = (hl.make_folds(s, reg, seed=0) if s in ("R", "H") else hl.make_folds_within(reg) if s == "Sw"
                 else hl.make_folds("SH", reg))
            tests[s] = np.where(f == f[src])[0]
        elif s == "DD":
            tests[s] = hl.dd_splits(reg)[src][1]
        else:
            tests[s] = hl.ddh_splits(reg)[list(L).index(src)][1]
    return tests


# ======================================================================================================================
# Fig. 1: hemispheric homology in brain maps, in spin tests and in cross-validation schemes
# ======================================================================================================================
def fig1():
    from nilearn import datasets, plotting
    from PIL import Image
    fs.apply()
    X, Y, reg, geo = hl.load_data()
    hom = reg["homologue_idx"].to_numpy()
    L = np.where(reg["hemi"] == "L")[0]
    fig = plt.figure(figsize=(fs.W2, 128 * fs.MM))
    gtop = fig.add_gridspec(1, 3, left=0.07, right=0.99, top=0.95, bottom=0.57, wspace=0.5, width_ratios=[1, 1.05, 1.35])

    # a: bilateral symmetry of targets and predictors
    ax = fig.add_subplot(gtop[0, 0])
    groups = [("Disorder\nmaps (12)", Y.columns, Y), ("PET (19)", [c for c in X.columns if c.startswith("PET")], X),
              ("MEG (6)", [c for c in X.columns if c.startswith("MEG")], X),
              ("Other (3)", ["SC_strength", "FC_strength", "Mesulam_class"], X)]
    for gi, (lab, cols, df) in enumerate(groups):
        v = np.array([np.corrcoef(df[c].to_numpy()[L], df[c].to_numpy()[hom[L]])[0, 1] for c in cols])
        jit = np.random.default_rng(gi).uniform(-0.16, 0.16, len(v))
        ax.scatter(np.full(len(v), gi) + jit, v, s=9, color=fs.INK2 if gi == 0 else fs.MUTED, edgecolor="white",
                   linewidth=0.4, zorder=3)
        ax.hlines(np.median(v), gi - 0.3, gi + 0.3, color=fs.INK, lw=1.2, zorder=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([g[0] for g in groups], fontsize=5.6)
    ax.set_ylim(0.65, 1.01)
    ax.set_ylabel("Bilateral correlation (left vs right homologues)")
    ax.set_title("Bilateral symmetry of maps", fontsize=6.5, loc="left")
    fs.panel(ax, "a", x=-0.34)

    # b: sphere-frame misalignment
    ax = fig.add_subplot(gtop[0, 1])
    conds = [("fsavg\nDK", "fsa5", "aparc", False), ("fsavg\nGlasser", "fsa5", "glasser_360", False),
             ("fs_LR\nDK", "conte69", "aparc", False), ("fs_LR\nGlasser", "conte69", "glasser_360", False),
             ("fsavg DK\naligned", "fsa5", "aparc", True)]
    for i, (lab, s, p, al) in enumerate(conds):
        a = sphere_angles(s, p, al)[1]
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(a))
        col = fs.NULL_COL["spin_std"] if s == "fsa5" and not al else (fs.NULL_COL["spin_fslr"] if s == "conte69"
                                                                      else fs.NULL_COL["spin_aligned"])
        ax.scatter(np.full(len(a), i) + jit, a, s=3 if len(a) > 50 else 7, color=col, alpha=0.75, linewidth=0, zorder=3)
        ax.hlines(a.mean(), i - 0.32, i + 0.32, color=fs.INK, lw=1.2, zorder=4)
        ax.text(i, a.max() + 1.0, f"{a.mean():.1f}°", ha="center", fontsize=5.6, color=fs.INK)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[0] for c in conds], fontsize=5.6)
    ax.set_ylim(-0.8, 23)
    ax.set_ylabel("Left vs mirrored right centroid (°)")
    ax.set_title("Hemispheric sphere-frame offset", fontsize=6.5, loc="left")
    fs.panel(ax, "b", x=-0.3)

    # c: bilateral r kept by each null
    ax = fig.add_subplot(gtop[0, 2])
    b = pd.read_csv(RES / "v3_bilateral_null.csv").query("kind == 'target'").set_index("map")
    xs = np.arange(len(fs.NULL_ORDER) + 1)
    for m, row in b.iterrows():
        ax.plot(xs, [row["bilateral_r"]] + [row[f"median_{n}"] for n in fs.NULL_ORDER], color=fs.GRAY_LIGHT, lw=0.6,
                zorder=1)
    ax.scatter(np.zeros(len(b)), b["bilateral_r"], s=8, color=fs.INK, zorder=3, linewidth=0)
    for i, n in enumerate(fs.NULL_ORDER, start=1):
        ax.scatter(np.full(len(b), i), b[f"median_{n}"], s=8, color=fs.NULL_COL[n], zorder=3, linewidth=0)
        ax.text(i, 1.04, f"{b[f'median_{n}'].median():.2f}", ha="center", fontsize=5.4, color=fs.INK2)
    ax.text(0, 1.04, f"{b['bilateral_r'].median():.2f}", ha="center", fontsize=5.4, color=fs.INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Observed"] + [fs.NULL_SHORT[n] for n in fs.NULL_ORDER], rotation=35, ha="right", fontsize=5.6)
    ax.set_ylim(-0.1, 1.09)
    ax.set_ylabel("Bilateral correlation of null maps\n(median over 1,000 draws)")
    ax.set_title("Bilateral structure of null maps", fontsize=6.5, loc="left")
    fs.panel(ax, "c", x=-0.24)

    # d: test sets of each scheme on the inflated cortex
    aparc = np.loadtxt(RAW / "aparc_fsa5.csv").astype(int)
    lab2reg = {lab: r for r, lab in enumerate(hl.A_IDX)}
    fsavg = datasets.fetch_surf_fsaverage("fsaverage5")
    NV = 10242
    cmap = ListedColormap([fs.GRAY_LIGHT, "#e87ba4", "#8a6fdf"])

    def to_vertex(values, hemi):
        lab = aparc[:NV] if hemi == "L" else aparc[NV:]
        return np.array([values[lab2reg[v]] if v in lab2reg else 0 for v in lab], float)

    def render(lab, hemi):
        f2 = plt.figure(figsize=(3, 2.2))
        a2 = f2.add_subplot(111, projection="3d")
        plotting.plot_surf_roi(fsavg.infl_left if hemi == "L" else fsavg.infl_right, roi_map=to_vertex(lab, hemi),
                               hemi="left" if hemi == "L" else "right", view="lateral",
                               bg_map=fsavg.sulc_left if hemi == "L" else fsavg.sulc_right, cmap=cmap, vmin=1, vmax=3,
                               axes=a2, figure=f2, colorbar=False, bg_on_data=True, engine="matplotlib")
        a2.set_box_aspect(None, zoom=1.5)
        buf = io.BytesIO()
        f2.savefig(buf, dpi=400, format="png", facecolor="white")
        plt.close(f2)
        im = np.array(Image.open(buf).convert("RGB"))
        rows, cols = np.where((im < 245).any(2))
        return im[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    test_sets = example_test_sets(reg)
    shown = list(test_sets)
    gb = fig.add_gridspec(2, len(shown), left=0.03, right=0.76, top=0.42, bottom=0.02, hspace=0.02, wspace=0.04)
    for ci, s in enumerate(shown):
        lab = np.ones(68)
        lab[test_sets[s]] = 3.0                      # test region
        for t in test_sets[s]:
            if lab[hom[t]] != 3.0:
                lab[hom[t]] = 2.0                    # homologue of a test region sitting in the training set
        for ri, hemi in enumerate("LR"):
            ax = fig.add_subplot(gb[ri, ci])
            ax.imshow(render(lab, hemi))
            ax.set_axis_off()
            if ri == 0:
                ax.set_title(fs.SCHEME_LAB[s].replace(", ", "\n"), fontsize=5.8, color=fs.INK, pad=2)
            if ci == 0:
                ax.text(-0.04, 0.5, "Left" if hemi == "L" else "Right", transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=5.8, color=fs.INK2)
    fig.text(0.01, 0.485, "d", fontsize=8, fontweight="bold")
    fig.text(0.035, 0.485, "One test set per scheme (source region: left superior frontal gyrus). Purple, test regions; "
             "pink, homologues of test regions that are in the training set", fontsize=5.8, color=fs.INK2)

    # e: homologue leakage per scheme
    ax = fig.add_axes([0.84, 0.06, 0.15, 0.36])
    st = scheme_leakage()
    order = ["Sw", "R", "S", "DD", "H", "SH", "DD-H"]
    vals = [fs.sel(st, base=s_)["homologue_in_training"].mean() for s_ in order]
    ax.barh(range(len(order)), vals, color=[fs.SCHEME_COL[s] for s in order], height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{100 * v:.0f}%", va="center", fontsize=5.6, color=fs.INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50", "100"])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Homologue in the\ntraining set (% of\ntest regions)")
    fs.panel(ax, "e", x=-0.42, y=1.04)
    fs.save(fig, "Fig1", OUT)


# ======================================================================================================================
# Fig. 2: false-positive rates of spatial nulls for map-to-map correlations, and consequences in ENIGMA data
# ======================================================================================================================
GEN = [("bs_indep", "BrainSMASH ind.", "#eb6834", False), ("bs_sym", "BrainSMASH sym.", "#eb6834", True),
       ("msr_indep", "Moran ind.", "#e87ba4", False), ("msr_sym", "Moran sym.", "#e87ba4", True)]


def fig2():
    fs.apply()
    t = pd.read_csv(RES / "v3_stats_bivariate_fpr.csv")
    two = t[t["pvalue"] == "two_sided"]
    fig = plt.figure(figsize=(fs.W2, 66 * fs.MM))
    gs = fig.add_gridspec(1, 4, left=0.06, right=0.99, top=0.86, bottom=0.30, wspace=0.42,
                          width_ratios=[2.0, 0.85, 0.85, 2.0])
    order = ["shuffle", "TB", "spin_enigma", "spin_std", "spin_fslr", "spin_aligned", "spin_sym"]
    labels = ["Shuffle", "ENIGMA TB†", "ENIGMA TB", "Conventional", "fs_LR", "Aligned", "Symmetric"]

    # a: calibrated condition
    ax = fig.add_subplot(gs[0, 0])
    c = two[two["cond"] == "lam77_rho0.9"].set_index("null")
    tb = t[(t["cond"] == "lam77_rho0.9") & (t["pvalue"] == "enigma_toolbox")].iloc[0]
    for i, n in enumerate(order):
        r = tb if n == "TB" else c.loc[n]
        col = fs.NULL_COL["spin_enigma"] if n == "TB" else fs.NULL_COL[n]
        if n == "TB":
            ax.bar(i, r["fpr"], width=0.72, facecolor="white", edgecolor=col, hatch="//////", linewidth=0.8)
        else:
            ax.bar(i, r["fpr"], width=0.72, color=col)
        ax.errorbar(i, r["fpr"], yerr=[[r["fpr"] - r["lo"]], [r["hi"] - r["fpr"]]], fmt="none", ecolor=fs.INK2,
                    elinewidth=0.6, capsize=1.2)
        ax.text(i, r["hi"] + 0.008, f"{100 * r['fpr']:.1f}", ha="center", fontsize=5.4, color=fs.INK)
    gfile = RES / "v3_stats_gen_biv_fpr.csv"
    xg = []
    if gfile.exists():
        gf = pd.read_csv(gfile).set_index(["cond", "null"])
        for k, (n, lab_, col, hatched) in enumerate(GEN):
            x = len(order) + 0.6 + k
            xg.append(x)
            r = gf.loc[("lam77_rho0.9", n)]
            if hatched:
                ax.bar(x, r["fpr"], width=0.72, facecolor="white", edgecolor=col, hatch="\\\\\\", linewidth=0.8)
            else:
                ax.bar(x, r["fpr"], width=0.72, color=col)
            ax.errorbar(x, r["fpr"], yerr=[[r["fpr"] - r["lo"]], [r["hi"] - r["fpr"]]], fmt="none", ecolor=fs.INK2,
                        elinewidth=0.6, capsize=1.2)
            ax.text(x, r["hi"] + 0.008, f"{100 * r['fpr']:.1f}", ha="center", fontsize=5.4, color=fs.INK)
        ax.axvline(len(order) - 0.2, color=fs.GRID, lw=0.8)
        ax.text(len(order) + 2.1, 0.305, "Generative nulls", ha="center", fontsize=5.4, color=fs.INK2)
        ax.text(3.0, 0.305, "Spins and shuffle", ha="center", fontsize=5.4, color=fs.INK2)
    ax.axhline(0.05, color=fs.INK, lw=0.7, ls=(0, (2, 2)))
    ax.set_xticks(list(range(len(order))) + xg)
    ax.set_xticklabels(labels + [g[1] for g in GEN][:len(xg)], rotation=40, ha="right", fontsize=5.6)
    ax.set_ylim(0, 0.32)
    ax.set_ylabel("False-positive rate (%; α = 0.05)")
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    ax.set_yticklabels(["0", "5", "10", "15", "20", "25", "30"])
    ax.set_title("λ = 77 mm, ρ = 0.9", fontsize=6.2, loc="left")
    ax.text(0.30, 0.86, "† as implemented\n(directional P)", transform=ax.transAxes, ha="left", va="top", fontsize=5.2,
            color=fs.NULL_COL["spin_enigma"])
    fs.panel(ax, "a", x=-0.22)

    # b, c: sweeps over bilateral coupling and correlation length
    def sweep(ax, conds, xlabels, xlabel, title, letter):
        xvals = np.arange(len(conds))
        g = two.set_index(["cond", "null"])
        for n in fs.NULL_ORDER:
            ax.fill_between(xvals, [g.loc[(cc, n), "lo"] for cc in conds], [g.loc[(cc, n), "hi"] for cc in conds],
                            color=fs.NULL_COL[n], alpha=0.15, linewidth=0)
            ax.plot(xvals, [g.loc[(cc, n), "fpr"] for cc in conds], color=fs.NULL_COL[n], marker="o", ms=2.5, lw=1.1,
                    label=fs.NULL_SHORT[n])
        ax.axhline(0.05, color=fs.INK, lw=0.7, ls=(0, (2, 2)))
        ax.set_xticks(xvals)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel(xlabel)
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([0.05, 0.1, 0.2, 0.3])
        ax.set_yticklabels(["5", "10", "20", "30"])
        ax.set_ylim(0.04, 0.36)
        ax.set_title(title, fontsize=6.2, loc="left")
        fs.panel(ax, letter, x=-0.28)

    ax = fig.add_subplot(gs[0, 1])
    sweep(ax, ["lam77_rho0.0", "lam77_rho0.5", "lam77_rho0.9", "lam77_rho1.0"], ["0", "0.5", "0.9", "1"],
          "Bilateral coupling ρ", "λ = 77 mm", "b")
    ax.set_ylabel("False-positive rate (%, log scale)")
    ax = fig.add_subplot(gs[0, 2])
    sweep(ax, ["lam25_rho0.9", "lam50_rho0.9", "lam77_rho0.9", "lam100_rho0.9"], ["25", "50", "77", "100"],
          "Correlation length λ (mm)", "ρ = 0.9", "c")
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, bbox_to_anchor=(0.56, 1.02), fontsize=5.8, handlelength=1.6,
               columnspacing=1.0)

    # d: ENIGMA data
    ax = fig.add_subplot(gs[0, 3])
    r = pd.read_csv(RES / "v3_bivariate_real_counts.csv")
    rr = r[r["pvalue"] == "two_sided"].set_index("null")
    tbr = r[r["pvalue"] == "enigma_toolbox"].iloc[0]
    for i, n in enumerate(order):
        rw = tbr if n == "TB" else rr.loc[n]
        col = fs.NULL_COL["spin_enigma"] if n == "TB" else fs.NULL_COL[n]
        if n == "TB":
            ax.bar(i, rw["n_p05"], width=0.72, facecolor="white", edgecolor=col, hatch="//////", linewidth=0.8, alpha=0.6)
            ax.bar(i, rw["n_q05"], width=0.72, facecolor=col, edgecolor=col, hatch="//////", linewidth=0.8)
        else:
            ax.bar(i, rw["n_p05"], width=0.72, color=col, alpha=0.35)
            ax.bar(i, rw["n_q05"], width=0.72, color=col)
        ax.text(i, rw["n_p05"] + 2, f"{int(rw['n_q05'])}", ha="center", fontsize=5.6, color=fs.INK, fontweight="bold")
    xg = []
    if (RES / "v3_gen_real_counts.csv").exists():
        gq = pd.read_csv(RES / "v3_gen_real_counts.csv").set_index("null")
        for k, (n, lab_, col, hatched) in enumerate(GEN):
            x = len(order) + 0.6 + k
            xg.append(x)
            rw = gq.loc[n]
            if hatched:
                ax.bar(x, rw["n_p05"], width=0.72, facecolor="white", edgecolor=col, hatch="\\\\\\", linewidth=0.8,
                       alpha=0.6)
                ax.bar(x, rw["n_q05"], width=0.72, facecolor=col, edgecolor=col, hatch="\\\\\\", linewidth=0.8)
            else:
                ax.bar(x, rw["n_p05"], width=0.72, color=col, alpha=0.35)
                ax.bar(x, rw["n_q05"], width=0.72, color=col)
            ax.text(x, rw["n_p05"] + 2, f"{int(rw['n_q05'])}", ha="center", fontsize=5.6, color=fs.INK,
                    fontweight="bold")
        ax.axvline(len(order) - 0.2, color=fs.GRID, lw=0.8)
    ax.axhline(336 * 0.05, color=fs.INK, lw=0.7, ls=(0, (2, 2)))
    ax.set_xticks(list(range(len(order))) + xg)
    ax.set_xticklabels(labels + [g[1] for g in GEN][:len(xg)], rotation=40, ha="right", fontsize=5.6)
    ax.set_ylabel("Correlations (of 336)")
    ax.set_title("ENIGMA: 12 maps × 28 annotations", fontsize=6.2, loc="left")
    ax.legend(handles=[Patch(color=fs.MUTED, alpha=0.35, label="P < 0.05"), Patch(color=fs.MUTED, label="FDR q < 0.05")],
              loc="upper right", fontsize=5.4, handlelength=1.0, borderaxespad=0.2)
    fs.panel(ax, "d", x=-0.25)
    fs.save(fig, "Fig2", OUT)


# ======================================================================================================================
# Fig. 3: homologue leakage in cross-validation under the null (lambda = 77 mm), and inner-loop tuning
# ======================================================================================================================
def _box(ax, v, pos, col, hatch=None):
    bp = ax.boxplot(v, positions=[pos], widths=0.36, patch_artist=True, showfliers=False,
                    medianprops=dict(color=fs.INK, lw=0.9), whiskerprops=dict(color=fs.INK2, lw=0.6),
                    capprops=dict(color=fs.INK2, lw=0.6), boxprops=dict(lw=0.6))
    b = bp["boxes"][0]
    b.set_facecolor("white" if hatch else col)
    b.set_edgecolor(col)
    if hatch:
        b.set_hatch(hatch)
    ax.scatter(pos, np.mean(v), marker="D", s=6, color="white", edgecolor=fs.INK, linewidth=0.5, zorder=4)


def _fpr_heatmap(ax, fpr, m, order, nulls, title, fontsize=5.3, label_fontsize=None):
    cm = LinearSegmentedColormap.from_list("seq", ["#f4f8fd"] + fs.SEQ[1:])
    sub = fpr[fpr["model"] == m].set_index(["scheme", "null"])
    M = np.array([[sub.loc[(sc, n), "fpr"] for n in nulls] for sc in order])
    V = np.array([[sub.loc[(sc, n), "bradley"] for n in nulls] for sc in order])
    ax.imshow(np.clip(M, 0.01, 1), cmap=cm, norm=LogNorm(0.02, 1), aspect="auto")
    for i in range(len(order)):
        for j in range(len(nulls)):
            val = M[i, j]
            nominal = V[i, j] == "acceptable"     # bold-outlined: within the 2.5-7.5% band
            ax.text(j, i, f"{100 * val:.0f}" if val >= 0.095 else f"{100 * val:.1f}", ha="center", va="center",
                    fontsize=fontsize, color="white" if val > 0.25 else fs.INK, fontweight="bold" if nominal else "normal")
            if nominal:
                ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=fs.INK, lw=0.7))
    ax.set_xticks(range(len(nulls)))
    ax.set_xticklabels([fs.NULL_SHORT[n] for n in nulls], rotation=40, ha="right",
                       **({"fontsize": label_fontsize} if label_fontsize else {}))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, **({"fontsize": 5.8} if label_fontsize else {}))
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=6.3, loc="left")


def fig3():
    fs.apply()
    d77 = cv_null_results().query("cond == 'lam77'")
    fpr = pd.read_csv(RES / "v3_stats_cv_fpr.csv").query("cond == 'lam77'")
    st = scheme_leakage()
    leak = {s_: fs.sel(st, base=s_)["homologue_in_training"].mean() for s_ in ["R", "H", "S", "SH", "Sw", "DD", "DD-H"]}
    leak["R"] = st.query("base == 'R' and scheme in ['R_seed0','R_seed1','R_seed2','R_seed3','R_seed4']")[
        "homologue_in_training"].mean()  # the five partitions used in the simulation
    leak["H"] = 0.0
    ORDER = ["Sw", "R", "S", "DD", "H", "SH", "DD-H"]
    NULLS = ["shuffle", "spin_std", "spin_fslr", "spin_aligned", "spin_sym"]
    n_maps = d77["map"].nunique()
    fig = plt.figure(figsize=(fs.W2, 118 * fs.MM))
    g1 = fig.add_gridspec(1, 4, left=0.065, right=0.985, top=0.93, bottom=0.60, wspace=0.45,
                          width_ratios=[1.35, 0.75, 1.05, 1.05])

    # a: R2_cv on signal-free maps (partition schemes)
    ax = fig.add_subplot(g1[0, 0])
    PART = ["Sw", "R", "S", "H", "SH"]
    for i, sc in enumerate(PART):
        for m, off, h in [("ridge", -0.2, None), ("knn", 0.2, "//////")]:
            _box(ax, fs.sel(d77, model=m, scheme=sc)["r2"].to_numpy(), i + off, fs.SCHEME_COL[sc], h)
    ax.axhline(0, color=fs.INK, lw=0.7)
    ax.set_xticks(range(len(PART)))
    ax.set_xticklabels(PART)
    ax.set_ylim(-1.0, 0.8)
    ax.set_ylabel("$R^2_{cv}$ on signal-free maps")
    ax.set_title("Partition schemes", fontsize=6.0, loc="left")
    ax.legend(handles=[Patch(facecolor=fs.MUTED, label="ridge"),
                       Patch(facecolor="white", edgecolor=fs.MUTED, hatch="//////", label="kNN")],
              loc="upper right", fontsize=5.4, handlelength=1.2, ncol=2, borderaxespad=0.2)
    fs.panel(ax, "a", x=-0.26)

    # b: mean test-set r on signal-free maps (distance-dependent schemes)
    ax = fig.add_subplot(g1[0, 1])
    for i, sc in enumerate(["DD", "DD-H"]):
        for m, off, h in [("ridge", -0.2, None), ("knn", 0.2, "//////")]:
            _box(ax, fs.sel(d77, model=m, scheme=sc)["stat"].to_numpy(), i + off, fs.SCHEME_COL[sc], h)
    ax.axhline(0, color=fs.INK, lw=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["DD", "DD-H"])
    ax.set_ylim(-0.55, 0.8)
    ax.set_ylabel("Mean test-set r")
    ax.set_title("Distance-dependent", fontsize=6.0, loc="left")
    fs.panel(ax, "b", x=-0.5)

    # c, d: false-positive-rate heatmaps
    for mi, m in enumerate(["ridge", "knn"]):
        ax = fig.add_subplot(g1[0, 2 + mi])
        _fpr_heatmap(ax, fpr, m, ORDER, NULLS, f"{'Ridge' if m == 'ridge' else 'kNN'}: false positives (%)",
                     label_fontsize=5.6)
        fs.panel(ax, "c" if mi == 0 else "d", x=-0.3)

    g2 = fig.add_gridspec(1, 2, left=0.065, right=0.985, top=0.44, bottom=0.08, wspace=0.3, width_ratios=[1.3, 1.0])
    # e: false positives vs homologue leakage
    ax = fig.add_subplot(g2[0, 0])
    for n, mk in [("shuffle", "o"), ("spin_std", "^"), ("spin_sym", "D")]:
        for m, fill in [("ridge", True), ("knn", False)]:
            xs = [leak[s] for s in ORDER]
            ys = [fs.sel(fpr, model=m, scheme=s, null=n)["fpr"].iloc[0] for s in ORDER]
            jit = np.array([{"H": -0.03, "SH": 0.0, "DD-H": 0.03}.get(sc, 0.0) for sc in ORDER])
            ax.scatter(np.array(xs) + jit, ys, marker=mk, s=13, facecolor=fs.NULL_COL[n] if fill else "white",
                       edgecolor=fs.NULL_COL[n], linewidth=0.7, zorder=3)
            if n == "shuffle" and m == "ridge":
                for sc, x_, y_ in zip(ORDER, np.array(xs) + jit, ys):
                    ax.annotate(sc, (x_, y_), xytext=(3, 3), textcoords="offset points", fontsize=5.0, color=fs.INK2)
    ax.axhline(0.05, color=fs.INK, lw=0.7, ls=(0, (2, 2)))
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1])
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("Homologue leakage (share of test regions whose homologue is in the training set)")
    ax.set_ylabel("False-positive rate (%, log scale)")
    ax.set_xlim(-0.08, 1.1)
    ax.legend(handles=[Line2D([], [], color=fs.NULL_COL[n], marker=mk, ms=3, ls="", label=fs.NULL_SHORT[n])
                       for n, mk in [("shuffle", "o"), ("spin_std", "^"), ("spin_sym", "D")]] +
              [Line2D([], [], color=fs.MUTED, marker="o", ms=3, ls="", label="ridge (filled)"),
               Line2D([], [], color=fs.MUTED, marker="o", mfc="white", ms=3, ls="", label="kNN (open)")],
              loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=5.4, handlelength=1.2, ncol=2, framealpha=0.95)
    ax.set_title("False positives track homologue leakage", fontsize=6.3, loc="left")
    fs.panel(ax, "e", x=-0.12)

    # f: inner tuning (pre-specified plain leave-one-out vs group-matched; lambda = 50 mm, first 200 maps)
    ax = fig.add_subplot(g2[0, 1])
    pre = pd.read_csv(RES / "sim_B1_linear.csv").query("model == 'ridge' and map < 200")
    post = pd.read_csv(RES / "rev_sim_B1.csv").query("cond == 'B1_lam50' and model == 'ridge'")
    pos = 0
    ticks, labels = [], []
    for s in ["H", "S", "SH"]:
        for lab, df, col in [("LOO", pre, fs.MUTED), ("matched", post, fs.SCHEME_COL[s])]:
            bp = ax.boxplot(fs.sel(df, scheme=s)["r2"].to_numpy(), positions=[pos], widths=0.6, patch_artist=True,
                            showfliers=False, medianprops=dict(color=fs.INK, lw=0.9),
                            whiskerprops=dict(color=fs.INK2, lw=0.6), capprops=dict(color=fs.INK2, lw=0.6),
                            boxprops=dict(lw=0))
            bp["boxes"][0].set_facecolor(col)
            ticks.append(pos)
            labels.append(lab)
            pos += 1
        ax.text(pos - 1.5, -0.13, s, ha="center", fontsize=6, color=fs.INK, transform=ax.get_xaxis_transform())
        pos += 0.5
    ax.axhline(0, color=fs.INK, lw=0.6)
    ax.set_ylim(-2.8, 0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=5.4)
    ax.set_ylabel("Ridge $R^2_{cv}$ on signal-free maps")
    ax.set_title("Inner tuning: plain leave-one-out vs mirroring the outer split", fontsize=6.3, loc="left")
    fs.panel(ax, "f", x=-0.16)
    fig.text(0.99, 0.005, f"λ = 77 mm, ρ = 0.9, {n_maps} maps (a–e); λ = 50 mm, 200 maps (f)", ha="right",
             fontsize=5.2, color=fs.MUTED)
    fs.save(fig, "Fig3", OUT)


# ======================================================================================================================
# Fig. 4: testing protocols as classifiers of maps with vs without signal (lambda = 77 mm)
# ======================================================================================================================
def fig4():
    fs.apply()
    roc = pd.read_csv(RES / "v3_stats_roc_curves.csv")
    auc = pd.read_csv(RES / "v3_stats_auc.csv")
    cmx = pd.read_csv(RES / "v3_stats_confusion.csv")
    est = pd.read_csv(RES / "v3_stats_estimation.csv")
    SCH = ["R", "Sw", "DD", "H", "SH", "DD-H"]
    lv = [0.1, 0.3, 0.5]
    fig = plt.figure(figsize=(fs.W2, 112 * fs.MM))
    g = fig.add_gridspec(2, 3, left=0.065, right=0.985, top=0.94, bottom=0.08, wspace=0.34, hspace=0.55)

    # a, b: ROC curves with the symmetric spin at s = 0.3
    for mi, m in enumerate(["ridge", "knn"]):
        ax = fig.add_subplot(g[0, mi])
        for sc in SCH:
            r = fs.sel(roc, model=m, scheme=sc, null="spin_sym", s=0.3)
            a = fs.sel(auc, model=m, scheme=sc, null="spin_sym", s=0.3).iloc[0]
            ax.plot(r["fpr"], r["tpr"], color=fs.SCHEME_COL[sc], ls=fs.SCHEME_LS[sc], lw=1.1, label=f"{sc} ({a['auc']:.2f})")
        ax.plot([0, 1], [0, 1], color=fs.MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(axis="both")
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.set_title(f"{fs.MODEL_LAB[m]}, s = 0.3 (AUC)", fontsize=6.2, loc="left")
        ax.legend(loc="lower right", fontsize=5.3, handlelength=1.8, labelspacing=0.25)
        fs.panel(ax, "ab"[mi], x=-0.3)

    # c: AUC vs signal share
    ax = fig.add_subplot(g[0, 2])
    for sc in SCH:
        for m, mk, fill in [("ridge", "o", True), ("knn", "s", False)]:
            a = fs.sel(auc, model=m, scheme=sc, null="spin_sym").sort_values("s")
            x = np.arange(3) + (-0.06 if m == "ridge" else 0.06)
            ax.errorbar(x, a["auc"], yerr=[a["auc"] - a["lo"], a["hi"] - a["auc"]], color=fs.SCHEME_COL[sc], lw=0.9,
                        ls=fs.SCHEME_LS[sc] if m == "ridge" else (0, (1, 1.2)), marker=mk, ms=2.8,
                        mfc=fs.SCHEME_COL[sc] if fill else "white", elinewidth=0.5, capsize=0)
    ax.axhline(0.5, color=fs.MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xticks(range(3))
    ax.set_xticklabels([str(v) for v in lv])
    ax.set_xlabel("Signal share s")
    ax.set_ylabel("AUC (95% CI)")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("AUC, symmetric spin (filled: ridge; open: kNN)", fontsize=6.0, loc="left")
    fs.panel(ax, "c", x=-0.28)

    # d: size-adjusted power at s = 0.3; e: PPV at 10% prevalence (nominal threshold)
    nulls, w = ["spin_std", "spin_fslr", "spin_sym"], 0.26
    for k, (col, thr, ylab, title) in enumerate([
            ("sensitivity", "size_adjusted", "Size-adjusted power", "Ridge, s = 0.3: power at a true 5% error rate"),
            ("PPV_prev10", "nominal_0.05", "Positive predictive value", "Ridge, s = 0.3, P < 0.05, 10% of maps with signal")]):
        ax = fig.add_subplot(g[1, k])
        for ni, n in enumerate(nulls):
            vals = [fs.sel(cmx, model="ridge", scheme=sc_, null=n, s=0.3, threshold=thr)[col].iloc[0] for sc_ in SCH]
            ax.bar(np.arange(len(SCH)) + (ni - 1) * w, vals, width=w - 0.03, color=fs.NULL_COL[n], label=fs.NULL_SHORT[n])
        ax.set_xticks(range(len(SCH)))
        ax.set_xticklabels(SCH)
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=6.0, loc="left")
        if k == 0:
            ax.legend(loc="upper left", fontsize=5.4, ncol=3, handlelength=1.0, columnspacing=0.8)
        fs.panel(ax, "de"[k], x=-0.28)

    # f: estimation
    ax = fig.add_subplot(g[1, 2])
    for sc in ["R", "Sw", "H", "SH"]:
        e = fs.sel(est, model="ridge", scheme=sc).sort_values("s")
        x = np.arange(3) + {"R": -0.15, "Sw": -0.05, "H": 0.05, "SH": 0.15}[sc]
        ax.errorbar(x, e["skill_median"], yerr=[e["skill_median"] - e["skill_q25"], e["skill_q75"] - e["skill_median"]],
                    color=fs.SCHEME_COL[sc], marker="o", ms=2.8, lw=0.9, ls=fs.SCHEME_LS[sc], elinewidth=0.6,
                    capsize=0, label=sc)
    leg = ax.legend(loc="upper left", fontsize=5.2, ncol=4, handlelength=1.6, columnspacing=0.9)
    ax.add_artist(leg)
    tr = fs.sel(est, model="ridge", scheme="R").sort_values("s")["true_share"].to_numpy()
    em = pd.read_csv(RES / "v3_estimand.csv")
    rep = em[em["model"] == "ridge"].groupby("s")[["replicate_r2", "replicate_r2_loo", "replicate_r2_best"]].median()
    rep = rep.reindex(lv)
    band = ax.fill_between(np.arange(3), rep["replicate_r2_loo"], rep["replicate_r2_best"], color=fs.MUTED, alpha=0.2,
                           lw=0)
    bench = [ax.plot(np.arange(3), v, color=fs.INK, lw=0.8, ls=ls)[0]
             for v, ls in [(tr, (0, (3, 2))), (rep["replicate_r2"], (0, (1, 1.2)))]]
    for v in [tr, rep["replicate_r2"]]:    # exact values; drawn separately so that the legend shows the line styles
        ax.plot(np.arange(3), v, color=fs.INK, ls="none", marker="_", ms=8)
    ax.legend(bench + [band], ["realized share", "replicate R², pair-tuned", "replicate R², LOO-tuned to best penalty"],
              loc="upper left", bbox_to_anchor=(0, 0.91), fontsize=5.2, handlelength=2.6)
    ax.axhline(0, color=fs.MUTED, lw=0.5)
    ax.set_ylim(-0.35, 1.3)
    ax.set_yticks(np.arange(-0.2, 0.81, 0.2))
    ax.set_xticks(range(3))
    ax.set_xticklabels([str(v) for v in lv])
    ax.set_xlabel("Signal share s")
    ax.set_ylabel("Cross-validated skill (median, IQR)")
    ax.set_title("Ridge: cross-validated skill vs benchmarks", fontsize=6.0, loc="left")
    fs.panel(ax, "f", x=-0.28)
    fs.save(fig, "Fig4", OUT)


# ======================================================================================================================
# Fig. 5: reanalysis of 12 ENIGMA disorder maps with 28 annotations
# ======================================================================================================================
def count_heatmap(ax, M, ylabels, xlabels, fontsize=6):
    cm = LinearSegmentedColormap.from_list("cnt", ["#f7f7f5"] + fs.SEQ[1:])
    ax.imshow(M, cmap=cm, vmin=0, vmax=12, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=fontsize,
                    color="white" if M[i, j] > 7 else fs.INK)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=5.8)
    ax.set_xticks(range(len(xlabels)))
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


def fig5():
    fs.apply()
    d = pd.read_csv(RES / "v3_real_cv.csv")
    d = d[d["set"] == "multimodal"]
    ORDER = ["R", "Sw", "DD", "S", "H", "SH", "DD-H"]
    NULLS = ["shuffle", "spin_enigma", "spin_std", "spin_fslr", "spin_aligned", "spin_sym"]
    fig = plt.figure(figsize=(fs.W2, 100 * fs.MM))
    g = fig.add_gridspec(2, 3, left=0.08, right=0.985, top=0.94, bottom=0.1, wspace=0.42, hspace=0.62,
                         width_ratios=[1.15, 1, 1])

    # a: ridge R2 under R vs SH per map
    ax = fig.add_subplot(g[:, 0])
    r = d.query("model == 'ridge' and scheme == 'R'").set_index("map")
    sh = d.query("model == 'ridge' and scheme == 'SH'").set_index("map")
    order = r["stat"].sort_values().index.tolist()
    yy = np.arange(len(order))
    lo = -0.8
    ax.hlines(yy, np.clip(sh.loc[order, "stat"], lo, None), r.loc[order, "stat"], color=fs.GRAY_LIGHT, lw=1.2, zorder=1)
    for sc, df, col in [("R", r, fs.SCHEME_COL["R"]), ("SH", sh, fs.SCHEME_COL["SH"])]:
        sig = (df.loc[order, "q_spin_sym"] < 0.05).to_numpy()
        x = np.clip(df.loc[order, "stat"].to_numpy(), lo, None)
        ax.scatter(x[sig], yy[sig], s=22, color=col, zorder=3, edgecolor=col, linewidth=0.8)
        ax.scatter(x[~sig], yy[~sig], s=22, facecolor="white", edgecolor=col, zorder=3, linewidth=0.8)
    ax.axvline(0, color=fs.INK2, lw=0.6)
    ax.set_yticks(yy)
    ax.set_yticklabels(order)
    ax.set_xlim(lo - 0.05, 0.9)
    ax.set_xlabel("Ridge $R^2_{cv}$")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=fs.SCHEME_COL["R"], label="R (10 partitions)"),
                       Line2D([], [], marker="o", ls="", color=fs.SCHEME_COL["SH"], label="SH"),
                       Line2D([], [], marker="o", ls="", mfc="white", color=fs.MUTED, label="not significant")],
              loc="lower right", fontsize=5.4, handletextpad=0.2)
    ax.set_title("Filled: FDR q < 0.05, symmetric spin", fontsize=6.0, loc="left")
    fs.panel(ax, "a", x=-0.3, y=1.02)

    # b, c: number of significant maps per protocol
    for mi, m in enumerate(["ridge", "knn"]):
        ax = fig.add_subplot(g[mi, 1:])
        M = np.array([[int((fs.sel(d, model=m, scheme=sc_)[f"q_{n_}"] < 0.05).sum()) for sc_ in ORDER] for n_ in NULLS])
        count_heatmap(ax, M, [fs.NULL_SHORT[n] for n in NULLS], ORDER)
        ax.set_xticklabels([fs.SCHEME_LAB[sc] for sc in ORDER], fontsize=5.4, rotation=20, ha="right")
        ax.set_title(f"{fs.MODEL_LAB[m]}: disorder maps 'predicted' (of 12; FDR q < 0.05)", fontsize=6.2, loc="left")
        fs.panel(ax, "bc"[mi], x=-0.14, y=1.03)
    fs.save(fig, "Fig5", OUT)


# ======================================================================================================================
# Supplementary Figs 1-7
# ======================================================================================================================
def supp_lambda():
    c = pd.read_csv(RES / "rev_lambda_calibration.csv")
    s = pd.read_csv(RES / "rev_lambda_summary.csv", index_col=0).iloc[:, 0]
    fig, ax = plt.subplots(figsize=(fs.W1, 62 * fs.MM))
    ax.fill_between(c["generator_lambda"], c["estimated_iqr_lo"], c["estimated_iqr_hi"], color=fs.SEQ[1], alpha=0.6,
                    linewidth=0, label="simulated estimates (IQR)")
    ax.plot(c["generator_lambda"], c["estimated_median"], color=fs.SEQ[4], marker="o", ms=2.5, label="median")
    ax.axhline(s["empirical_targets_mm"], color=fs.INK, lw=0.7, ls=(0, (2, 2)))
    ax.axvline(s["calibrated_generator_lambda_mm"], color=fs.INK, lw=0.7, ls=(0, (2, 2)))
    ax.text(s["calibrated_generator_lambda_mm"] + 2, 12, f"calibrated λ = {s['calibrated_generator_lambda_mm']:.0f} mm",
            fontsize=5.8)
    ax.text(32, s["empirical_targets_mm"] + 1.5, f"targets: plug-in estimate {s['empirical_targets_mm']:.1f} mm",
            fontsize=5.8)
    ax.set_xlabel("Generating correlation length λ (mm)")
    ax.set_ylabel("Plug-in correlogram estimate (mm)")
    ax.grid(axis="both")
    ax.legend(loc="lower right", fontsize=5.6)
    fs.save(fig, "SuppFig1_lambda_calibration", OUT_SUPP)


def supp_biv_annot():
    d = pd.read_csv(RES / "v3_bivariate_sim.csv.gz")
    d = d[(d["cond"] == "lam77_rho0.9") & (d["pvalue"] == "two_sided")]
    b = pd.read_csv(RES / "v3_bilateral_null.csv").query("kind == 'predictor'").set_index("map")
    fig, ax = plt.subplots(figsize=(fs.W1 * 1.2, 70 * fs.MM))
    for n in ["spin_enigma", "spin_std", "spin_fslr", "spin_sym"]:
        g = d[d["null"] == n].groupby("annotation")["p"].apply(lambda p: (p < 0.05).mean())
        ax.scatter(b.loc[g.index, "bilateral_r"], g.values, s=10, color=fs.NULL_COL[n], label=fs.NULL_SHORT[n],
                   edgecolor="white", linewidth=0.4)
    ax.axhline(0.05, color=fs.INK, lw=0.7, ls=(0, (2, 2)))
    ax.set_xlabel("Bilateral correlation of the annotation")
    ax.set_ylabel("False-positive rate (1,000 null maps)")
    ax.grid(axis="both")
    ax.legend(fontsize=5.6, loc="upper left")
    ax.set_title("Per-annotation false-positive rates (λ = 77 mm, ρ = 0.9)", fontsize=6.2, loc="left")
    fs.save(fig, "SuppFig2_bivariate_per_annotation", OUT_SUPP)


def supp_partition():
    q = pd.read_csv(RES / "exploratory_partition_robustness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(fs.W2 * 0.7, 58 * fs.MM), gridspec_kw={"wspace": 0.3})
    for ax, m in zip(axes, ["ridge", "knn"]):
        for n, mk in [("shuffle", "o"), ("spin_std", "^"), ("spin_sym", "D")]:
            g = q[(q["model"] == m) & (q["null"] == n)].sort_values("homologue_same_fold")
            ax.plot(100 * g["homologue_same_fold"], g["fpr"], marker=mk, ms=3, lw=1, color=fs.NULL_COL[n],
                    label=fs.NULL_SHORT[n])
        ax.axhline(0.05, color=fs.INK, lw=0.7, ls=(0, (2, 2)))
        ax.set_xlabel("Regions sharing a fold with their homologue (%)")
        ax.set_ylim(0, 1)
        ax.set_title(f"{fs.MODEL_LAB[m]}: single random partitions (λ = 50 mm)", fontsize=6.2, loc="left")
    axes[0].set_ylabel("False-positive rate")
    axes[0].legend(fontsize=5.6)
    fs.save(fig, "SuppFig3_partition_dose_response", OUT_SUPP)


def supp_cv50():
    fpr = pd.read_csv(RES / "v3_stats_cv_fpr.csv").query("cond == 'lam50'")
    fig, axes = plt.subplots(1, 2, figsize=(fs.W2 * 0.7, 62 * fs.MM), gridspec_kw={"wspace": 0.35})
    for ax, m in zip(axes, ["ridge", "knn"]):
        _fpr_heatmap(ax, fpr, m, ["Sw", "R", "S", "DD", "H", "SH", "DD-H"],
                     ["shuffle", "spin_std", "spin_fslr", "spin_aligned", "spin_sym"],
                     f"{fs.MODEL_LAB[m]}: false positives (%), λ = 50 mm", fontsize=5.4)
    fs.save(fig, "SuppFig4_cv_fpr_lambda50", OUT_SUPP)


def supp_roc_all():
    roc = pd.read_csv(RES / "v3_stats_roc_curves.csv")
    SCH = ["R", "Sw", "DD", "H", "SH", "DD-H"]
    nulls = ["spin_std", "spin_fslr", "spin_sym"]
    fig, axes = plt.subplots(2, 9, figsize=(fs.W2, 58 * fs.MM), gridspec_kw={"wspace": 0.12, "hspace": 0.45})
    for mi, m in enumerate(["ridge", "knn"]):
        for ni, n in enumerate(nulls):
            for li, lv in enumerate([0.1, 0.3, 0.5]):
                ax = axes[mi, ni * 3 + li]
                for sc in SCH:
                    r = fs.sel(roc, model=m, scheme=sc, null=n, s=lv)
                    ax.plot(r["fpr"], r["tpr"], color=fs.SCHEME_COL[sc], ls=fs.SCHEME_LS[sc], lw=0.8, label=sc)
                ax.plot([0, 1], [0, 1], color=fs.MUTED, lw=0.5, ls=(0, (2, 2)))
                ax.set_aspect("equal")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.tick_params(labelsize=5)
                if li != 0 or ni != 0:
                    ax.set_yticklabels([])
                ax.set_title(f"s = {lv}", fontsize=5.4)
                if li == 0 and ni == 0:
                    ax.set_ylabel(("Ridge" if m == "ridge" else "kNN") + "\nTPR", fontsize=6)
                if mi == 1:
                    ax.set_xlabel("FPR", fontsize=5.4, labelpad=1)
    for ni, n in enumerate(nulls):  # null-model headers above each group of three panels
        x0 = (axes[0, ni * 3].get_position().x0 + axes[0, ni * 3 + 2].get_position().x1) / 2
        fig.text(x0, axes[0, 0].get_position().y1 + 0.075, fs.NULL_SHORT[n] + " spin", ha="center", fontsize=6.4,
                 color=fs.INK)
    axes[1, 0].legend(fontsize=4.8, loc="lower right", handlelength=1.2, labelspacing=0.15)
    fs.save(fig, "SuppFig5_roc_all", OUT_SUPP)


def supp_rf_dd():
    d = pd.read_csv(RES / "v3_real_cv.csv")
    d = d[d["set"] == "multimodal"]
    fig, axes = plt.subplots(1, 2, figsize=(fs.W2 * 0.8, 70 * fs.MM), gridspec_kw={"wspace": 0.5})
    ax = axes[0]
    order = d[(d["model"] == "ridge") & (d["scheme"] == "DD")].set_index("map")["stat"].sort_values().index
    for m, mk in [("ridge", "o"), ("knn", "s")]:
        a = d[(d["model"] == m) & (d["scheme"] == "DD")].set_index("map")
        b = d[(d["model"] == m) & (d["scheme"] == "DD-H")].set_index("map")
        yy = np.arange(len(order)) + (0.15 if m == "knn" else -0.15)
        ax.hlines(yy, b.loc[order, "stat"], a.loc[order, "stat"], color=fs.GRAY_LIGHT, lw=1)
        ml = "ridge" if m == "ridge" else "kNN"
        ax.scatter(a.loc[order, "stat"], yy, marker=mk, s=12, color=fs.SCHEME_COL["DD"], label=f"DD ({ml})")
        ax.scatter(b.loc[order, "stat"], yy, marker=mk, s=12, color=fs.SCHEME_COL["DD-H"], label=f"DD-H ({ml})")
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order)
    ax.axvline(0, color=fs.INK2, lw=0.6)
    ax.set_xlabel("Mean test-set r")
    ax.legend(fontsize=5.4, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.grid(axis="x")
    ax.set_title("Distance-dependent CV on ENIGMA maps", fontsize=6.2, loc="left")
    fs.panel(ax, "a", x=-0.3)
    ax = axes[1]
    rf = pd.concat([pd.read_csv(RES / f) for f in ("v3_rf_main.csv", "v3_rf_alt.csv")])
    offs = {("main", "R"): -0.22, ("main", "SH"): 0.0, ("alt", "SH"): 0.22}
    grp = dict(tuple(rf.groupby(["variant", "scheme"])))
    for (v, sc) in [k for k in offs if k in grp]:
        g = grp[(v, sc)].set_index("map").sort_index()
        col = fs.SCHEME_COL["R"] if sc == "R" else fs.SCHEME_COL["SH"]
        mk = "o" if v == "main" else "^"
        sig = g["q_spin_sym"] < 0.05
        yy = np.arange(len(g)) + offs.get((v, sc), 0.0)
        lab = f"{sc} ({'pre-specified settings' if v == 'main' else 'alternative settings'})"
        ax.scatter(g["r2"][sig], yy[sig.to_numpy()], s=14, color=col, marker=mk, label=lab)
        ax.scatter(g["r2"][~sig], yy[~sig.to_numpy()], s=14, facecolor="white", edgecolor=col, marker=mk,
                   linewidths=0.9, label=None if sig.any() else lab)
    ax.set_yticks(np.arange(len(g)))
    ax.set_yticklabels(g.index)
    ax.axvline(0, color=fs.INK2, lw=0.6)
    ax.set_xlabel("Random forest $R^2_{cv}$ (filled: FDR q < 0.05)")
    ax.legend(fontsize=5.4, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.grid(axis="x")
    ax.set_title("Random forests over repeated partitions", fontsize=6.2, loc="left")
    fs.panel(ax, "b", x=-0.3)
    fs.save(fig, "SuppFig6_dd_rf_real", OUT_SUPP)


def supp_sens():
    hy = pd.read_csv(RES / "v3_sens_hyper.csv")
    ms = pd.read_csv(RES / "v3_sens_measurement.csv")
    fig, axes = plt.subplots(1, 3, figsize=(fs.W2, 62 * fs.MM), gridspec_kw={"wspace": 0.45, "width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    cnt = hy.groupby(["model", "variant", "scheme"])["q_spin_sym"].apply(lambda q: int((q < 0.05).sum())).unstack()
    cnt = cnt[["R", "SH", "DD-H"]]
    rows_ = [("ridge", "alpha 1e-2..1e4 (main)"), ("ridge", "alpha 1e-3..1e6"), ("knn", "k=3"), ("knn", "k=5 (main)"),
             ("knn", "k=10"), ("knn", "k=20")]
    labs_ = ["Ridge, α 10$^{-2}$–10$^{4}$ (main)", "Ridge, α 10$^{-3}$–10$^{6}$", "kNN, k = 3", "kNN, k = 5 (main)",
             "kNN, k = 10", "kNN, k = 20"]
    cnt = cnt.loc[[r for r in rows_ if r in cnt.index]]
    ax.imshow(cnt.to_numpy(), cmap=LinearSegmentedColormap.from_list("c", ["#f7f7f5"] + fs.SEQ[1:]), vmin=0, vmax=12,
              aspect="auto")
    for i in range(cnt.shape[0]):
        for j in range(cnt.shape[1]):
            ax.text(j, i, str(cnt.iloc[i, j]), ha="center", va="center", fontsize=6)
    ax.set_yticks(range(cnt.shape[0]))
    ax.set_yticklabels([labs_[rows_.index(r)] for r in cnt.index], fontsize=5.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["R", "SH", "DD-H"])
    ax.grid(False)
    ax.set_title("Hyper-parameters: maps significant\n(of 12; symmetric spin, FDR)", fontsize=6.0, loc="left")
    fs.panel(ax, "a", x=-0.9)
    rm = pd.read_csv(RES / "v3_real_metrics.csv")
    rm = rm[rm["set"] == "multimodal"]
    for k, m in enumerate(["ridge", "knn"]):
        ax = axes[1 + k]
        order = sorted(ms["map"].unique())
        for sc, col in [("R", fs.SCHEME_COL["R"]), ("SH", fs.SCHEME_COL["SH"])]:
            g = ms[(ms["model"] == m) & (ms["scheme"] == sc)]
            q = g.groupby("map")["r2"].quantile([0.025, 0.5, 0.975]).unstack().loc[order]
            yy = np.arange(len(order)) + (0.15 if sc == "SH" else -0.15)
            ax.hlines(yy, q[0.025], q[0.975], color=col, lw=1.2)
            ax.scatter(q[0.5], yy, s=8, color=col, zorder=3, label=sc)
            u = rm[(rm["model"] == m) & (rm["scheme"] == sc)].set_index("map").loc[order, "r2"]
            ax.scatter(u, yy, s=11, marker="D", facecolor="white", edgecolor=col, linewidths=0.7, zorder=4)
        ax.axvline(0, color=fs.INK2, lw=0.6)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if k == 0 else [], fontsize=5.6)
        ax.set_xlabel("$R^2_{cv}$ across 100 noise-perturbation replicates")
        ax.grid(axis="x")
        ax.set_title(f"{fs.MODEL_LAB[m]} (median, 95% interval)", fontsize=6.0, loc="left")
        if k == 0:
            hd = [Line2D([], [], marker="o", ls="", ms=3, color=fs.SCHEME_COL["R"], label="R, perturbed"),
                  Line2D([], [], marker="o", ls="", ms=3, color=fs.SCHEME_COL["SH"], label="SH, perturbed"),
                  Line2D([], [], marker="D", ls="", ms=3, mfc="white", mec=fs.INK2, label="unperturbed")]
            ax.legend(handles=hd, fontsize=5.6, loc="upper center", bbox_to_anchor=(1.12, -0.2), ncol=3, frameon=False)
            fs.panel(ax, "b", x=-0.35)
    fs.save(fig, "SuppFig7_sensitivity", OUT_SUPP)


def _fpr_bars(ax, labels, fpr, lo, hi, cols, ref=None):
    """False-positive rates with 95% CIs as bars; grey band 2.5-7.5%; ref: optional {position: value} (open diamonds)."""
    x = np.arange(len(labels))
    ax.axhspan(0.025, 0.075, color=fs.GRAY_LIGHT, lw=0, zorder=0)
    ax.axhline(0.05, color=fs.INK, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.bar(x, fpr, width=0.62, color=cols, zorder=2)
    ax.errorbar(x, fpr, yerr=[np.asarray(fpr) - lo, np.asarray(hi) - fpr], fmt="none", ecolor=fs.INK, elinewidth=0.6,
                capsize=0, zorder=3)
    for k, v in (ref or {}).items():
        ax.scatter(k, v, marker="D", s=12, facecolor="white", edgecolor=fs.INK, linewidth=0.6, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("False-positive rate")


def supp_half():
    """Supplementary Fig. 12: one value per homologous pair (34 regions)."""
    biv = pd.read_csv(RES / "v3_stats_half_biv.csv")
    main = pd.read_csv(RES / "v3_stats_bivariate_fpr.csv").query("cond == 'lam77_rho0.9' and pvalue == 'two_sided'")
    cv = pd.read_csv(RES / "v3_stats_half_cv_fpr.csv")
    pw = pd.read_csv(RES / "v3_stats_half_power.csv")
    auc68 = pd.read_csv(RES / "v3_stats_auc.csv").query("null == 'spin_sym'")
    real = pd.read_csv(RES / "v3_stats_half_real.csv")
    short = {"ridge": "Ridge", "knn": "kNN"}
    fig = plt.figure(figsize=(fs.W2, 118 * fs.MM))
    g = fig.add_gridspec(2, 4, left=0.07, right=0.985, top=0.94, bottom=0.12, wspace=0.65, hspace=0.75,
                         width_ratios=[1.25, 0.8, 0.8, 1.25])

    # a: map-to-map correlations
    ax = fig.add_subplot(g[0, 0])
    rows = [(main, None, "shuffle", "68: shuffle"), (main, None, "spin_std", "68: conventional"),
            (main, None, "spin_sym", "68: symmetric")]
    rows += [(biv, h, n, f"34 {'avg' if h == 'avg' else 'left'}: {n}") for h in ("avg", "left") for n in ("shuffle", "spin")]
    sel = [(d_[d_["null"] == n] if h is None else d_[(d_["how"] == h) & (d_["null"] == n)]).iloc[0] for d_, h, n, _ in rows]
    cols = [{"shuffle": fs.NULL_COL["shuffle"], "spin_std": fs.NULL_COL["spin_std"], "spin_sym": fs.NULL_COL["spin_sym"],
             "spin": fs.INK2}[r[2]] for r in rows]
    _fpr_bars(ax, [r[3] for r in rows], [s_["fpr"] for s_ in sel], [s_["lo"] for s_ in sel], [s_["hi"] for s_ in sel],
              cols)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1)
    ax.set_title("Map-to-map correlations", fontsize=6.2, loc="left")
    fs.panel(ax, "a", x=-0.42)

    # b, c: cross-validation false-positive rates
    order = ["avg: R", "avg: S", "avg: DD", "left: R", "left: S", "left: DD"]
    fs.NULL_SHORT.setdefault("spin", "Spin")
    for k, m in enumerate(["ridge", "knn"]):
        ax = fig.add_subplot(g[0, 1 + k])
        sub = cv[cv["model"] == m].assign(scheme=lambda x: x["how"] + ": " + x["scheme"])
        _fpr_heatmap(ax, sub, m, order, ["shuffle", "spin"], f"{short[m]}: CV FPR (%)", fontsize=5.6)
        if k:
            ax.set_yticklabels([])
        fs.panel(ax, "bc"[k], x=-0.75 if k == 0 else -0.2)

    # d: ENIGMA maps
    ax = fig.add_subplot(g[0, 3])
    cols_ = [(s_, n) for s_ in ["R", "S", "DD"] for n in ["shuffle", "spin"]]
    rws = [(h, m) for h in ("avg", "left") for m in ("ridge", "knn")]
    M = np.array([[real[(real["how"] == h) & (real["model"] == m) & (real["scheme"] == s_) & (real["null"] == n)][
        "n_bh05"].iloc[0] for s_, n in cols_] for h, m in rws])
    count_heatmap(ax, M, [f"{h}, {short[m]}" for h, m in rws], cols_)
    ax.set_xticklabels([f"{s_}, {n}" for s_, n in cols_], rotation=40, ha="right", fontsize=5.8)
    ax.set_title("ENIGMA: maps significant (BH)", fontsize=6.2, loc="left")
    fs.panel(ax, "d", x=-0.45)

    # e, f: AUC against the signal share, 34 regions (spin) versus 68 regions (symmetric spin)
    for k, m in enumerate(["ridge", "knn"]):
        ax = fig.add_subplot(g[1, 2 * k:2 * k + 2])
        for s_, ls in [("R", "-"), ("S", (0, (4, 1.5))), ("DD", (0, (1, 1.2)))]:
            for h, mk in [("avg", "o"), ("left", "s")]:
                e = pw[(pw["how"] == h) & (pw["model"] == m) & (pw["scheme"] == s_)].sort_values("s")
                ax.errorbar(np.arange(3) + (-0.05 if h == "avg" else 0.05), e["auc"],
                            yerr=[e["auc"] - e["lo"], e["hi"] - e["auc"]], color=fs.SCHEME_COL[s_],
                            ls=ls, marker=mk, ms=2.8, lw=0.9, elinewidth=0.5, capsize=0,
                            mfc=fs.SCHEME_COL[s_] if h == "avg" else "white", label=f"34 {h}: {s_}")
        for s_, ls in [("SH", "-"), ("DD-H", (0, (4, 1.5)))]:
            e = auc68[(auc68["model"] == m) & (auc68["scheme"] == s_)].sort_values("s")
            ax.plot(np.arange(3), e["auc"], color=fs.MUTED, lw=1.8, ls=ls, alpha=0.6, label=f"68: {s_}, symmetric")
        ax.axhline(0.5, color=fs.MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.set_xticks(range(3))
        ax.set_xticklabels(["0.1", "0.3", "0.5"])
        ax.set_xlabel("Signal share s")
        ax.set_ylabel("AUC (95% CI)")
        ax.set_ylim(0.45, 1.0)
        ax.set_title(f"{fs.MODEL_LAB[m]}: positive control", fontsize=6.2, loc="left")
        ax.legend(fontsize=5.0, ncol=2, loc="upper left", handlelength=1.8)
        fs.panel(ax, "ef"[k], x=-0.12)
    fs.save(fig, "SuppFig12_one_value_per_pair", OUT_SUPP)


def supp_glasser():
    """Supplementary Fig. 13: the Glasser parcellation (360 regions)."""
    gb = pd.read_csv(RES / "v3_stats_glasser_biv.csv")
    biv = gb[gb["cond"] == "lam77_rho0.9"].set_index("null")
    rho0 = gb[gb["cond"] == "lam77_rho0.0"].set_index("null")
    cv = pd.read_csv(RES / "v3_stats_glasser_cv_fpr.csv")
    nulls = ["shuffle", "spin_enigma", "spin_std", "spin_fslr", "spin_aligned", "spin_sym"]
    fig = plt.figure(figsize=(fs.W2, 64 * fs.MM))
    g = fig.add_gridspec(1, 4, left=0.06, right=0.99, top=0.9, bottom=0.24, wspace=0.55,
                         width_ratios=[1, 1.1, 1.05, 1.05])
    ax = fig.add_subplot(g[0, 0])
    sp = nulls[1:]
    ax.bar(np.arange(len(sp)), biv.loc[sp, "bilateral_r_null"], width=0.62, color=[fs.NULL_COL[n] for n in sp])
    ax.axhline(biv["bilateral_r_maps"].iloc[0], color=fs.INK, lw=0.7, ls=(0, (2, 2)))
    ax.text(len(sp) - 0.5, biv["bilateral_r_maps"].iloc[0] + 0.02, "maps", ha="right", va="bottom", fontsize=5.4)
    ax.set_xticks(np.arange(len(sp)))
    ax.set_xticklabels([fs.NULL_SHORT[n] for n in sp], rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Bilateral r of null maps (median)")
    ax.set_title("Bilateral structure", fontsize=6.2, loc="left")
    fs.panel(ax, "a", x=-0.38)
    ax = fig.add_subplot(g[0, 1])
    _fpr_bars(ax, [fs.NULL_SHORT[n] for n in nulls], biv.loc[nulls, "fpr"], biv.loc[nulls, "lo"], biv.loc[nulls, "hi"],
              [fs.NULL_COL[n] for n in nulls], ref={k: rho0.loc[n, "fpr"] for k, n in enumerate(nulls)})
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1)
    ax.set_title("Map-to-map correlations", fontsize=6.2, loc="left")
    fs.panel(ax, "b", x=-0.38)
    for k, m in enumerate(["ridge", "knn"]):
        ax = fig.add_subplot(g[0, 2 + k])
        _fpr_heatmap(ax, cv, m, ["Sw", "R", "S", "H", "SH"], ["shuffle", "spin_std", "spin_sym"],
                     f"{'Ridge' if m == 'ridge' else 'kNN'}: CV false positives (%)", fontsize=5.6)
        fs.panel(ax, "cd"[k], x=-0.3)
    fs.save(fig, "SuppFig13_glasser", OUT_SUPP)


def supp():
    for f in (supp_lambda, supp_biv_annot, supp_partition, supp_cv50, supp_roc_all, supp_rf_dd, supp_sens, supp_half,
              supp_glasser):
        fs.apply()
        try:
            f()
            print(f.__name__, "ok", flush=True)
        except FileNotFoundError as e:
            print(f.__name__, "skipped:", e, flush=True)


# ======================================================================================================================
# Supplementary Figs 8-11: pre-specified analyses (style of the initial analysis plan)
# ======================================================================================================================
class old:
    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
    GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#ffffff"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    GRAY_LIGHT = "#dcdbd5"
    NULL_COLORS = {"shuffle": BLUE, "spin_std": ORANGE, "spin_sym": "#1baf7a"}
    NULL_LABELS = {"shuffle": "Shuffle", "spin_std": "Spin (conventional)", "spin_sym": "Spin (symmetric)"}
    MODEL_LABELS = {"ridge": "Ridge regression", "knn": "k-nearest neighbours", "rf": "Random forest"}
    NULLS = ["shuffle", "spin_std", "spin_sym"]

    @staticmethod
    def apply():
        mpl.rcdefaults()
        mpl.rcParams.update({
            "font.family": FONT, "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
            "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
            "axes.edgecolor": old.AXIS, "axes.labelcolor": old.INK2, "xtick.color": old.INK2, "ytick.color": old.INK2,
            "text.color": old.INK, "axes.titlecolor": old.INK, "axes.linewidth": 0.8,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "axes.grid.axis": "y", "grid.color": old.GRID, "grid.linewidth": 0.6,
            "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
            "savefig.facecolor": old.SURF, "figure.facecolor": old.SURF, "axes.facecolor": old.SURF,
            "lines.linewidth": 1.6, "patch.linewidth": 0,
        })

    @staticmethod
    def panel(ax, letter, x=-0.12, y=1.06):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left",
                color=old.INK)

    @staticmethod
    def legend_r_sh():
        h = [Line2D([], [], color=old.NULL_COLORS[n], lw=1.6, label=old.NULL_LABELS[n]) for n in old.NULLS]
        return h + [Line2D([], [], color=old.INK2, lw=1.2, marker="o", ms=3.5, label="Random CV (R)"),
                    Line2D([], [], color=old.INK2, lw=1.2, ls=(0, (3, 1.5)), marker="s", mfc="white", ms=3.5,
                           label="Spatial + homologue CV (SH)")]


def supp8_null_simulation():
    import json
    d = pd.read_csv(RES / "sim_B1_linear.csv")
    if (RES / "sim_B1_rf_obs.csv").exists():
        d = pd.concat([d, pd.read_csv(RES / "sim_B1_rf_obs.csv")], ignore_index=True)
    fpr = pd.read_csv(RES / "table_B1_fpr.csv")
    rf = json.load(open(RES / "rev_summary.json")).get("rf_sim", {})
    if rf and not ((fpr["model"] == "rf") & (fpr["null"] == "shuffle")).any():
        fpr = pd.concat([fpr, pd.DataFrame([{"model": "rf", "scheme": k, "null": "shuffle", "fpr": v["shuffle_fpr"],
                                              "lo": v["lo"], "hi": v["hi"], "n": v["n"]} for k, v in rf.items()])],
                        ignore_index=True)
    models = [m for m in ["ridge", "knn", "rf"] if m in d["model"].unique()]
    schemes = ["R", "H", "S", "SH"]
    fig, axes = plt.subplots(2, len(models), figsize=(7.1, 4.9), gridspec_kw={"hspace": 0.62, "wspace": 0.28})
    axes = np.atleast_2d(axes)
    for mi, m in enumerate(models):
        ax = axes[0, mi]
        data = [d[(d["model"] == m) & (d["scheme"] == s_)]["r2"].to_numpy() for s_ in schemes]
        keep = [i for i, v in enumerate(data) if len(v)]
        bp = ax.boxplot([data[i] for i in keep], positions=keep, widths=0.55, patch_artist=True, showfliers=False,
                        whis=(5, 95), medianprops=dict(color=old.INK, lw=1.4), whiskerprops=dict(color=old.MUTED, lw=0.9),
                        capprops=dict(color=old.MUTED, lw=0.9), boxprops=dict(lw=0))
        for patch in bp["boxes"]:
            patch.set_facecolor("#c9dcf3")
        ax.axhline(0, color=old.INK2, lw=0.8, ls=(0, (3, 2)))
        ax.scatter(keep, [np.mean(data[i]) for i in keep], marker="D", s=14, color="white", edgecolor=old.INK,
                   linewidth=0.8, zorder=4)
        for i in keep:
            ax.text(i + 0.31, np.mean(data[i]), f"{np.mean(data[i]):+.2f}", ha="left", va="center", fontsize=6.0,
                    color=old.INK2)
        ax.set_axisbelow(True)
        ax.set_xticks(range(4))
        ax.set_xticklabels(schemes)
        ax.set_title(old.MODEL_LABELS[m], fontsize=7.8)
        lo = min(np.percentile(data[i], 5) for i in keep)
        hi = max(np.percentile(data[i], 95) for i in keep)
        ax.set_ylim(lo - 0.1, hi + 0.15)
        ax.set_xlim(-0.5, 3.8)
        if mi == 0:
            ax.set_ylabel("Out-of-sample $R^2_{cv}$\n(null maps: true value ≤ 0)")
            old.panel(ax, "a", x=-0.35)
        ax = axes[1, mi]
        w = 0.26
        for ni, nt in enumerate(old.NULLS):
            g = fpr.query("model == @m and null == @nt").set_index("scheme").reindex(schemes)
            x = np.arange(4) + (ni - 1) * w
            ok = g["fpr"].notna().to_numpy()
            ax.bar(x[ok], g["fpr"][ok], width=w - 0.03, color=old.NULL_COLORS[nt], label=old.NULL_LABELS[nt],
                   edgecolor="white", linewidth=0.6)
            err = np.vstack([g["fpr"] - g["lo"], g["hi"] - g["fpr"]])[:, ok]
            ax.errorbar(x[ok], g["fpr"][ok], yerr=err, fmt="none", ecolor=old.INK2, elinewidth=0.7, capsize=1.2)
        ax.axhline(0.05, color=old.INK, lw=0.9, ls=(0, (3, 2)))
        ax.text(-0.45, 0.055, "α = 0.05", va="bottom", ha="left", fontsize=6.2, color=old.INK)
        ax.set_axisbelow(True)
        ax.set_xticks(range(4))
        ax.set_xticklabels(schemes)
        ax.set_ylim(0, 0.82)
        ax.set_xlabel("Cross-validation scheme")
        if mi == 0:
            ax.set_ylabel("False-positive rate")
            old.panel(ax, "b", x=-0.35)
        if m == "rf":
            ax.text(0.02, 0.97, "RF: shuffle p-values against a common\nnull of 300 shuffled maps (300 maps)",
                    transform=ax.transAxes, fontsize=5.8, color=old.MUTED, va="top")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), handlelength=1.2)
    fig.savefig(OUT_SUPP / "SuppFig8_prespecified_null_simulation.png")
    plt.close(fig)


def supp9_sensitivity():
    f2 = pd.read_csv(RES / "table_B2_fpr.csv")
    f3 = pd.read_csv(RES / "table_B3_fpr.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.6), gridspec_kw={"hspace": 0.6, "wspace": 0.32,
                                                                     "width_ratios": [1.25, 0.8, 0.9]})
    for mi, m in enumerate(["ridge", "knn"]):
        for ci, (by, fixed, xlabel) in enumerate([("rho", "lam == 50", "Bilateral coupling ρ (λ = 50 mm)"),
                                                  ("lam", "rho == 0.9", "Correlation length λ, mm (ρ = 0.9)")]):
            ax = axes[mi, ci]
            for nt in old.NULLS:
                for s, ls, mk in [("R", "-", "o"), ("SH", (0, (3, 1.5)), "s")]:
                    g = f2.query(f"model == @m and scheme == @s and null == @nt and {fixed}").sort_values(by)
                    ax.plot(g[by], g["fpr"], ls=ls, marker=mk, ms=3.5, color=old.NULL_COLORS[nt], lw=1.4,
                            mfc=old.NULL_COLORS[nt] if s == "R" else "white", mew=1.0)
            ax.axhline(0.05, color=old.INK, lw=0.8, ls=(0, (2, 2)))
            if by == "rho":
                ax.set_xticks([0, 0.5, 0.9, 1.0])
                ax.set_ylabel(f"{old.MODEL_LABELS[m]}\nfalse-positive rate")
            else:
                ax.set_xscale("log")
                ax.set_xticks([25, 50, 100])
                ax.set_xticklabels(["25", "50", "100"])
                ax.minorticks_off()
            ax.set_xlabel(xlabel)
            ax.set_ylim(0, 0.95)
            ax.set_axisbelow(True)
        ax = axes[mi, 2]      # BrainSMASH generator
        w = 0.26
        for ni, nt in enumerate(old.NULLS):
            g = f3.query("model == @m and null == @nt").set_index("scheme").reindex(["R", "SH"])
            x = np.arange(2) + (ni - 1) * w
            ax.bar(x, g["fpr"], width=w - 0.03, color=old.NULL_COLORS[nt], edgecolor="white", linewidth=0.6)
            ax.errorbar(x, g["fpr"], yerr=np.vstack([g["fpr"] - g["lo"], g["hi"] - g["fpr"]]), fmt="none",
                        ecolor=old.INK2, elinewidth=0.7, capsize=1.2)
        ax.axhline(0.05, color=old.INK, lw=0.8, ls=(0, (2, 2)))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["R", "SH"])
        ax.set_ylim(0, 0.95)
        ax.set_xlabel("CV scheme")
        ax.set_axisbelow(True)
    axes[0, 0].set_title("Bilateral coupling", fontsize=7.8, loc="left")
    axes[0, 1].set_title("Correlation length", fontsize=7.8, loc="left")
    axes[0, 2].set_title("BrainSMASH surrogates", fontsize=7.8, loc="left")
    old.panel(axes[0, 0], "a", x=-0.38)
    old.panel(axes[0, 1], "b", x=-0.3)
    old.panel(axes[0, 2], "c", x=-0.3)
    fig.legend(handles=old.legend_r_sh(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03), handlelength=2.0,
               columnspacing=1.0, fontsize=6.5)
    fig.savefig(OUT_SUPP / "SuppFig9_prespecified_sensitivity.png")
    plt.close(fig)


def supp10_positive_control():
    pw = pd.read_csv(RES / "table_B4_power.csv")
    raw = pd.read_csv(RES / "sim_B4_power.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8), gridspec_kw={"hspace": 0.62, "wspace": 0.3})
    for mi, m in enumerate(["ridge", "knn"]):
        ax = axes[0, mi]
        for nt in old.NULLS:
            for s, ls, mk in [("R", "-", "o"), ("SH", (0, (3, 1.5)), "s")]:
                g = pw.query("model == @m and scheme == @s and null == @nt").sort_values("s")
                ax.plot(g["s"], g["power"], ls=ls, marker=mk, ms=3.5, color=old.NULL_COLORS[nt], lw=1.4,
                        mfc=old.NULL_COLORS[nt] if s == "R" else "white", mew=1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0.1, 0.3, 0.5])
        ax.set_xlabel("Signal share s")
        ax.set_title(old.MODEL_LABELS[m], fontsize=7.8)
        if mi == 0:
            ax.set_ylabel("Power (α = 0.05)")
            old.panel(ax, "a", x=-0.28)
        ax.set_axisbelow(True)
        ax = axes[1, mi]
        offs = {"R": -0.3, "H": -0.1, "S": 0.1, "SH": 0.3}
        ticks, labels = [], []
        for gi, lvl in enumerate([0.1, 0.3, 0.5]):
            true = raw.query("model == @m and s == @lvl")["realised_share"].mean()
            ax.hlines(true, gi - 0.42, gi + 0.42, color=old.MUTED, lw=1.0, ls=(0, (2, 2)), zorder=1)
            if gi == 0:
                ax.text(gi - 0.42, true + 0.04, "true share", fontsize=6, color=old.MUTED, ha="left")
            for sc in ["R", "H", "S", "SH"]:
                q1, med, q3 = np.percentile(raw.query("model == @m and scheme == @sc and s == @lvl")["r2"], [25, 50, 75])
                x = gi + offs[sc]
                col = old.BLUE if sc == "R" else (old.INK if sc == "SH" else old.INK2)
                ax.vlines(x, q1, q3, color=col, lw=2.0, alpha=0.85)
                ax.scatter(x, med, s=16, color="white", edgecolor=col, linewidth=1.1, zorder=3)
                ticks.append(x)
                labels.append(sc)
            ax.text(gi, -0.2, f"s = {lvl}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7,
                    color=old.INK2)
        ax.axhline(0, color=old.INK2, lw=0.6)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-1.0, 0.75)
        ax.grid(axis="x", visible=False)
        if mi == 0:
            ax.set_ylabel("$R^2_{cv}$ (median and IQR)")
            old.panel(ax, "b", x=-0.28)
        ax.set_axisbelow(True)
    fig.legend(handles=old.legend_r_sh(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02), fontsize=6.5,
               handlelength=2.0, columnspacing=1.0)
    fig.savefig(OUT_SUPP / "SuppFig10_prespecified_positive_control.png")
    plt.close(fig)


def supp11_real_data():
    dl = pd.read_csv(RES / "real_linear.csv").query("set == 'multimodal'")
    rows = [{"model": r["model"], "scheme": r["scheme"], "map": r["map"], "r2": r["r2"], "q_naive": r["q_shuffle"],
             "q_rec": r["q_spin_sym"]} for _, r in dl.iterrows()]
    if (RES / "real_rf.csv").exists():
        rf = pd.read_csv(RES / "real_rf.csv")
        for (s, mp), g in rf.groupby(["scheme", "map"]):
            rows.append({"model": "rf", "scheme": s, "map": mp, "r2": g["r2"].iloc[0],
                         "q_naive": g.query("null == 'shuffle'")["q"].iloc[0] if "shuffle" in set(g["null"]) else np.nan,
                         "q_rec": g.query("null == 'spin_sym'")["q"].iloc[0] if "spin_sym" in set(g["null"]) else np.nan})
    t = pd.DataFrame(rows)
    models = [m for m in ["ridge", "knn", "rf"] if m in t["model"].unique()]
    order = t.query("model == 'ridge' and scheme == 'R'").sort_values("r2")["map"].tolist()
    fig, axes = plt.subplots(1, len(models), figsize=(7.1, 3.6), sharey=True, gridspec_kw={"wspace": 0.12})
    axes = np.atleast_1d(axes)
    for mi, m in enumerate(models):
        ax = axes[mi]
        g = t.query("model == @m").set_index(["scheme", "map"])
        yy = np.arange(len(order))
        rR = np.array([g.loc[("R", mp), "r2"] for mp in order])
        rS = np.array([g.loc[("SH", mp), "r2"] for mp in order])
        qR = np.array([g.loc[("R", mp), "q_naive"] for mp in order])
        qS = np.array([g.loc[("SH", mp), "q_rec"] for mp in order])
        lo = -1.0
        ax.hlines(yy, np.clip(rS, lo, None), rR, color=old.GRAY_LIGHT, lw=1.3, zorder=1)
        for yv, r, q in zip(yy, rR, qR):
            ax.scatter(r, yv, s=26, zorder=3, color=old.BLUE if q < 0.05 else "white", edgecolor=old.BLUE, linewidth=1.2)
        for yv, r, q in zip(yy, rS, qS):
            ax.scatter(max(r, lo), yv, s=26, marker="s", zorder=3, color=old.ORANGE if q < 0.05 else "white",
                       edgecolor=old.ORANGE, linewidth=1.2)
            if r < lo:
                ax.annotate("", xy=(lo - 0.04, yv), xytext=(lo + 0.02, yv),
                            arrowprops=dict(arrowstyle="-|>", color=old.ORANGE, lw=0.8, mutation_scale=6))
        ax.axvline(0, color=old.INK2, lw=0.7)
        ax.set_xlim(lo - 0.08, 0.8)
        ax.set_title(old.MODEL_LABELS[m], fontsize=7.8)
        ax.set_xlabel("$R^2_{cv}$")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.text(0.98, 0.02, f"FDR-significant:\nR + shuffle: {int((qR < 0.05).sum())}/12\n"
                f"SH + sym. spin: {int((qS < 0.05).sum())}/12", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.2, color=old.INK2, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=old.GRID, lw=0.6))
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels(order)
    h = [Line2D([], [], marker="o", ls="none", color=old.BLUE, ms=5, label="Random CV (R); filled = q < 0.05, shuffle"),
         Line2D([], [], marker="s", ls="none", color=old.ORANGE, ms=5,
                label="Spatial + homologue CV (SH); filled = q < 0.05, symmetric spin")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), fontsize=6.5, columnspacing=1.5)
    fig.savefig(OUT_SUPP / "SuppFig11_prespecified_real_data.png")
    plt.close(fig)


def prespecified():
    OUT_SUPP.mkdir(parents=True, exist_ok=True)
    for f in (supp8_null_simulation, supp9_sensitivity, supp10_positive_control, supp11_real_data):
        old.apply()
        f()
        print(f.__name__, "ok", flush=True)


# ======================================================================================================================
# Source Data workbook
# ======================================================================================================================
def source_data():
    X, Y, reg, geo = hl.load_data()
    hom = reg["homologue_idx"].to_numpy()
    sheets = {}
    b = pd.read_csv(RES / "v3_bilateral_null.csv")
    sheets["Fig1a"] = b[["map", "kind", "bilateral_r"]]
    rows = []
    for surf, sname in [("fsa5", "fsaverage5"), ("conte69", "fs_LR 32k")]:
        for parc, pname in [("aparc", "Desikan-Killiany"), ("glasser_360", "Glasser")]:
            lk, ang = sphere_angles(surf, parc)
            rows += [{"sphere": sname, "parcellation": pname, "left_label": k, "angle_deg": float(a)}
                     for k, a in zip(lk, ang)]
    sheets["Fig1b"] = pd.DataFrame(rows)
    sheets["Fig1c"] = b[b["kind"] == "target"][["map", "bilateral_r"] + [c for c in b.columns if c.startswith("median_")]]
    tests = example_test_sets(reg)
    rows = []
    for s in ["R", "H", "Sw", "SH", "DD", "DD-H"]:
        te = tests[s]
        for r in range(68):
            rr = "test" if r in te else ("homologue of a test region (training)" if hom[r] in te else "training")
            rows.append({"scheme": s, "region": reg["name"][r], "role": rr})
    sheets["Fig1d"] = pd.DataFrame(rows)
    sheets["Fig1e"] = pd.read_csv(RES / "v3_scheme_structure.csv")
    sheets["Fig2a-c"] = pd.read_csv(RES / "v3_stats_bivariate_fpr.csv")
    sheets["Fig2d"] = pd.read_csv(RES / "v3_bivariate_real_counts.csv")
    sheets["Fig2d_all_tests"] = pd.read_csv(RES / "v3_bivariate_real.csv")
    sheets["Fig2a_generative"] = pd.read_csv(RES / "v3_stats_gen_biv_fpr.csv")
    sheets["Fig2d_generative"] = pd.read_csv(RES / "v3_gen_real_counts.csv")
    d = cv_null_results()
    sheets["Fig3a-b_per_map"] = d[d["cond"] == "lam77"][["map", "model", "scheme", "stat", "r2", "skill", "r"]]
    sheets["Fig3c-e"] = pd.read_csv(RES / "v3_stats_cv_fpr.csv")
    pre = pd.read_csv(RES / "sim_B1_linear.csv")
    pre = pre[(pre["model"] == "ridge") & (pre["map"] < 200)][["map", "scheme", "r2"]]
    pre["tuning"] = "plain leave-one-out (pre-specified)"
    post = pd.read_csv(RES / "rev_sim_B1.csv")
    post = post[(post["cond"] == "B1_lam50") & (post["model"] == "ridge")][["map", "scheme", "r2"]]
    post["tuning"] = "nested, mirrors outer scheme"
    sheets["Fig3f"] = pd.concat([pre, post])
    for name, f in [("Fig4a-b_ROC", "v3_stats_roc_curves.csv"), ("Fig4c_AUC", "v3_stats_auc.csv"),
                    ("Fig4d-e_confusion", "v3_stats_confusion.csv"), ("Fig4f_estimation", "v3_stats_estimation.csv"),
                    ("Fig4f_replicate_R2", "v3_estimand.csv")]:
        sheets[name] = pd.read_csv(RES / f)
    rc = pd.read_csv(RES / "v3_real_cv.csv")
    sheets["Fig5a"] = rc[(rc["set"] == "multimodal") & (rc["model"] == "ridge") & rc["scheme"].isin(["R", "SH"])][
        ["scheme", "map", "stat", "p_spin_sym", "q_spin_sym"]]
    sheets["Fig5b-c"] = pd.read_csv(RES / "v3_stats_real_counts.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT / "Source_Data.xlsx") as w:
        pd.DataFrame({"Sheet": list(sheets), "Rows": [len(v) for v in sheets.values()]}).to_excel(
            w, sheet_name="README", index=False)
        for k, v in sheets.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    print("wrote", OUT / "Source_Data.xlsx", len(sheets), "sheets")


# ======================================================================================================================
# Supplementary Data 1: Supplementary Tables 1-10 (one sheet per table), built from the result files
# ======================================================================================================================
LAB = {"shuffle": "shuffle", "spin_std": "conventional", "spin_enigma": "ENIGMA TB", "spin_fslr": "fs_LR",
       "spin_aligned": "aligned", "spin_sym": "symmetric", "bs_indep": "BrainSMASH, independent",
       "bs_shared": "BrainSMASH, shared seed", "bs_whole": "BrainSMASH, whole brain", "bs_sym": "BrainSMASH, symmetric",
       "msr_indep": "MSR, independent", "msr_shared": "MSR, shared seed", "msr_sym": "MSR, symmetric",
       "knn": "kNN", "ridge": "ridge", "two_sided": "two-sided", "enigma_toolbox": "ENIGMA TB, as implemented",
       "enigma_toolbox_doubled": "ENIGMA TB, P doubled", "multimodal": "28 annotations", "receptor_only": "19 receptors"}
TABLE9 = [  # verdicts under the pre-specified decision rules (Supplementary Notes 1 and 3)
    ("H1a: null R²(R) > R²(SH)", "Supported", "Supported with group-matched tuning at λ = 50 and 77 mm"),
    ("H1b: R–SH gap larger for flexible models", "Not supported (ridge gap larger, an artefact of inner-loop leakage)",
     "Supported for kNN versus ridge; random-forest gap 0.48 (pre-specified single partition)"),
    ("H2: homology component > proximity component (kNN, RF)", "Supported for kNN",
     "Supported for kNN and ridge on both decomposition paths (R → H → SH and R → Sw → SH)"),
    ("H3a: shuffle FPR > 0.10 under R", "Supported (ridge 57%, kNN 74%, RF 98%)", "Supported (λ = 77 mm: 85–91%)"),
    ("H3b: symmetric-spin FPR ≤ 0.075 in every scheme", "Not supported (kNN 8.9% under R, 8.7% under H)",
     "Not supported (Supplementary Table 4a)"),
    ("H4: conventional spin more liberal than symmetric spin under R", "Supported",
     "Supported; aligning the sphere frames removes most of the difference"),
    ("H5: fewer significant maps under (SH, symmetric spin) than (R, shuffle)", "Supported",
     "Supported (ridge 12 → 0; kNN 12 → 1)"),
    ("H6a: generative nulls with independent hemispheres lose bilateral structure (median bilateral r of null maps "
     "below 0.2)", "Supported (median bilateral r of null maps: BrainSMASH 0.00, Moran 0.00)",
     "– (specified before the additional analysis)"),
    ("H6b: at ρ = 0.9 they are liberal for map-to-map correlations and more liberal than the conventional spin",
     "Supported (13.8% and 16.3%; conventional spin 8.7%)", "–"),
    ("H6c: their symmetric variants are within the 2.5–7.5% band or within 2 points of the symmetric spin",
     "Supported (4.4% and 5.4%; symmetric spin 5.7%)", "–"),
    ("H6d: in cross-validation, independent variants are liberal under R and Sw, and symmetric variants have "
     "false-positive rates ≤ 7.5% under SH and DD-H",
     "Supported (independent: 48.5–82.0% under R and Sw; symmetric: 0.5–6.0% under SH and DD-H)", "–")]


def _cond(c):
    """lam77_rho0.9 -> 'λ = 77 mm, ρ = 0.9'; lam50 -> '50 mm'."""
    lam, rho, syn = re.match(r"lam(\d+)(?:_rho([\d.]+))?(_synthX)?$", c).groups()
    if rho is None:
        return f"{lam} mm"
    return f"λ = {lam} mm, ρ = {rho}" + (" (simulated annotations)" if syn else "")


def _lab(s):
    """Replace result-file codes (spin_sym, bs_indep, ...) by their labels inside a string."""
    return re.sub(r"\b(spin_\w+|bs_\w+|msr_\w+|shuffle)\b", lambda m: LAB.get(m.group(1), m.group(1)), s)


def supp_tables():
    """{sheet: (caption, note, table)}; numbers are kept at full precision."""
    T = {}

    d = pd.read_csv(RES / "v3_misalignment.csv")
    T["S1"] = ("Supplementary Table 1. Deviation of homologous parcel centroids from mirror symmetry on the fsaverage5 "
               "and fs_LR spheres.", "Angles between each left-hemisphere centroid and the mirror image of its right "
               "homologue.", pd.DataFrame({
                   "Sphere": d["sphere"], "Parcellation": d["parcellation"], "Pairs": d["n_pairs"],
                   "Mean (°)": d["mean_deg"], "Median (°)": d["median_deg"], "Max (°)": d["max_deg"],
                   "Kabsch rotation (°)": d["kabsch_rotation_deg"], "After alignment (°)": d["after_alignment_mean_deg"],
                   "L/R frames identical": d["vertex_frames_identical"],
                   "L/R frames mirrored": d["vertex_frames_mirrored"]}))

    d = pd.read_csv(RES / "v3_stats_bivariate_fpr.csv")
    T["S2a"] = ("Supplementary Table 2a. False-positive rates of null models for map-to-map correlations (28 annotations "
                "per map; two-sided α = 0.05).", "Confidence intervals from 2,000 bootstrap resamples of maps; tests "
                "against 5% are one-sample t-tests on per-map rejection proportions, Holm-corrected. Simulated "
                "annotations: the 28 annotations replaced by 28 simulated maps. 'ENIGMA TB, as implemented': the "
                "toolbox's directional P value; 'ENIGMA TB, P doubled': the same P value doubled.", pd.DataFrame({
                    "Condition": d["cond"].map(_cond), "Null": d["null"].map(LAB), "P value": d["pvalue"].map(LAB),
                    "Maps": d["n_maps"], "FPR": d["fpr"], "CI low": d["lo"], "CI high": d["hi"],
                    "P vs 5%": d["p_vs_0.05"], "Holm P": d["p_holm"], "Classification": d["bradley"]}))

    d = pd.read_csv(RES / "v3_stats_bivariate_contrasts.csv")
    T["S2b"] = ("Supplementary Table 2b. Paired differences in rejection rates between each null and the symmetric spin "
                "(map-level Wilcoxon signed-rank tests).", "", pd.DataFrame({
                    "Condition": d["cond"].map(_cond), "Contrast": d["contrast"].map(_lab), "Difference": d["diff"],
                    "CI low": d["lo"], "CI high": d["hi"], "Wilcoxon P": d["p_wilcoxon"], "Holm P": d["p_holm"]}))

    d = pd.read_csv(RES / "v3_approx_coupling_biv.csv")
    T["S2c"] = ("Supplementary Table 2c. Map-to-map correlations when hemispheric correspondence is approximate (500 "
                "maps × 28 annotations; λ = 77 mm, ρ = 0.9, τ = 20 mm).", "", pd.DataFrame({
                    "Null": d["null"].map(LAB), "FPR": d["fpr"], "CI low": d["lo"], "CI high": d["hi"],
                    "Median bilateral r of maps": d["median_bilateral_r_of_maps"]}))

    d = pd.read_csv(RES / "v3_bivariate_joint_null.csv")
    T["S2d"] = ("Supplementary Table 2d. Number of significant disorder–annotation correlations (of 336) against a "
                "joint null that preserves the dependence among tests.", "", pd.DataFrame({
                    "Null": d["null"].map(LAB), "Observed P<0.05": d["observed_p05"],
                    "Joint-null median": d["null_median_p05"], "Joint-null 95th pct": d["null_q95_p05"],
                    "P (count)": d["P_count_p05"], "Observed q<0.05": d["observed_q05"],
                    "Joint-null 95th pct (q)": d["null_q95_q05"], "P (count, q)": d["P_count_q05"]}))

    b = pd.read_csv(RES / "v3_gen_bilateral.csv")
    c = pd.read_csv(RES / "v3_gen_real_counts.csv").set_index("null")
    obs = b.drop_duplicates("map")["bil_obs"]
    rows = [{"Null": "Observed disorder maps", "Bilateral r, median over maps": obs.median(),
             "Bilateral r, minimum": obs.min(), "Bilateral r, maximum": obs.max()}]
    for v, g in b.groupby("variant", sort=False):
        rows.append({"Null": LAB[v], "Bilateral r, median over maps": g["bil_null_median"].median(),
                     "Bilateral r, minimum": g["bil_null_median"].min(), "Bilateral r, maximum": g["bil_null_median"].max(),
                     "P < 0.05": c.loc[v, "n_p05"], "BH q < 0.05": c.loc[v, "n_q05"], "Holm < 0.05": c.loc[v, "n_holm05"]})
    rows = pd.DataFrame(rows).astype({"P < 0.05": "Int64", "BH q < 0.05": "Int64", "Holm < 0.05": "Int64"})
    T["S2e"] = ("Supplementary Table 2e. Generative null models on the 12 ENIGMA disorder maps: bilateral structure of "
                "the null maps (1,000 surrogates per map; median of each map's null bilateral r, with its median and "
                "range over maps) and number of significant disorder–annotation correlations (of 336; 10,000 "
                "surrogates per map).", "Independent: each hemisphere generated separately with independent random "
                "numbers (neuromaps default); shared seed: the same random numbers for both hemispheres; whole brain: "
                "Euclidean distances between parcel centroids; symmetric: generator applied to the homologous sum and "
                "difference maps. MSR, Moran spectral randomization.", rows)

    d = pd.read_csv(RES / "v3_stats_gen_biv_fpr.csv")
    k = pd.read_csv(RES / "v3_stats_gen_biv_contrasts.csv")
    k = k[k["contrast"].str.endswith(" - spin_sym")].assign(null=lambda x: x["contrast"].str.replace(" - spin_sym", ""))
    d = d.merge(k[["cond", "null", "diff", "lo", "hi"]], on=["cond", "null"], how="left", suffixes=("", "_d"))
    order = ["shuffle", "spin_enigma", "spin_std", "spin_sym", "bs_indep", "bs_shared", "bs_whole", "bs_sym",
             "msr_indep", "msr_shared", "msr_sym"]
    d = d.sort_values(["cond", "null"], ascending=[False, True], key=lambda v: v.map(order.index) if v.name == "null"
                      else v)
    T["S2f"] = ("Supplementary Table 2f. False-positive rates of generative null models for map-to-map correlations "
                "(28 annotations; λ = 77 mm; 500 draws per generative null).", "Maps: the first 500 maps of the main "
                "simulation at ρ = 0.9 and the first 250 maps at ρ = 0; the spins are those of the main analysis "
                "(1,000 draws) on the same maps. Map-level bootstrap confidence intervals; t-tests against 5%, "
                "Holm-corrected; differences from the symmetric spin (percentage points) with map-level bootstrap "
                "confidence intervals.", pd.DataFrame({
                    "Condition": d["cond"].map(_cond), "Null": d["null"].map(LAB), "Maps": d["n_maps"],
                    "FPR": d["fpr"], "CI low": d["lo"], "CI high": d["hi"], "Holm P vs 5%": d["p_holm"],
                    "Classification": d["bradley"], "Difference from symmetric spin (points)": 100 * d["diff"],
                    "Difference, CI low": 100 * d["lo_d"], "Difference, CI high": 100 * d["hi_d"]}))

    d = pd.read_csv(RES / "v3_stats_cv_optimism.csv")
    T["S3a"] = ("Supplementary Table 3a. Cross-validated statistic on signal-free maps (R²cv for partition schemes; mean "
                "test-set r for DD and DD-H).", "Bootstrap 95% confidence intervals over maps.", pd.DataFrame({
                    "λ": d["cond"].map(_cond), "Model": d["model"].map(LAB), "Scheme": d["scheme"],
                    "Mean statistic": d["mean_stat"], "CI low": d["lo"], "CI high": d["hi"], "Median": d["median_stat"],
                    "Mean skill": d["mean_skill"], "Wilcoxon P vs 0": d["p_wilcoxon_vs_0"]}))

    d = pd.read_csv(RES / "v3_stats_dd_ols.csv")
    T["S3b"] = ("Supplementary Table 3b. Re-implementation of the published distance-dependent analysis (ordinary least "
                "squares on the 19 receptor maps, no tuning) on signal-free maps (λ = 77 mm, ρ = 0.9).",
                "500 draws per null.", pd.DataFrame({
                    "Scheme": d["scheme"], "Maps": d["n_maps"], "Mean test r": d["mean_test_r"],
                    "CI low": d["mean_test_r_lo"], "CI high": d["mean_test_r_hi"], "FPR shuffle": d["fpr_shuffle"],
                    "FPR conventional": d["fpr_spin_std"], "FPR symmetric": d["fpr_spin_sym"]}))

    d = pd.read_csv(RES / "v3_stats_cv_wilcoxon.csv")
    T["S3c"] = ("Supplementary Table 3c. Paired comparisons of null statistics between schemes (Wilcoxon signed-rank "
                "tests).", "", pd.DataFrame({
                    "λ": d["cond"].map(_cond), "Model": d["model"].map(LAB), "Contrast": d["contrast"], "Maps": d["n"],
                    "Mean difference": d["mean_diff"], "Hodges–Lehmann": d["hl"], "HL low": d["hl_lo"],
                    "HL high": d["hl_hi"], "W": d["W"], "P": d["p"], "Rank-biserial r": d["rank_biserial"],
                    "Holm P": d["p_holm"]}))

    d = pd.read_csv(RES / "v3_stats_cv_friedman_nemenyi.csv")
    T["S3d"] = ("Supplementary Table 3d. Friedman tests across the five partition schemes with Nemenyi post hoc "
                "comparisons (null R²cv).", "", pd.DataFrame({
                    "λ": d["cond"].map(_cond), "Model": d["model"].map(LAB), "Scheme A": d["a"], "Scheme B": d["b"],
                    "Mean rank A": d["mean_rank_a"], "Mean rank B": d["mean_rank_b"], "Nemenyi P": d["p_nemenyi"],
                    "Friedman χ²": d["friedman_chi2"], "Friedman P": d["friedman_p"]}))

    e = pd.read_csv(RES / "v3_estimand.csv").groupby(["model", "s"]).median(numeric_only=True).reset_index()
    T["S3e"] = ("Supplementary Table 3e. Benchmarks for cross-validated skill in the positive control (medians over "
                "150 maps per signal share).", "Replicate R²: model fitted to the whole map, evaluated on an "
                "independent replicate with the same signal and a new noise field. Ridge penalty of the whole-map fit "
                "chosen by leave-one-pair-out error (pair-tuned; main benchmark), by plain leave-one-out error "
                "(LOO-tuned) or as the single penalty per signal share with the largest median replicate R² (best "
                "penalty, chosen in hindsight; an upper bound). kNN: k = 5, no tuning. In-sample R²: the pair-tuned "
                "fit (ridge) evaluated on the map it was fitted to.", pd.DataFrame({
                    "Model": e["model"].map(LAB), "s": e["s"], "Replicate R², pair-tuned": e["replicate_r2"],
                    "Replicate R², LOO-tuned": e["replicate_r2_loo"], "Replicate R², best penalty": e["replicate_r2_best"],
                    "In-sample R²": e["insample_r2"], "Realized signal share": e["realised_share"]}))

    d = pd.read_csv(RES / "v3_dd_ols_real.csv")
    T["S3f"] = ("Supplementary Table 3f. Re-implementation of the published distance-dependent analysis on the 12 "
                "ENIGMA maps (ordinary least squares on the 19 receptor maps, no tuning; 2,000 draws per null).", "",
                pd.DataFrame({"Map": d["map"], "Scheme": d["scheme"], "Mean test r": d["mean_test_r"],
                              "P shuffle": d["p_shuffle"], "P conventional": d["p_spin_std"],
                              "P symmetric": d["p_spin_sym"]}))

    d = pd.read_csv(RES / "v3_stats_cv_fpr.csv")
    T["S4a"] = ("Supplementary Table 4a. False-positive rates of every combination of model, cross-validation scheme "
                "and null model.", "Wilson 95% confidence intervals; exact two-sided binomial tests against 5%, "
                "Holm-corrected within each condition.", pd.DataFrame({
                    "λ": d["cond"].map(_cond), "Model": d["model"].map(LAB), "Scheme": d["scheme"],
                    "Null": d["null"].map(LAB), "Rejections": d["k"], "Maps": d["n"], "FPR": d["fpr"], "CI low": d["lo"],
                    "CI high": d["hi"], "MC SE": d["mc_se"], "Binomial P": d["p_binom_vs_0.05"], "Holm P": d["p_holm"],
                    "Classification": d["bradley"]}))

    d = pd.read_csv(RES / "v3_stats_approx_coupling_cv.csv")
    T["S4b"] = ("Supplementary Table 4b. Cross-validation false-positive rates when hemispheric correspondence is "
                "approximate.", "", pd.DataFrame({
                    "Model": d["model"].map(LAB), "Scheme": d["scheme"], "Null": d["null"].map(LAB), "FPR": d["fpr"],
                    "CI low": d["lo"], "CI high": d["hi"], "Maps": d["n"], "Mean statistic": d["mean_stat"],
                    "Classification": d["bradley"]}))

    d = pd.read_csv(RES / "v3_stats_gen_cv_fpr.csv")
    order = {"model": ["ridge", "knn"], "scheme": ["R", "Sw", "SH", "DD-H"],
             "null": ["shuffle", "spin_std", "spin_sym", "bs_indep", "bs_sym", "msr_indep", "msr_sym"]}
    d = d.sort_values(["model", "scheme", "null"], key=lambda v: v.map(order[v.name].index))
    T["S4c"] = ("Supplementary Table 4c. Cross-validation false-positive rates with generative null models (first 200 "
                "maps of the main simulation; λ = 77 mm, ρ = 0.9; 500 draws).", "Shuffle, conventional and symmetric "
                "spins: results of the main simulation on the same maps. Wilson 95% confidence intervals; exact "
                "binomial tests against 5%, Holm-corrected.", pd.DataFrame({
                    "Model": d["model"].map(LAB), "Scheme": d["scheme"], "Null": d["null"].map(LAB), "Rejections": d["k"],
                    "Maps": d["n"], "FPR": d["fpr"], "CI low": d["lo"], "CI high": d["hi"], "Holm P": d["p_holm"],
                    "Classification": d["bradley"]}))

    d = pd.read_csv(RES / "v3_stats_cv_mcnemar.csv")
    T["S5a"] = ("Supplementary Table 5a. Exact McNemar tests comparing rejections of two protocols on the same null "
                "maps.", "Holm correction within each family (nulls; schemes) per condition and model.", pd.DataFrame({
                    "λ": d["cond"].map(_cond), "Model": d["model"].map(LAB), "Family": d["family"],
                    "Contrast": d["contrast"].map(_lab), "Only first rejects": d["n_only_first"],
                    "Only second rejects": d["n_only_second"], "Exact P": d["p_exact"], "Holm P": d["p_holm"]}))

    d = pd.read_csv(RES / "v3_stats_cv_cochranq.csv")
    T["S5b"] = ("Supplementary Table 5b. Cochran's Q tests across the five null models within each scheme.", "",
                pd.DataFrame({"λ": d["cond"].map(_cond), "Model": d["model"].map(LAB), "Scheme": d["scheme"],
                              "Cochran's Q": d["Q"], "df": d["df"], "P": d["p"]}))

    d = pd.read_csv(RES / "v3_stats_auc.csv")
    T["S6a"] = ("Supplementary Table 6a. Area under the ROC curve of each testing protocol (score: z relative to the "
                "map's null).", "Stratified bootstrap 95% confidence intervals (1,000 resamples).", pd.DataFrame({
                    "Model": d["model"].map(LAB), "Scheme": d["scheme"], "Null": d["null"].map(LAB), "s": d["s"],
                    "AUC": d["auc"], "CI low": d["lo"], "CI high": d["hi"], "AUC from P": d["auc_from_p"],
                    "Positive maps": d["n_pos"], "Null maps": d["n_neg"]}))

    d = pd.read_csv(RES / "v3_stats_auc_contrasts.csv")
    T["S6b"] = ("Supplementary Table 6b. Paired bootstrap comparisons of AUC between schemes.", "", pd.DataFrame({
        "Model": d["model"].map(LAB), "Null": d["null"].map(LAB), "s": d["s"], "Contrast": d["contrast"],
        "Difference": d["diff"], "CI low": d["lo"], "CI high": d["hi"], "Bootstrap P": d["p_boot"],
        "Holm P": d["p_holm"]}))

    d = pd.read_csv(RES / "v3_stats_confusion.csv")
    T["S6c"] = ("Supplementary Table 6c. Confusion matrices (positives: maps with signal; negatives: signal-free maps) "
                "at the nominal (P < 0.05) and size-adjusted thresholds.", "Size-adjusted: z-statistic above the 95th "
                "percentile of the 500 signal-free maps, so that exactly 25 of them (5%) are false positives. PPV/NPV "
                "at 10% and 50% prevalence of maps with signal; MCC for a balanced design.", pd.DataFrame({
                    "Model": d["model"].map(LAB), "Scheme": d["scheme"], "Null": d["null"].map(LAB), "s": d["s"],
                    "Threshold": d["threshold"].map({"nominal_0.05": "nominal (P < 0.05)",
                                                     "size_adjusted": "size-adjusted"}),
                    "Critical P": d["p_crit"], "Critical z": d["z_crit"], "TP": d["TP"], "FN": d["FN"], "FP": d["FP"],
                    "TN": d["TN"], "Sens.": d["sensitivity"], "Spec.": d["specificity"],
                    "Bal. acc.": d["balanced_accuracy"], "MCC": d["MCC_balanced"], "PPV 10%": d["PPV_prev10"],
                    "NPV 10%": d["NPV_prev10"], "PPV 50%": d["PPV_prev50"]}))

    rc = pd.read_csv(RES / "v3_real_cv.csv")
    rc = rc[rc["set"] == "multimodal"]
    rm = pd.read_csv(RES / "v3_real_metrics.csv")
    d = rm[rm["set"] == "multimodal"].merge(rc[["model", "scheme", "map", "p_spin_sym", "q_spin_sym", "holm_spin_sym"]],
                                            on=["model", "scheme", "map"])
    part = d[d["scheme"].isin(["R", "H", "S", "SH", "Sw"])]
    T["S7a"] = ("Supplementary Table 7a. Out-of-fold performance of ridge and kNN on the 12 ENIGMA maps (28 "
                "annotations), partition schemes.", "R and H: means over 10 random partitions. P values: symmetric "
                "spin, 2,000 draws.", pd.DataFrame({
                    "Model": part["model"].map(LAB), "Scheme": part["scheme"], "Map": part["map"], "R²cv": part["r2"],
                    "Skill": part["skill"], "RMSE": part["rmse"], "MAE": part["mae"], "Pearson r": part["pearson_r"],
                    "Spearman ρ": part["spearman_rho"], "Calib. slope": part["calib_slope"],
                    "P (sym)": part["p_spin_sym"], "q BH (sym)": part["q_spin_sym"],
                    "P Holm (sym)": part["holm_spin_sym"]}))
    dd = d[d["scheme"].isin(["DD", "DD-H"])]
    T["S7b"] = ("Supplementary Table 7b. Distance-dependent schemes on the 12 ENIGMA maps.", "P values: symmetric spin, "
                "2,000 draws.", pd.DataFrame({
                    "Model": dd["model"].map(LAB), "Scheme": dd["scheme"], "Map": dd["map"],
                    "Mean test r": dd["mean_test_r"], "Mean test R²": dd["mean_test_r2"], "Mean skill": dd["skill"],
                    "P (sym)": dd["p_spin_sym"], "q BH (sym)": dd["q_spin_sym"]}))

    c = pd.read_csv(RES / "v3_stats_real_counts.csv")
    rows = []
    for (s_, n_), g in c.groupby(["scheme", "null"], sort=False):
        r = {"Scheme": s_, "Null": LAB[n_]}
        for st in ["multimodal", "receptor_only"]:
            for m in ["ridge", "knn"]:
                x = g[(g["set"] == st) & (g["model"] == m)].iloc[0]
                r[f"{LAB[st]}, {LAB[m]}: BH"] = x["n_bh05"]
                r[f"{LAB[st]}, {LAB[m]}: Holm"] = x["n_holm05"]
        rows.append(r)
    sch = {s_: i for i, s_ in enumerate(["R", "Sw", "DD", "S", "H", "SH", "DD-H"])}
    nul = {n_: i for i, n_ in enumerate(["shuffle", "ENIGMA TB", "conventional", "fs_LR", "aligned", "symmetric"])}
    p7 = pd.DataFrame(rows).sort_values(["Scheme", "Null"], key=lambda v: v.map(sch if v.name == "Scheme" else nul))
    T["S7c"] = ("Supplementary Table 7c. Number of ENIGMA maps significant under each protocol (of 12), with BH-FDR and "
                "Holm correction, for all 28 annotations and for the 19 receptor maps as predictors.", "Which maps were "
                "significant: sheet S7c_full.", p7.reset_index(drop=True))
    T["S7c_full"] = ("Supplementary Table 7c (full). Significant ENIGMA maps under each protocol.", "Maps (BH): maps "
                     "with FDR q < 0.05.", pd.DataFrame({
                         "Predictors": c["set"].map(LAB), "Model": c["model"].map(LAB), "Scheme": c["scheme"],
                         "Null": c["null"].map(LAB), "Maps P < 0.05": c["n_p05"], "Maps BH q < 0.05": c["n_bh05"],
                         "Maps Holm < 0.05": c["n_holm05"], "Maps (BH)": c["maps_bh"].fillna("")}))

    d = pd.read_csv(RES / "v3_sens_hyper.csv")
    g = d.groupby(["model", "variant", "scheme"]).agg(med=("stat", "median"),
                                                      n=("q_spin_sym", lambda q: int((q < 0.05).sum()))).reset_index()
    T["S8a"] = ("Supplementary Table 8a. Hyper-parameter sensitivity on the ENIGMA maps.", "Per-map values: sheet "
                "S8a_full.", pd.DataFrame({
                    "Model": g["model"].map(LAB), "Variant": g["variant"], "Scheme": g["scheme"],
                    "Median statistic": g["med"], "Maps significant (BH, sym)": g["n"]}))
    T["S8a_full"] = ("Supplementary Table 8a (full). Hyper-parameter sensitivity on the ENIGMA maps, per map.",
                     "Statistic: R²cv (R, SH) or mean test-set r (DD-H); symmetric spin.", pd.DataFrame({
                         "Model": d["model"].map(LAB), "Variant": d["variant"], "Scheme": d["scheme"], "Map": d["map"],
                         "Statistic": d["stat"], "P (sym)": d["p_spin_sym"], "q BH (sym)": d["q_spin_sym"]}))

    d = pd.read_csv(RES / "v3_sens_measurement.csv")
    g = d.groupby(["model", "scheme", "map"], sort=True).agg(
        med=("r2", "median"), lo=("r2", lambda v: np.quantile(v, 0.025)), hi=("r2", lambda v: np.quantile(v, 0.975)),
        share=("q_spin_sym", lambda q: (q < 0.05).mean())).reset_index()
    T["S8b"] = ("Supplementary Table 8b. Noise-perturbation sensitivity: each map was perturbed with noise on the scale "
                "of its reported standard errors (100 replicates).", "", pd.DataFrame({
                    "Model": g["model"].map(LAB), "Scheme": g["scheme"], "Map": g["map"], "Median R²cv": g["med"],
                    "2.5%": g["lo"], "97.5%": g["hi"], "Share of replicates with q < 0.05": g["share"]}))

    d = pd.concat([pd.read_csv(RES / "v3_rf_main.csv"), pd.read_csv(RES / "v3_rf_alt.csv")])
    T["S8c"] = ("Supplementary Table 8c. Random forests on the ENIGMA maps (symmetric spin).", "", pd.DataFrame({
        "Variant": d["variant"], "Scheme": d["scheme"], "Map": d["map"], "Draws": d["n_draws"], "R²cv": d["r2"],
        "P (sym)": d["p_spin_sym"], "Null 95th pct": d["null95"], "q BH (sym)": d["q_spin_sym"]}))

    T["S9"] = ("Supplementary Table 9. Pre-specified hypotheses (H1–H5 from the analysis plan; H6a–H6d written down "
               "before the generative-null analysis) and verdicts under the pre-specified decision rules.", "",
               pd.DataFrame(TABLE9, columns=["Hypothesis", "Pre-specified analysis", "Additional analyses"]))

    src = json.loads((RES / "v3_target_se_sources.json").read_text())
    rows = []
    for m_, v in src.items():
        s0, med, imp = [x.strip() for x in v.split(";")]
        rows.append({"Map": m_, "Source": s0.replace("p-values", "P values"),
                     "Median SE": float(med.replace("median SE ", "")),
                     "SE implied by reported P values": float(imp.replace("p-implied ", "")) if "implied" in imp else None})
    T["S10"] = ("Supplementary Table 10. Source of the region-wise standard errors of each ENIGMA map.", "",
                pd.DataFrame(rows))

    # one value per homologous pair (34 regions) and the Glasser parcellation (360 regions)
    VER = {"avg": "homologue average", "left": "left hemisphere"}
    half_note = ("Homologue average: every map (targets and annotations) averaged over the two homologues; left "
                 "hemisphere: the left hemisphere only. Spin: rotation of the left fsaverage5 sphere with Hungarian "
                 "re-assignment (no right hemisphere to mirror).")
    if (RES / "v3_stats_half_biv.csv").exists():
        d = pd.read_csv(RES / "v3_stats_half_biv.csv")
        T["S11a"] = ("Supplementary Table 11a. False-positive rates of tests of map-to-map correlations with one value "
                     "per homologous pair (34 regions; the first 500 maps of the main simulation, λ = 77 mm, ρ = 0.9, "
                     "against the 28 annotations; 1,000 draws).", half_note + " Map-level bootstrap confidence "
                     "intervals; t-tests against 5%, Holm-corrected.", pd.DataFrame({
                         "Version": d["how"].map(VER), "Null": d["null"], "Maps": d["n_maps"], "FPR": d["fpr"],
                         "CI low": d["lo"], "CI high": d["hi"], "Holm P vs 5%": d["p_holm"],
                         "Classification": d["bradley"]}))
        d = pd.read_csv(RES / "v3_stats_half_biv_real.csv")
        T["S11b"] = ("Supplementary Table 11b. Number of significant ENIGMA disorder–annotation correlations (of 336) "
                     "with one value per homologous pair (10,000 draws).", half_note, pd.DataFrame({
                         "Version": d["how"].map(VER), "Null": d["null"], "P < 0.05": d["n_p05"],
                         "BH q < 0.05": d["n_q05"], "Holm < 0.05": d["n_holm05"]}))
        d = pd.read_csv(RES / "v3_stats_half_cv_fpr.csv")
        T["S11c"] = ("Supplementary Table 11c. Cross-validation false-positive rates with one value per homologous pair "
                     "(500 null maps of the main simulation at λ = 77 mm; 500 draws).", half_note + " Schemes: R (5 "
                     "random partitions), S (10 spatial blocks) and DD (distance-dependent, within the hemisphere). "
                     "Wilson 95% confidence intervals; exact binomial tests against 5%, Holm-corrected within each "
                     "version.", pd.DataFrame({
                         "Version": d["how"].map(VER), "Model": d["model"].map(LAB), "Scheme": d["scheme"],
                         "Null": d["null"], "Rejections": d["k"], "Maps": d["n"], "FPR": d["fpr"], "CI low": d["lo"],
                         "CI high": d["hi"], "Mean statistic": d["mean_stat"], "Mean null statistic": d["mean_null"],
                         "Holm P": d["p_holm"],
                         "Classification": d["bradley"]}))
        d = pd.read_csv(RES / "v3_stats_half_power.csv")
        T["S11d"] = ("Supplementary Table 11d. Positive control with one value per homologous pair: the maps of the "
                     "positive control (150 per signal share) and the 500 signal-free maps, tested with the spin.",
                     half_note + " AUC of the z-statistic relative to each map's null (stratified bootstrap 95% "
                     "confidence intervals); size-adjusted power: sensitivity at the 95th percentile of the z-statistics "
                     "of the signal-free maps.", pd.DataFrame({
                         "Version": d["how"].map(VER), "Model": d["model"].map(LAB), "Scheme": d["scheme"], "s": d["s"],
                         "AUC": d["auc"], "CI low": d["lo"], "CI high": d["hi"],
                         "Power, size-adjusted": d["power_size_adjusted"], "Power, P < 0.05": d["power_nominal"],
                         "FPR, P < 0.05": d["fpr_nominal"], "Median skill": d["skill_median"]}))
        d = pd.read_csv(RES / "v3_stats_half_real.csv")
        T["S11e"] = ("Supplementary Table 11e. Number of ENIGMA maps significantly predicted (of 12) with one value per "
                     "homologous pair (2,000 draws).", half_note + " R: 10 random partitions. Statistic: R²cv (R, S) "
                     "or mean test-set r (DD).", pd.DataFrame({
                         "Version": d["how"].map(VER), "Model": d["model"].map(LAB), "Scheme": d["scheme"],
                         "Null": d["null"], "Median statistic": d["median_stat"], "Maps P < 0.05": d["n_p05"],
                         "Maps BH q < 0.05": d["n_bh05"], "Maps Holm < 0.05": d["n_holm05"],
                         "Maps (BH)": d["maps_bh"].fillna("")}))
    if (RES / "v3_stats_glasser_biv.csv").exists():
        d = pd.read_csv(RES / "v3_stats_glasser_biv.csv")
        T["S12a"] = ("Supplementary Table 12a. Tests of map-to-map correlations on the Glasser parcellation (360 "
                     "regions, 180 homologous pairs; 500 simulated null maps, bilaterally coupled or not, against 28 "
                     "simulated annotation maps, λ = 77 mm, ρ = 0.9; 1,000 draws).", "Map-level bootstrap confidence "
                     "intervals; t-tests against "
                     "5%, Holm-corrected. Bilateral r of null maps: median over the 28 annotation maps of the median "
                     "bilateral correlation of their null versions (observed: median "
                     f"{d['bilateral_r_maps'].iloc[0]:.2f}).", pd.DataFrame({
                         "Condition": d["cond"].map(_cond), "Null": d["null"].map(LAB), "Maps": d["n_maps"],
                         "FPR": d["fpr"], "CI low": d["lo"],
                         "CI high": d["hi"], "Holm P vs 5%": d["p_holm"], "Classification": d["bradley"],
                         "Bilateral r of null maps": d["bilateral_r_null"]}))
    if (RES / "v3_stats_glasser_cv_fpr.csv").exists():
        d = pd.read_csv(RES / "v3_stats_glasser_cv_fpr.csv")
        T["S12b"] = ("Supplementary Table 12b. Cross-validation false-positive rates on the Glasser parcellation (300 "
                     "null maps predicted from 28 simulated annotation maps; 500 draws).", "Schemes: R and H (one random "
                     "10-fold partition each), S, SH and Sw (10 blocks). Mean statistic: R²cv of the signal-free maps; mean "
                     "null statistic: mean of their null distributions (equal if the null reproduces the dependence that "
                     "the scheme lets leak). Wilson 95% confidence intervals; exact binomial tests against 5%, "
                     "Holm-corrected.", pd.DataFrame({
                         "Model": d["model"].map(LAB), "Scheme": d["scheme"], "Null": d["null"].map(LAB),
                         "Rejections": d["k"], "Maps": d["n"], "FPR": d["fpr"], "CI low": d["lo"], "CI high": d["hi"],
                         "Mean statistic": d["mean_stat"], "Mean null statistic": d["mean_null"], "Holm P": d["p_holm"],
                         "Classification": d["bradley"]}))
    for k_, (cap, note, t) in T.items():
        T[k_] = (cap, note, t.reset_index(drop=True))
    return T


def supp_data():
    """Supplementary Data 1 (Excel): a contents sheet and one sheet per Supplementary Table (caption, note, table)."""
    from openpyxl.styles import Alignment, Font
    T = supp_tables()
    pcols = re.compile(r"(^P\b|\bP$|P \(|P vs|Holm P|q BH|Wilcoxon P|Binomial P|Exact P|Bootstrap P|Nemenyi P|"
                       r"Friedman P|P shuffle|P conventional|P symmetric|P \(count)")
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "Supplementary_Data_1.xlsx"
    with pd.ExcelWriter(f, engine="openpyxl") as w:
        pd.DataFrame({"Sheet": list(T), "Table": [v[0] for v in T.values()], "Rows": [len(v[2]) for v in T.values()]}
                     ).to_excel(w, sheet_name="Contents", index=False)
        for k_, (cap, note, t) in T.items():
            t.to_excel(w, sheet_name=k_, index=False, startrow=3)
            ws = w.sheets[k_]
            ws["A1"], ws["A2"] = cap, note
            ws["A1"].font = Font(bold=True)
            ws.freeze_panes = "A5"
            for j, col in enumerate(t.columns, start=1):
                letter = ws.cell(row=4, column=j).column_letter
                ws.column_dimensions[letter].width = max(9, min(60, len(str(col)) + 2,
                                                                 max((len(str(x)) for x in t[col]), default=0) + 2))
                ws.cell(row=4, column=j).alignment = Alignment(wrap_text=True, vertical="top")
                if pd.api.types.is_float_dtype(t[col]):
                    fmt = "[=0]0.000;[<0.001]0.0E+00;0.000" if pcols.search(col) else "0.000"
                    for i in range(len(t)):
                        ws.cell(row=5 + i, column=j).number_format = fmt
        w.sheets["Contents"].column_dimensions["B"].width = 120
    print("wrote", f, len(T), "tables")


STEPS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "supp": supp,
         "prespecified": prespecified, "source_data": source_data, "supp_data": supp_data}

if __name__ == "__main__":
    for step in (sys.argv[1:] or list(STEPS)):
        if step not in STEPS:
            raise SystemExit(__doc__)
        STEPS[step]()
        print(step, "done", flush=True)
