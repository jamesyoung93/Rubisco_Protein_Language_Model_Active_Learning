#!/usr/bin/env python3
"""Generate updated Figure 3 and three supplementary figures."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

REPO = "/mmfs1/scratch/jacks.local/jyoung67391/Rubisco_Protein_Language_Model_Active_Learning"
FIG_DIR = os.path.join(REPO, "results/figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --- Manuscript palette ---
C_XGB      = "#5B9BD5"  # steel blue
C_TABPFN   = "#E8903A"  # warm orange
C_SVR      = "#6AAF6A"  # green
C_MLP      = "#C26B6B"  # muted red
C_XGB_LT   = "#A8C8E8"  # lighter steel blue
C_TAB_LT   = "#F5C99A"  # lighter orange
C_DEPTH    = "#6AAF6A"  # green (for depth split)

# Publication defaults
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "savefig.facecolor": "white",
})


def load_esm2_units():
    """Load ESM2 runs_raw, filter to fixed config, aggregate to unit level."""
    df = pd.read_csv(os.path.join(REPO, "results/results_pubready_xgb_tabpfn/runs_raw.csv"))
    # Fixed config: PCA=128, tabpfn_cap=0 or NaN (xgb)
    df = df[(df["pca_dim"] == 128) & ((df["tabpfn_cap"] == 0.0) | df["tabpfn_cap"].isna())]
    # Aggregate model_seeds within each (target, split, split_id, model) → one value per unit
    units = df.groupby(["target", "split", "split_id", "model"])["spearman"].mean().reset_index()
    units["embedding"] = "ESM2"
    return units


def load_prott5_units():
    """Load ProtT5 runs_raw, aggregate to unit level."""
    df = pd.read_csv(os.path.join(REPO, "results/results_prott5_comparison/prott5_runs_raw.csv"))
    units = df.groupby(["target", "split", "split_id", "model"])["spearman"].mean().reset_index()
    units["embedding"] = "ProtT5"
    return units


def pval_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


# ===================================================================
# TASK 1: Updated Figure 3 — Boxplots with ProtT5
# ===================================================================
def fig3_updated():
    esm2 = load_esm2_units()
    prott5 = load_prott5_units()

    # p-values from paired_tests.csv (ESM2 TabPFN vs XGB)
    pt = pd.read_csv(os.path.join(REPO, "results/results_pubready_xgb_tabpfn/paired_tests.csv"))
    pvals = {}
    for _, r in pt.iterrows():
        pvals[(r["target"], r["split"])] = r["paired_t_pvalue"]

    # 6 groups: 3 targets × 2 splits
    groups = [
        ("dms_enrichment_mean", "within_position", "Enrichment\n(within-pos)"),
        ("dms_enrichment_mean", "pos_holdout", "Enrichment\n(pos-holdout)"),
        ("dms_KmCO2_logfit", "within_position", "log(Km$_{CO_2}$)\n(within-pos)"),
        ("dms_KmCO2_logfit", "pos_holdout", "log(Km$_{CO_2}$)\n(pos-holdout)"),
        ("dms_VmaxRatio_logfit", "within_position", "log(Vmax ratio)\n(within-pos)"),
        ("dms_VmaxRatio_logfit", "pos_holdout", "log(Vmax ratio)\n(pos-holdout)"),
    ]

    # 4 conditions per group
    conditions = [
        ("ESM2", "xgb",    "ESM2 + XGB",    C_XGB,    None,  0.5),
        ("ESM2", "tabpfn", "ESM2 + TabPFN", C_TABPFN, None,  0.5),
        ("ProtT5", "xgb",    "ProtT5 + XGB",    C_XGB,    "///", 0.85),
        ("ProtT5", "tabpfn", "ProtT5 + TabPFN", C_TABPFN, "///", 0.85),
    ]

    fig, ax = plt.subplots(figsize=(13, 5.0))

    box_width = 0.17
    offsets = [-1.5, -0.5, 0.5, 1.5]
    group_centers = np.arange(len(groups))

    all_bp = []
    for gi, (target, split, label) in enumerate(groups):
        for ci, (emb, model, cond_label, color, hatch, alpha) in enumerate(conditions):
            if emb == "ESM2":
                sub = esm2[(esm2["target"] == target) & (esm2["split"] == split) &
                           (esm2["model"] == model)]
            else:
                sub = prott5[(prott5["target"] == target) & (prott5["split"] == split) &
                             (prott5["model"] == model)]

            vals = sub["spearman"].values
            if len(vals) == 0:
                continue

            pos = group_centers[gi] + offsets[ci] * box_width
            bp = ax.boxplot(
                [vals], positions=[pos], widths=box_width * 0.85,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=1.2),
                whiskerprops=dict(color="black", lw=0.8),
                capprops=dict(color="black", lw=0.8),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(alpha)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)
                if hatch:
                    patch.set_hatch(hatch)
            all_bp.append(bp)

        # p-value annotation (ESM2 TabPFN vs XGB only)
        key = (target, split)
        if key in pvals:
            p = pvals[key]
            stars = pval_stars(p)
            # Position bracket above ESM2 XGB and ESM2 TabPFN boxes
            x1 = group_centers[gi] + offsets[0] * box_width
            x2 = group_centers[gi] + offsets[1] * box_width

            esm_xgb_vals = esm2[(esm2["target"]==target) & (esm2["split"]==split) &
                                (esm2["model"]=="xgb")]["spearman"].values
            esm_tab_vals = esm2[(esm2["target"]==target) & (esm2["split"]==split) &
                                (esm2["model"]=="tabpfn")]["spearman"].values
            if len(esm_xgb_vals) > 0 and len(esm_tab_vals) > 0:
                y_top = max(esm_xgb_vals.max(), esm_tab_vals.max())
                # Also check ProtT5 boxes to avoid overlap
                pt_xgb = prott5[(prott5["target"]==target) & (prott5["split"]==split) &
                                (prott5["model"]=="xgb")]["spearman"].values
                pt_tab = prott5[(prott5["target"]==target) & (prott5["split"]==split) &
                                (prott5["model"]=="tabpfn")]["spearman"].values
                for v in [pt_xgb, pt_tab]:
                    if len(v) > 0:
                        y_top = max(y_top, v.max())

                y_bar = y_top + 0.015
                ax.plot([x1, x1, x2, x2], [y_bar - 0.005, y_bar, y_bar, y_bar - 0.005],
                        color="black", lw=0.8)
                ax.text((x1 + x2) / 2, y_bar + 0.003, stars,
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(group_centers)
    ax.set_xticklabels([g[2] for g in groups], fontsize=10)
    ax.set_ylabel("Spearman $\\rho$", fontsize=12)
    ax.set_title(
        "TabPFN-2.5 Outperforms XGBoost Across Targets;\n"
        "ProtT5 Embeddings Provide Consistent Improvement",
        fontsize=12, fontweight="bold", pad=12,
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_XGB, alpha=0.5, edgecolor="black", lw=0.8,
                       label="ESM2 + XGBoost"),
        mpatches.Patch(facecolor=C_TABPFN, alpha=0.5, edgecolor="black", lw=0.8,
                       label="ESM2 + TabPFN"),
        mpatches.Patch(facecolor=C_XGB, alpha=0.85, edgecolor="black", lw=0.8,
                       hatch="///", label="ProtT5 + XGBoost"),
        mpatches.Patch(facecolor=C_TABPFN, alpha=0.85, edgecolor="black", lw=0.8,
                       hatch="///", label="ProtT5 + TabPFN"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              frameon=True, framealpha=0.9, edgecolor="none")

    # Adjust y-axis to give breathing room
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    ax.set_ylim(ymin - 0.02, ymax + 0.03)

    path = os.path.join(FIG_DIR, "fig3_updated_with_prott5.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path} ({os.path.getsize(path):,} bytes)")


# ===================================================================
# SUPP FIGURE A: Model comparison dot plot (TabPFN + XGB only)
# ===================================================================
def suppfig_a_model_comparison():
    v1 = pd.read_csv(os.path.join(REPO, "_orginal_v1/results/results_pubready_xgb_tabpfn/summary_pub.csv"))
    v1f = v1[v1["config_label"] == "fixed"].copy()

    splits = [
        ("within_position", "dms_enrichment_mean", "DMS", "DMS within-position"),
        ("pos_holdout", "dms_enrichment_mean", "DMS", "DMS position-holdout"),
        ("depth_holdout", "hoff_delta_O2_minus_N2", "HOFF_delta_direct", "HOFF depth-holdout"),
    ]

    fig, ax = plt.subplots(figsize=(6, 3.0))
    y_map = {s[0]: i for i, s in enumerate(reversed(splits))}

    for split_key, target, task, label in splits:
        y = y_map[split_key]
        vals = {}
        for model, color, marker in [("tabpfn", C_TABPFN, "D"), ("xgb", C_XGB, "s")]:
            if "hoff" in target:
                match = v1f[(v1f["target"] == target) & (v1f["split"] == split_key) &
                            (v1f["model"] == model) & (v1f["task_name"] == task)]
            else:
                match = v1f[(v1f["target"] == target) & (v1f["split"] == split_key) &
                            (v1f["model"] == model)]
            if len(match) > 0:
                v = match.iloc[0]["spearman_mean"]
                vals[model] = v
                ax.scatter(v, y, c=color, marker=marker, s=100,
                           edgecolors="white", linewidths=0.6, zorder=3)

        if "tabpfn" in vals and "xgb" in vals:
            ax.plot([vals["xgb"], vals["tabpfn"]], [y, y],
                    color="#CCCCCC", lw=1.5, zorder=1)

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels([s[3] for s in reversed(splits)], fontsize=11)
    ax.set_xlabel("Spearman $\\rho$", fontsize=12)
    ax.set_xlim(0.30, 0.96)
    ax.set_ylim(-0.5, 2.5)

    handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_TABPFN,
               markersize=10, label="TabPFN"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_XGB,
               markersize=10, label="XGBoost"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=10,
              frameon=True, framealpha=0.9, edgecolor="none")

    path = os.path.join(FIG_DIR, "suppfig_model_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path} ({os.path.getsize(path):,} bytes)")


# ===================================================================
# SUPP FIGURE B: Balance experiment bar chart
# ===================================================================
def suppfig_b_balance():
    df = pd.read_csv(os.path.join(REPO, "results/balance_experiment.csv"))
    stats = df.groupby(["model", "condition"])["spearman"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(4.8, 4.2))

    bar_width = 0.30
    x_pos = np.array([0, 1])
    group_labels = ["XGBoost", "TabPFN"]

    bar_specs = [
        ("original",          {"xgb": C_XGB, "tabpfn": C_TABPFN}, "Original"),
        ("quantile_balanced", {"xgb": C_XGB_LT, "tabpfn": C_TAB_LT}, "Quantile-balanced"),
    ]

    for i, (condition, colors, label) in enumerate(bar_specs):
        means, sds = [], []
        bar_colors = []
        for model in ["xgb", "tabpfn"]:
            row = stats[(stats["model"] == model) & (stats["condition"] == condition)]
            means.append(row["mean"].values[0])
            sds.append(row["std"].values[0])
            bar_colors.append(colors[model])

        positions = x_pos + (i - 0.5) * bar_width
        bars = ax.bar(positions, means, bar_width,
                       yerr=sds, capsize=4,
                       color=bar_colors, edgecolor="white", linewidth=0.5,
                       zorder=3, error_kw=dict(lw=1.0, capthick=1.0))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_labels, fontsize=12)
    ax.set_ylabel("Spearman $\\rho$", fontsize=12)
    ax.set_ylim(0.80, 0.925)

    # Break symbol at bottom of y-axis
    d = 0.015
    kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, lw=1)
    ax.plot((-d, +d), (-d, +d), **kwargs)
    ax.plot((-d + 0.01, +d + 0.01), (-d, +d), **kwargs)

    # Gap annotations
    xgb_orig = stats[(stats["model"]=="xgb") & (stats["condition"]=="original")]["mean"].values[0]
    tab_orig = stats[(stats["model"]=="tabpfn") & (stats["condition"]=="original")]["mean"].values[0]
    xgb_bal  = stats[(stats["model"]=="xgb") & (stats["condition"]=="quantile_balanced")]["mean"].values[0]
    tab_bal  = stats[(stats["model"]=="tabpfn") & (stats["condition"]=="quantile_balanced")]["mean"].values[0]
    gap_orig = tab_orig - xgb_orig
    gap_bal  = tab_bal - xgb_bal

    y1 = 0.907
    ax.annotate("", xy=(0 - 0.5*bar_width, y1),
                xytext=(1 - 0.5*bar_width, y1),
                arrowprops=dict(arrowstyle="<->", color="black", lw=0.9))
    ax.text(0.5, y1 + 0.002, f"$\\Delta$ = {gap_orig:.3f}",
            ha="center", va="bottom", fontsize=9.5)

    y2 = 0.916
    ax.annotate("", xy=(0 + 0.5*bar_width, y2),
                xytext=(1 + 0.5*bar_width, y2),
                arrowprops=dict(arrowstyle="<->", color="black", lw=0.9))
    ax.text(0.5, y2 + 0.002, f"$\\Delta$ = {gap_bal:.3f}",
            ha="center", va="bottom", fontsize=9.5)

    # Legend: use original colors as representatives
    legend_elements = [
        mpatches.Patch(facecolor=C_XGB, edgecolor="white", label="Original"),
        mpatches.Patch(facecolor=C_XGB_LT, edgecolor="white", label="Quantile-balanced"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10,
              frameon=True, framealpha=0.9, edgecolor="none")

    path = os.path.join(FIG_DIR, "suppfig_balance_experiment.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path} ({os.path.getsize(path):,} bytes)")


# ===================================================================
# SUPP FIGURE C: ProtT5 vs ESM2 scatter
# ===================================================================
def suppfig_c_prott5_scatter():
    prott5 = pd.read_csv(os.path.join(REPO, "results/results_prott5_comparison/prott5_summary.csv"))
    v1 = pd.read_csv(os.path.join(REPO, "_orginal_v1/results/results_pubready_xgb_tabpfn/summary_pub.csv"))
    v1f = v1[v1["config_label"] == "fixed"].copy()

    points = []
    for _, pr in prott5.iterrows():
        target, split, model = pr["target"], pr["split"], pr["model"]
        if "hoff" in target:
            esm = v1f[(v1f["target"] == target) & (v1f["split"] == split) &
                       (v1f["model"] == model) & (v1f["task_name"] == "HOFF_delta_direct")]
        else:
            esm = v1f[(v1f["target"] == target) & (v1f["split"] == split) &
                       (v1f["model"] == model)]
        if len(esm) == 0:
            continue
        points.append({
            "esm2": esm.iloc[0]["spearman_mean"],
            "prott5": pr["spearman_mean"],
            "split": split,
            "model": model,
        })
    pts = pd.DataFrame(points)

    split_colors = {
        "within_position": C_XGB,
        "pos_holdout": C_TABPFN,
        "depth_holdout": C_DEPTH,
    }
    model_markers = {"xgb": "s", "tabpfn": "o"}

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    lo, hi = 0.1, 1.0
    ax.plot([lo, hi], [lo, hi], ls="--", color="#CCCCCC", lw=1, zorder=1)

    for _, r in pts.iterrows():
        ax.scatter(r["esm2"], r["prott5"],
                   c=split_colors[r["split"]],
                   marker=model_markers[r["model"]],
                   s=70, edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_xlabel("ESM2 Spearman $\\rho$", fontsize=12)
    ax.set_ylabel("ProtT5 Spearman $\\rho$", fontsize=12)
    ax.set_xlim(0.12, 0.96)
    ax.set_ylim(0.12, 0.96)
    ax.set_aspect("equal")

    mean_delta = (pts["prott5"] - pts["esm2"]).mean()
    ax.annotate(f"Mean $\\Delta$ = {mean_delta:+.3f}",
                xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=10, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.5))

    split_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_XGB,
               markersize=8, label="Within-position"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_TABPFN,
               markersize=8, label="Position-holdout"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_DEPTH,
               markersize=8, label="Depth-holdout"),
    ]
    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=8, label="TabPFN"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black",
               markersize=8, label="XGBoost"),
    ]
    ax.legend(handles=split_handles + model_handles,
              loc="lower right", fontsize=9, frameon=True,
              framealpha=0.9, edgecolor="none",
              handletextpad=0.3, borderpad=0.4)

    path = os.path.join(FIG_DIR, "suppfig_prott5_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path} ({os.path.getsize(path):,} bytes)")


# ===================================================================
if __name__ == "__main__":
    print("=== TASK 1: Updated Figure 3 ===")
    fig3_updated()
    print("\n=== TASK 2: Supplementary Figures ===")
    suppfig_a_model_comparison()
    suppfig_b_balance()
    suppfig_c_prott5_scatter()
    print("\nAll figures done.")
