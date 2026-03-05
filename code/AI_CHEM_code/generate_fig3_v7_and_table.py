#!/usr/bin/env python3
"""Generate Figure 3 v7 (clean, no brackets) and supplementary pairwise table."""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

REPO = "/mmfs1/scratch/jacks.local/jyoung67391/Rubisco_Protein_Language_Model_Active_Learning"
FIG_DIR = os.path.join(REPO, "results/figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --- Manuscript palette ---
C_XGB    = "#5B9BD5"
C_TABPFN = "#E8903A"
C_XGB_DARK = "#3A7AB5"
C_TAB_DARK = "#C86A1A"

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


# ── Data loading (identical to v6) ──────────────────────────────────────

def load_esm2_units():
    df = pd.read_csv(os.path.join(REPO, "results/results_pubready_xgb_tabpfn/runs_raw.csv"))
    df = df[(df["pca_dim"] == 128) & ((df["tabpfn_cap"] == 0.0) | df["tabpfn_cap"].isna())]
    units = df.groupby(["target", "split", "split_id", "model"])["spearman"].mean().reset_index()
    units["embedding"] = "ESM2"
    return units


def load_prott5_units():
    df = pd.read_csv(os.path.join(REPO, "results/results_prott5_comparison/prott5_runs_raw.csv"))
    units = df.groupby(["target", "split", "split_id", "model"])["spearman"].mean().reset_index()
    units["embedding"] = "ProtT5"
    return units


def get_vals(df, target, split, model):
    sub = df[(df["target"] == target) & (df["split"] == split) & (df["model"] == model)]
    return sub.sort_values("split_id")["spearman"].values


def jitter(n, width=0.04, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


GROUPS = [
    ("dms_enrichment_mean",   "within_position", "Enrichment\n(within-pos)"),
    ("dms_enrichment_mean",   "pos_holdout",     "Enrichment\n(pos-holdout)"),
    ("dms_KmCO2_logfit",      "within_position", "log(Km$_{CO_2}$)\n(within-pos)"),
    ("dms_KmCO2_logfit",      "pos_holdout",     "log(Km$_{CO_2}$)\n(pos-holdout)"),
    ("dms_VmaxRatio_logfit",  "within_position", "log(Vmax ratio)\n(within-pos)"),
    ("dms_VmaxRatio_logfit",  "pos_holdout",     "log(Vmax ratio)\n(pos-holdout)"),
]

CONDITIONS = [
    ("ESM2",   "xgb",    C_XGB,    False, "black",     "o"),
    ("ESM2",   "tabpfn", C_TABPFN, False, "black",     "o"),
    ("ProtT5", "xgb",    C_XGB,    True,  C_XGB_DARK,  "D"),
    ("ProtT5", "tabpfn", C_TABPFN, True,  C_TAB_DARK,  "D"),
]


# ── Task 1: Clean figure (v7) ──────────────────────────────────────────

def fig3_clean():
    esm2 = load_esm2_units()
    prott5 = load_prott5_units()

    fig, ax = plt.subplots(figsize=(14, 6.0))

    strip_width = 0.22
    offsets = [-1.5, -0.5, 0.5, 1.5]
    group_centers = np.arange(len(GROUPS))

    DOT_SIZE = 90
    CROSSBAR_HW = 0.07

    for gi, (target, split, label) in enumerate(GROUPS):
        for ci, (emb, model, color, is_prott5, edgecol, marker) in enumerate(CONDITIONS):
            src = esm2 if emb == "ESM2" else prott5
            vals = get_vals(src, target, split, model)
            center = group_centers[gi] + offsets[ci] * strip_width

            if len(vals) == 0:
                continue

            jit = jitter(len(vals), width=0.035, seed=gi * 100 + ci)
            alpha = 0.85 if is_prott5 else 0.7
            ax.scatter(
                center + jit, vals,
                s=DOT_SIZE, c=color, marker=marker,
                edgecolors=edgecol, linewidths=0.8,
                alpha=alpha, zorder=4,
            )

            med = np.median(vals)
            ax.plot(
                [center - CROSSBAR_HW, center + CROSSBAR_HW],
                [med, med],
                color="black", lw=2.5, zorder=5, solid_capstyle="round",
            )

    # No brackets or annotations — clean figure

    # --- Axis labels and title ---
    ax.set_xticks(group_centers)
    ax.set_xticklabels([g[2] for g in GROUPS], fontsize=10)
    ax.set_ylabel("Spearman $\\rho$", fontsize=12)
    ax.set_title(
        "Model and Embedding Comparison Across DMS Targets",
        fontsize=12, fontweight="bold", pad=12,
    )

    ax.text(
        0.5, 1.01,
        "Each point is one split replicate (within-position n=3, pos-holdout n=5); "
        "crossbars show medians. See Supplementary Table SX for pairwise comparisons.",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8, fontstyle="italic", color="0.4",
    )

    # --- Legend ---
    legend_elements = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=C_XGB,
                       markeredgecolor="black", markersize=9,
                       label="ESM2 + XGBoost"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=C_TABPFN,
                       markeredgecolor="black", markersize=9,
                       label="ESM2 + TabPFN"),
        mlines.Line2D([], [], marker="D", color="w", markerfacecolor=C_XGB,
                       markeredgecolor=C_XGB_DARK, markersize=8,
                       label="ProtT5 + XGBoost"),
        mlines.Line2D([], [], marker="D", color="w", markerfacecolor=C_TABPFN,
                       markeredgecolor=C_TAB_DARK, markersize=8,
                       label="ProtT5 + TabPFN"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              frameon=True, framealpha=0.9, edgecolor="none")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.02, ymax + 0.05)

    path = os.path.join(FIG_DIR, "fig3_updated_with_prott5_v7.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path} ({os.path.getsize(path):,} bytes)")


# ── Task 2: Supplementary pairwise comparison table ─────────────────────

def cohens_d_paired(a, b):
    """Cohen's d for paired samples: mean(diff) / sd(diff)."""
    diff = a - b
    sd = diff.std(ddof=1)
    if sd == 0:
        return np.nan
    return diff.mean() / sd


def supplementary_table():
    esm2 = load_esm2_units()
    prott5 = load_prott5_units()

    comparisons = [
        ("ESM2+XGB vs ESM2+TabPFN",     "ESM2", "xgb",    "ESM2", "tabpfn"),
        ("ProtT5+XGB vs ProtT5+TabPFN", "ProtT5", "xgb",  "ProtT5", "tabpfn"),
        ("ESM2+XGB vs ProtT5+XGB",      "ESM2", "xgb",    "ProtT5", "xgb"),
        ("ESM2+TabPFN vs ProtT5+TabPFN", "ESM2", "tabpfn", "ProtT5", "tabpfn"),
    ]

    rows = []
    for target, split, _ in GROUPS:
        for comp_name, emb_a, mod_a, emb_b, mod_b in comparisons:
            src_a = esm2 if emb_a == "ESM2" else prott5
            src_b = esm2 if emb_b == "ESM2" else prott5

            vals_a = get_vals(src_a, target, split, mod_a)
            vals_b = get_vals(src_b, target, split, mod_b)

            n = min(len(vals_a), len(vals_b))
            if n < 2:
                continue

            a = vals_a[:n]
            b = vals_b[:n]

            t_stat, p_val = stats.ttest_rel(a, b)
            d = cohens_d_paired(a, b)

            rows.append({
                "target": target,
                "split": split,
                "comparison": comp_name,
                "n_paired": n,
                "mean_A": round(a.mean(), 4),
                "mean_B": round(b.mean(), 4),
                "mean_delta": round((a - b).mean(), 4),
                "paired_t_statistic": round(t_stat, 4),
                "paired_t_pvalue": round(p_val, 6),
                "cohens_d": round(d, 4),
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(REPO, "results/supplementary_table_pairwise_comparisons.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"\n{df.to_string(index=False)}\n")
    return df


if __name__ == "__main__":
    fig3_clean()
    supplementary_table()
