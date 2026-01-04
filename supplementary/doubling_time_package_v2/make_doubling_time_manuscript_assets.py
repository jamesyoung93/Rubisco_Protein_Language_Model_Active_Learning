#!/usr/bin/env python3
"""
make_doubling_time_manuscript_assets.py

Turn doubling_time_outputs/ into manuscript-ready assets:
- Table S1 (best-case per strain) in CSV/LaTeX/Markdown
- Figure S2: time to N doublings by strain
- Matched-condition comparison UTEX2973 vs PCC7942 from Yu et al. Table 1 (tidy CSV)
- Snippets (bullet points) with numbers filled in

Inputs expected (defaults assume you run from repo root or doubling_time_package_v2):
  doubling_time_outputs/cyano_doubling_times_compiled_best_plus_pcc11801.csv
  doubling_time_outputs/cyano_doubling_times_yu2015_srep08132_table1_long.csv
  doubling_time_outputs/yu2015_table1_doubling_time_heatmap.png  (optional)

No internet required.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def pick_best_row(group: pd.DataFrame) -> pd.Series:
    """
    Pick "best-case" row per strain:
    - minimize doubling_time_min_h
    - tie-break: prefer condition-specific over approx/range
    """
    g = group.copy()
    g["is_condition_specific"] = (g["data_type"].astype(str) == "condition-specific").astype(int)
    g = g.sort_values(
        by=["doubling_time_min_h", "is_condition_specific"],
        ascending=[True, False],
        na_position="last",
    )
    return g.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--compiled_csv",
        default="doubling_time_outputs/cyano_doubling_times_compiled_best_plus_pcc11801.csv",
        help="Compiled doubling-time dataset (CSV).",
    )
    ap.add_argument(
        "--yu_table_long_csv",
        default="doubling_time_outputs/cyano_doubling_times_yu2015_srep08132_table1_long.csv",
        help="Yu et al. 2015 Table 1 extracted to long/tidy CSV.",
    )
    ap.add_argument(
        "--yu_heatmap_png",
        default="doubling_time_outputs/yu2015_table1_doubling_time_heatmap.png",
        help="Heatmap PNG produced earlier (optional).",
    )
    ap.add_argument(
        "--out_dir",
        default="manuscript_assets/doubling_time",
        help="Where to write manuscript-ready assets.",
    )
    ap.add_argument(
        "--n_doublings",
        type=int,
        default=10,
        help="N doublings used for the throughput metric (time = N * doubling time).",
    )
    ap.add_argument(
        "--num_batches",
        type=int,
        default=0,
        help=(
            "Optional: if >0, compute a growth-only lower-bound campaign duration "
            "for num_batches measurement batches (parallel growth assumed)."
        ),
    )
    args = ap.parse_args()

    compiled_path = Path(args.compiled_csv)
    yu_long_path = Path(args.yu_table_long_csv)
    heatmap_path = Path(args.yu_heatmap_png)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------
    # Load inputs
    # ----------------------
    df = pd.read_csv(compiled_path)
    required_cols = {
        "strain",
        "temp_C",
        "co2",
        "light_umol_photons_m2_s",
        "doubling_time_min_h",
        "doubling_time_max_h",
        "data_type",
        "reference",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in compiled CSV: {sorted(missing)}")

    # Best-case per strain table
    best = df.groupby("strain", as_index=False).apply(pick_best_row).reset_index(drop=True)

    # Add throughput columns
    n = int(args.n_doublings)
    best["time_for_N_doublings_min_h"] = best["doubling_time_min_h"] * n
    best["time_for_N_doublings_max_h"] = best["doubling_time_max_h"] * n

    # Sort for readability
    best = best.sort_values(by="doubling_time_min_h", ascending=True, na_position="last")

    # Format for tables (avoid too many decimals)
    table_cols = [
        "strain",
        "doubling_time_min_h",
        "doubling_time_max_h",
        "temp_C",
        "co2",
        "light_umol_photons_m2_s",
        "data_type",
        "reference",
    ]
    table = best[table_cols].copy()

    # Numeric formatting
    for c in ["doubling_time_min_h", "doubling_time_max_h", "temp_C", "light_umol_photons_m2_s"]:
        table[c] = table[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.2f}".rstrip("0").rstrip("."))

    # ----------------------
    # Write Table S1
    # ----------------------
    table_csv = out_dir / "Table_S1_cyano_doubling_times_bestcase.csv"
    table.to_csv(table_csv, index=False)

    # LaTeX (simple, MDPI-friendly enough for revision stage)
    table_tex = out_dir / "Table_S1_cyano_doubling_times_bestcase.tex"
    latex_body = table.to_latex(index=False, escape=True)
    caption = (
        "Best-case cyanobacterial doubling times used for throughput context. "
        "Condition-specific measurements are retained where available; approximate ranges "
        "are included from curated sources (see reference column)."
    )
    label = "tab:cyano_doubling_times_bestcase"
    table_tex.write_text(
        "\\begin{table}[ht]\n\\centering\n"
        + latex_body
        + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{table}\n",
        encoding="utf-8",
    )

    # Markdown (for quick copy/paste into draft)
    table_md = out_dir / "Table_S1_cyano_doubling_times_bestcase.md"
    table_md.write_text(df_to_markdown_table(table), encoding="utf-8")

    # ----------------------
    # Copy heatmap (if present)
    # ----------------------
    if heatmap_path.exists():
        (out_dir / "Figure_S1_yu2015_table1_doubling_time_heatmap.png").write_bytes(heatmap_path.read_bytes())

    # ----------------------
    # Figure S2: Time to N doublings
    # ----------------------
    fig_df = best.copy()
    fig_df["time_min"] = fig_df["doubling_time_min_h"] * n
    fig_df["time_max"] = fig_df["doubling_time_max_h"] * n
    fig_df["err_plus"] = (fig_df["time_max"] - fig_df["time_min"]).where(~fig_df["time_max"].isna(), 0.0)

    x = np.arange(len(fig_df))
    y = fig_df["time_min"].astype(float).to_numpy()

    yerr = fig_df["err_plus"].astype(float).to_numpy()
    # Matplotlib expects symmetric yerr; use (0, +err) style by stacking
    yerr_asym = np.vstack([np.zeros_like(yerr), yerr])

    plt.figure(figsize=(10, 4.5))
    plt.bar(x, y, yerr=yerr_asym, capsize=3)
    plt.xticks(x, fig_df["strain"].astype(str).tolist(), rotation=45, ha="right")
    plt.ylabel(f"Time for {n} doublings (h)")
    plt.title(f"Growth-throughput lower bound across chassis (N={n} doublings)")
    plt.tight_layout()
    fig_path = out_dir / f"Figure_S2_time_to_{n}_doublings.png"
    plt.savefig(fig_path, dpi=600)
    plt.close()

    # ----------------------
    # Matched-condition comparison from Yu et al. Table 1 (UTEX2973 vs PCC7942)
    # ----------------------
    yu = pd.read_csv(yu_long_path)

    # Normalize strain naming if needed
    yu["strain"] = yu["strain"].astype(str).str.strip()

    key_cols = ["temp_C", "co2", "light_umol_photons_m2_s"]
    u = yu[(yu["strain"] == "UTEX2973") & (~yu["doubling_time_h"].isna())][key_cols + ["doubling_time_h"]].copy()
    p = yu[(yu["strain"] == "PCC7942") & (~yu["doubling_time_h"].isna())][key_cols + ["doubling_time_h"]].copy()

    merged = u.merge(p, on=key_cols, suffixes=("_UTEX2973", "_PCC7942"))
    if not merged.empty:
        merged["ratio_PCC7942_over_UTEX2973"] = (
            merged["doubling_time_h_PCC7942"] / merged["doubling_time_h_UTEX2973"]
        )
        merged = merged.sort_values(by="ratio_PCC7942_over_UTEX2973", ascending=False)
        merged_out = out_dir / "yu2015_utexpcc7942_matched_conditions.csv"
        merged.to_csv(merged_out, index=False)

    # ----------------------
    # Optional: growth-only campaign duration (parallel batches)
    # ----------------------
    if int(args.num_batches) > 0:
        B = int(args.num_batches)
        camp = best[["strain", "doubling_time_min_h", "doubling_time_max_h", "data_type"]].copy()
        camp["campaign_growth_time_days_min"] = (B * n * camp["doubling_time_min_h"]) / 24.0
        camp["campaign_growth_time_days_max"] = (B * n * camp["doubling_time_max_h"]) / 24.0
        camp = camp.sort_values(by="campaign_growth_time_days_min", ascending=True)

        camp_out = out_dir / f"campaign_growth_time_{B}_batches.csv"
        camp.to_csv(camp_out, index=False)

    # ----------------------
    # Snippets for Results/Discussion (numbers filled in)
    # ----------------------
    lines = []
    lines.append("## Doubling-time context bullets (auto-generated)\n")
    # UTEX best-case (from compiled)
    utex_row = best[best["strain"].astype(str) == "UTEX2973"].head(1)
    if not utex_row.empty:
        dt = float(utex_row["doubling_time_min_h"].iloc[0])
        lines.append(f"- UTEX 2973 best-case doubling time in the compiled dataset: **{dt:.2f} h**.")
        lines.append(f"  - Growth-only time for {n} doublings: **{dt*n:.1f} h**.\n")

    # Matched condition callout if available
    if not merged.empty:
        # Prefer the 38C/3%/500 condition if present
        target = merged[
            (merged["temp_C"] == 38) & (merged["co2"].astype(str) == "3%") & (merged["light_umol_photons_m2_s"] == 500)
        ]
        if target.empty:
            target = merged.head(1)
        r = target.iloc[0]
        lines.append(
            "- In Yu et al. Table 1 matched conditions (example): "
            f"UTEX2973 **{r['doubling_time_h_UTEX2973']:.1f} h** vs PCC7942 **{r['doubling_time_h_PCC7942']:.1f} h**, "
            f"ratio **{r['ratio_PCC7942_over_UTEX2973']:.2f}×** slower for PCC7942.\n"
        )

    # PCC11801 presence
    pcc11801 = best[best["strain"].astype(str).str.contains("11801", na=False)]
    if not pcc11801.empty:
        dt = float(pcc11801["doubling_time_min_h"].iloc[0])
        lines.append(f"- PCC 11801 doubling time in compiled dataset: **{dt:.2f} h** (ambient air/CO2 context).\n")

    snippet_path = out_dir / "doubling_time_snippets.md"
    snippet_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] wrote manuscript assets to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
