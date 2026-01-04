#!/usr/bin/env python3
"""Analyze compiled cyanobacterial doubling time datasets.

This script is intentionally lightweight: it assumes the CSVs produced by
build_cyano_doubling_time_dataset.py (or shipped with this package) already exist.

Common usage (from the package root):
    python analyze_cyano_doubling_times.py
or
    python analyze_cyano_doubling_times.py --compiled_csv doubling_time_outputs

If you run it with no arguments, it will prefer:
  doubling_time_outputs/cyano_doubling_times_compiled_best_plus_pcc11801.csv
falling back to:
  doubling_time_outputs/cyano_doubling_times_compiled_best.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_compiled_path(p: str) -> Path:
    """Allow passing either a file path or a directory."""
    path = Path(p).expanduser()
    if path.is_dir():
        cand1 = path / "cyano_doubling_times_compiled_best_plus_pcc11801.csv"
        cand2 = path / "cyano_doubling_times_compiled_best.csv"
        if cand1.is_file():
            return cand1
        if cand2.is_file():
            return cand2
        raise FileNotFoundError(f"No compiled CSV found in directory: {path}")
    return path


def default_compiled_path() -> Path:
    d = Path("doubling_time_outputs")
    if d.is_dir():
        return resolve_compiled_path(str(d))
    # fallback to cwd
    for cand in [
        Path("cyano_doubling_times_compiled_best_plus_pcc11801.csv"),
        Path("cyano_doubling_times_compiled_best.csv"),
    ]:
        if cand.is_file():
            return cand
    # last resort: raise
    raise FileNotFoundError(
        "Could not locate a compiled doubling time CSV.\n"
        "Expected one of:\n"
        "  doubling_time_outputs/cyano_doubling_times_compiled_best_plus_pcc11801.csv\n"
        "  doubling_time_outputs/cyano_doubling_times_compiled_best.csv\n"
        "Run build_cyano_doubling_time_dataset.py to generate outputs, or check your paths."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--compiled_csv",
        default=None,
        help="Path to compiled CSV (or a directory containing it). If omitted, uses doubling_time_outputs/ by default.",
    )
    ap.add_argument("--top_n", type=int, default=50, help="Print top N fastest entries (by mu_per_h).")
    args = ap.parse_args()

    compiled_path = resolve_compiled_path(args.compiled_csv) if args.compiled_csv else default_compiled_path()
    df = pd.read_csv(compiled_path)

    if "mu_per_h" not in df.columns:
        df["mu_per_h"] = np.log(2) / df["doubling_time_min_h"]

    df = df.sort_values("mu_per_h", ascending=False)

    keep_cols = [
        "strain",
        "temp_C",
        "co2",
        "light_umol_photons_m2_s",
        "doubling_time_min_h",
        "doubling_time_max_h",
        "mu_per_h",
        "data_type",
        "reference",
    ]

    top_n = max(1, int(args.top_n))
    print(f"# Using: {compiled_path}")
    print(df[keep_cols].head(top_n).to_string(index=False))


if __name__ == "__main__":
    main()
