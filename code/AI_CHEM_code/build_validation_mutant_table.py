#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="CSV with at least variant_id,predicted_value")
    ap.add_argument("--labels_csv", default="code/AI_CHEM_code/rubisco_datasets_merged.csv")
    ap.add_argument("--out_csv", default="results/validation_table_predictions.csv")
    ap.add_argument("--target", default="dms_enrichment_mean")
    ap.add_argument("--mutation_col", default="mutation")
    ap.add_argument("--variant_col", default="variant_id")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred_csv)
    labels = pd.read_csv(args.labels_csv)

    if args.variant_col not in pred.columns:
        raise ValueError(f"{args.variant_col} not in prediction file")
    if "predicted_value" not in pred.columns:
        raise ValueError("prediction file must include 'predicted_value'")

    d = labels[[args.variant_col, args.target] + ([args.mutation_col] if args.mutation_col in labels.columns else [])].copy()
    d = d.merge(pred[[args.variant_col, "predicted_value"]], on=args.variant_col, how="inner")
    d["observed_value"] = pd.to_numeric(d[args.target], errors="coerce")
    d = d[np.isfinite(d["observed_value"])].copy()

    d["predicted_rank"] = d["predicted_value"].rank(ascending=False, method="min").astype(int)
    n = len(d)
    k5 = max(1, int(np.ceil(0.05 * n)))
    d["in_predicted_top5pct"] = d["predicted_rank"] <= k5
    d["in_predicted_top10pct"] = d["predicted_rank"] <= max(1, int(np.ceil(0.10 * n)))

    true_rank = d["observed_value"].rank(ascending=False, method="min")
    d["in_true_top5pct"] = true_rank <= k5
    d["hit_or_miss"] = np.where(d["in_predicted_top5pct"] == d["in_true_top5pct"], "hit", "miss")

    cols = [args.variant_col]
    if args.mutation_col in d.columns:
        cols.append(args.mutation_col)
    cols += ["predicted_rank", "predicted_value", "observed_value", "in_predicted_top5pct", "in_predicted_top10pct", "in_true_top5pct", "hit_or_miss"]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    d[cols].sort_values("predicted_rank").to_csv(args.out_csv, index=False)

    misses = d[d["hit_or_miss"] == "miss"].copy().sort_values("predicted_rank").head(20)
    miss_path = os.path.splitext(args.out_csv)[0] + "_top_misses.csv"
    misses[cols].to_csv(miss_path, index=False)
    print("Wrote", args.out_csv)
    print("Wrote", miss_path)


if __name__ == "__main__":
    main()
