#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exploratory: relate WT RbcL PLM embeddings to cyanobacterial doubling time / growth rate.

Inputs:
  --compiled_csv : your compiled doubling-time dataset (has strain, doubling_time_min_h, mu_per_h, etc.)
  --emb_npy      : npy dict from embed_rbcL_fasta_esm2.py: {"ids": [...], "emb": [n,d]}

Method:
  - Collapse compiled table to one "best-case" record per strain (min doubling_time_min_h).
  - Merge by strain name (with small alias normalization).
  - LOOCV: StandardScaler -> PCA(d) -> Ridge(alpha) fitted on training only; predict held-out.
  - Output metrics + tables + 2 figures (PCA scatter and pred-vs-actual).

Outputs:
  out_dir/
    metrics.json
    Table_embedding_vs_growth.csv
    Figure_pca_embedding_space.png
    Figure_pred_vs_actual.png
    manuscript_snippet_embedding_vs_growth.md
"""

import argparse, json, os, re
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr


ALIASES = {
    # short codes -> canonical names (helps merge if your FASTA IDs differ from compiled strain labels)
    "PCC6803": "Synechocystis sp. PCC 6803",
    "PCC7002": "Synechococcus sp. PCC 7002",
    "PCC7942": "Synechococcus elongatus PCC 7942",
    "PCC6301": "Synechococcus elongatus PCC 6301",
    "UTEX2973": "Synechococcus elongatus UTEX 2973",
    "PCC11801": "Synechococcus elongatus PCC 11801",
    # historical naming
    "Anabaena sp. PCC 7120": "Nostoc sp. PCC 7120",
    "Anabaena ATCC 29413": "Nostoc sp. ATCC 29413",
}

def canon(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    # map alias if exact match
    return ALIASES.get(s, s)

def pick_best_row(g: pd.DataFrame) -> pd.Series:
    # Best-case = min doubling_time_min_h if present else min doubling_time_max_h
    g = g.copy()
    g["doubling_time_min_h"] = pd.to_numeric(g.get("doubling_time_min_h"), errors="coerce")
    g["doubling_time_max_h"] = pd.to_numeric(g.get("doubling_time_max_h"), errors="coerce")
    if g["doubling_time_min_h"].notna().any():
        return g.loc[g["doubling_time_min_h"].idxmin()]
    return g.loc[g["doubling_time_max_h"].idxmin()]

def loocv_predict(X: np.ndarray, y: np.ndarray, pca_dim: int, ridge_alpha: float, seed: int = 0) -> np.ndarray:
    n = len(y)
    preds = np.zeros(n, dtype=float)

    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        te = np.array([i])

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        k = int(min(pca_dim, Xtr.shape[0] - 1, Xtr.shape[1]))
        if k < 1:
            preds[i] = float(np.mean(y[tr]))
            continue

        pca = PCA(n_components=k, random_state=seed, svd_solver="randomized")
        Xtr_p = pca.fit_transform(Xtr)
        Xte_p = pca.transform(Xte)

        model = Ridge(alpha=ridge_alpha, random_state=seed)
        model.fit(Xtr_p, y[tr])
        preds[i] = float(model.predict(Xte_p)[0])

    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled_csv", required=True)
    ap.add_argument("--emb_npy", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target", choices=["mu_per_h", "doubling_time_min_h"], default="mu_per_h")
    ap.add_argument("--log_target", action="store_true", help="Apply natural log to target (recommended for doubling time).")
    ap.add_argument("--pca_dim", type=int, default=5)
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.compiled_csv)
    if "strain" not in df.columns:
        raise ValueError("compiled_csv must contain a 'strain' column")

    df["strain"] = df["strain"].map(canon)
    best = df.groupby("strain", as_index=False).apply(pick_best_row).reset_index(drop=True)

    # Load embeddings dict
    emb_obj = np.load(args.emb_npy, allow_pickle=True).item()
    emb_ids = [canon(x) for x in emb_obj["ids"].tolist()]
    X_all = np.asarray(emb_obj["emb"], dtype=np.float32)

    emb_df = pd.DataFrame({"strain": emb_ids, "emb_idx": np.arange(len(emb_ids))})

    merged = best.merge(emb_df, on="strain", how="inner")
    if merged.empty:
        raise RuntimeError(
            "No strains matched between compiled_csv and emb_npy IDs.\n"
            "Inspect the FASTA headers used for embedding and the compiled strain labels."
        )

    y = pd.to_numeric(merged[args.target], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y)
    merged = merged.loc[keep].reset_index(drop=True)
    y = y[keep]

    X = X_all[merged["emb_idx"].to_numpy()]

    if args.log_target:
        y = np.log(y)

    preds = loocv_predict(X, y, pca_dim=args.pca_dim, ridge_alpha=args.ridge_alpha, seed=args.seed)

    metrics = {
        "n": int(len(y)),
        "target": args.target,
        "log_target": bool(args.log_target),
        "pca_dim": int(args.pca_dim),
        "ridge_alpha": float(args.ridge_alpha),
        "spearman": float(spearmanr(y, preds).correlation),
        "pearson": float(pearsonr(y, preds)[0]),
        "r2": float(r2_score(y, preds)),
        "mse": float(mean_squared_error(y, preds)),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    out = merged[["strain", "doubling_time_min_h", "doubling_time_max_h", "mu_per_h", "data_type", "reference"]].copy()
    out["y_true"] = y
    out["y_pred_loocv"] = preds
    out.to_csv(os.path.join(args.out_dir, "Table_embedding_vs_growth.csv"), index=False)

    # Unsupervised PCA(2) plot colored by doubling time
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k2 = int(min(2, Xs.shape[0] - 1, Xs.shape[1]))
    if k2 >= 2:
        pca2 = PCA(n_components=2, random_state=args.seed, svd_solver="randomized")
        Z = pca2.fit_transform(Xs)

        c = pd.to_numeric(merged["doubling_time_min_h"], errors="coerce").to_numpy()
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(Z[:, 0], Z[:, 1], c=c, s=90)
        plt.colorbar(sc, label="Doubling time (best-case min, h)")
        for i, name in enumerate(merged["strain"].tolist()):
            plt.text(Z[i, 0], Z[i, 1], name, fontsize=8, ha="left", va="bottom")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("WT RbcL embedding space (ESM2 pooled) colored by best-case doubling time")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "Figure_pca_embedding_space.png"), dpi=300)
        plt.close()

    # Pred vs actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y, preds)
    mn = float(min(np.min(y), np.min(preds)))
    mx = float(max(np.max(y), np.max(preds)))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Observed" + (" (log)" if args.log_target else ""))
    plt.ylabel("LOOCV predicted" + (" (log)" if args.log_target else ""))
    plt.title(f"LOOCV Ridge(PCA)  Spearman={metrics['spearman']:.2f}  R2={metrics['r2']:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "Figure_pred_vs_actual.png"), dpi=300)
    plt.close()

    # Manuscript snippet (fill numbers automatically)
    snippet = []
    snippet.append("## Exploratory embedding–growth analysis (auto-generated)\n")
    snippet.append(f"- n={metrics['n']} strains with WT RbcL embeddings and best-case doubling-time labels.")
    snippet.append(f"- LOOCV Ridge(PCA) on pooled ESM2 embeddings predicting `{args.target}`"
                   f"{' (log-transformed)' if args.log_target else ''}:")
    snippet.append(f"  - Spearman ρ = {metrics['spearman']:.2f}, R² = {metrics['r2']:.2f}.")
    snippet.append("- Interpretation: hypothesis-generating only (small n; doubling time is condition- and physiology-dependent).")
    with open(os.path.join(args.out_dir, "manuscript_snippet_embedding_vs_growth.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(snippet) + "\n")

    print(f"[done] wrote {args.out_dir}/metrics.json")
    print(f"[done] wrote {args.out_dir}/Table_embedding_vs_growth.csv")
    print(f"[done] wrote {args.out_dir}/Figure_pred_vs_actual.png")
    if os.path.exists(os.path.join(args.out_dir, 'Figure_pca_embedding_space.png')):
        print(f"[done] wrote {args.out_dir}/Figure_pca_embedding_space.png")
    print(f"[done] wrote {args.out_dir}/manuscript_snippet_embedding_vs_growth.md")

if __name__ == "__main__":
    main()
