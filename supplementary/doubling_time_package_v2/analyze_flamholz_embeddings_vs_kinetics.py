#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flamholz WT kinetics validation:
- Merge Flamholz S1 (WT-only) kinetics with UniProt-fetched sequences (meta) and ESM2 embeddings.
- Aggregate replicate kinetics per species (median).
- For each target, drop missing/invalid values, log10-transform (optional), then do K-fold CV.
- IMPORTANT: StandardScaler and PCA are fit on TRAIN ONLY in each fold (no leakage).

Outputs:
  out_dir/
    summary_metrics.csv
    pred_<target>.csv
    scatter_<target>.png
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# scikit-learn >=1.4 provides root_mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error
    HAVE_RMSE = True
except Exception:
    HAVE_RMSE = False

from scipy.stats import spearmanr

KIN_DEFAULT = ["vC", "KC", "S", "KO", "vO", "KRuBP"]

def safe_log10(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.where(x > 0)
    return np.log10(x)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if HAVE_RMSE:
        return float(root_mean_squared_error(y_true, y_pred))
    # For older sklearn, use sqrt(MSE). Newer sklearn removed squared=... from mean_squared_error. :contentReference[oaicite:1]{index=1}
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def cv_predict_ridge_pca(X: np.ndarray, y: np.ndarray, k: int, seed: int, pca_dim: int, alpha: float) -> np.ndarray:
    cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    yhat = np.full(len(y), np.nan, dtype=float)

    for fold, (tr, te) in enumerate(cv.split(X, y)):
        Xtr, Xte = X[tr], X[te]
        ytr = y[tr]

        # Fit scaler on train only
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        # Fit PCA on train only (prevents leakage)
        ncomp = min(pca_dim, Xtr_s.shape[1], max(1, Xtr_s.shape[0] - 1))
        pca = PCA(n_components=ncomp, random_state=seed, svd_solver="randomized")
        Xtr_p = pca.fit_transform(Xtr_s)
        Xte_p = pca.transform(Xte_s)

        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(Xtr_p, ytr)
        yhat[te] = model.predict(Xte_p)

    return yhat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flamholz_csv", required=True)
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--emb_npy", required=True)  # dict npy: {"ids": [...], "emb": [...]}
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--targets", nargs="+", default=KIN_DEFAULT)
    ap.add_argument("--log10", action="store_true", default=True)

    ap.add_argument("--wt_only", action="store_true", default=True,
                    help="Filter Flamholz to primary==1 and mutant==0 if columns exist.")
    ap.add_argument("--aggregate", choices=["median", "mean"], default="median")

    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca_dim", type=int, default=16)
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    fl = pd.read_csv(args.flamholz_csv)
    if "species" not in fl.columns:
        raise SystemExit("Expected 'species' column in Flamholz CSV.")

    # WT-only filtering
    if args.wt_only:
        if "primary" in fl.columns:
            fl = fl[fl["primary"] == 1].copy()
        if "mutant" in fl.columns:
            fl = fl[fl["mutant"] == 0].copy()

    # Aggregate kinetics per species (prevents overweighting species with many measurements)
    numcols = [c for c in args.targets if c in fl.columns]
    if not numcols:
        raise SystemExit(f"None of requested targets found in CSV. Available columns: {list(fl.columns)}")

    aggfunc = args.aggregate  # use string "median"/"mean" to avoid pandas FutureWarning
    fl_agg = fl.groupby("species", as_index=False)[numcols].agg(aggfunc)

    # Load meta and embeddings dict
    meta = pd.read_csv(args.meta_csv)
    emb_obj = np.load(args.emb_npy, allow_pickle=True).item()
    if "ids" not in emb_obj or "emb" not in emb_obj:
        raise SystemExit("emb_npy must be a dict npy with keys: ids, emb")

    ids = [str(x) for x in emb_obj["ids"].tolist()]
    X = np.asarray(emb_obj["emb"], dtype=float)

    if len(ids) != X.shape[0]:
        raise SystemExit("Embedding ids and emb rows mismatch.")

    # Map embedding labels -> species via meta (meta.label matches FASTA headers used for embedding)
    if "label" not in meta.columns or "species" not in meta.columns:
        raise SystemExit("meta_csv must contain 'label' and 'species' columns.")

    lab2species = dict(zip(meta["label"].astype(str), meta["species"].astype(str)))
    species_for_id = [lab2species.get(lab, "") for lab in ids]

    emb_df = pd.DataFrame(X, columns=[f"e{i}" for i in range(X.shape[1])])
    emb_df["species"] = species_for_id

    merged = fl_agg.merge(emb_df, on="species", how="inner")
    print(f"[info] species with kinetics+sequence: {len(merged)}")

    feat_cols = [c for c in merged.columns if c.startswith("e")]

    rows = []
    for t in args.targets:
        if t not in merged.columns:
            continue

        y = safe_log10(merged[t]) if args.log10 else pd.to_numeric(merged[t], errors="coerce")
        sub = merged.loc[y.notna()].reset_index(drop=True)
        y = y.loc[y.notna()].to_numpy(dtype=float)

        if len(sub) < max(5, args.k):
            print(f"[warn] {t}: too few samples after filtering (n={len(sub)}); skipping.")
            continue

        Xsub = sub[feat_cols].to_numpy(dtype=float)
        yhat = cv_predict_ridge_pca(Xsub, y, k=args.k, seed=args.seed, pca_dim=args.pca_dim, alpha=args.ridge_alpha)

        sp = float(spearmanr(y, yhat).correlation)
        r2 = float(r2_score(y, yhat))
        e = rmse(y, yhat)

        rows.append({"target": t, "n": len(y), "spearman": sp, "r2": r2, "rmse": e,
                     "log10": args.log10, "k": args.k, "pca_dim": args.pca_dim, "alpha": args.ridge_alpha})

        # Save predictions + plot
        pd.DataFrame({"species": sub["species"], "y_true": y, "y_pred": yhat}).to_csv(
            os.path.join(args.out_dir, f"pred_{t}.csv"), index=False
        )

        plt.figure(figsize=(6, 6), dpi=200)
        plt.scatter(y, yhat)
        mn = float(min(np.min(y), np.min(yhat)))
        mx = float(max(np.max(y), np.max(yhat)))
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("Observed" + (" (log10)" if args.log10 else ""))
        plt.ylabel("CV predicted" + (" (log10)" if args.log10 else ""))
        plt.title(f"{t}  n={len(y)}  Spearman={sp:.2f}  R2={r2:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"scatter_{t}.png"))
        plt.close()

        print(f"[done] {t}: n={len(y)} spearman={sp:.3f} r2={r2:.3f} rmse={e:.3f}")

    summ = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    summ.to_csv(os.path.join(args.out_dir, "summary_metrics.csv"), index=False)
    print(f"[done] wrote {os.path.join(args.out_dir, 'summary_metrics.csv')}")

if __name__ == "__main__":
    main()
