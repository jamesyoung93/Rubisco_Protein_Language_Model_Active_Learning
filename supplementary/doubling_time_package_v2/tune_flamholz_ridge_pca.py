#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nested-CV tuning for Flamholz kinetics prediction using ESM2 embeddings + PCA + Ridge.

Fixes:
- Aggregates NUMERIC kinetics per species, and carries taxonomy as a separate non-numeric field
  (prevents "Cannot convert ... to numeric" when aggregating).
- Robust parsing of heterologous_expression (bool or string).
- No PCA leakage: StandardScaler and PCA are fit on train folds only inside CV loops.
- Graceful handling of "no results" (writes empty summary instead of KeyError).

Outputs:
  out_dir/
    summary_nestedcv.csv
    pred_<target>.csv
    chosen_params_<target>.csv
"""

import argparse
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

try:
    from sklearn.metrics import root_mean_squared_error
    HAVE_RMSE = True
except Exception:
    HAVE_RMSE = False


KIN_DEFAULT = ["vC", "KC", "S", "KO", "vO", "KRuBP"]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def safe_log10(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    arr = np.where(arr > 0, arr, np.nan)
    return np.log10(arr)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if HAVE_RMSE:
        return float(root_mean_squared_error(y_true, y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_falseish(series: pd.Series) -> pd.Series:
    """
    Return boolean mask: True for "False"/0/no/etc.
    Handles both bool dtype and string/object dtype.
    """
    if series.dtype == bool:
        return series == False
    s = series.astype(str).str.lower().str.strip()
    # treat missing as unknown (False mask); user can decide otherwise
    return s.isin(["false", "0", "no", "none"])


def aggregate_by_species(
    fl: pd.DataFrame,
    numcols: List[str],
    agg: str,
    taxonomy_col: Optional[str] = "taxonomy",
) -> pd.DataFrame:
    """
    Aggregate numeric kinetic columns per species (median/mean).
    Carry taxonomy as first non-null per species (no numeric aggregation on taxonomy).
    """
    df = fl[["species"] + ([taxonomy_col] if taxonomy_col in fl.columns else []) + numcols].copy()

    # Force numeric for kinetics
    for c in numcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Numeric aggregation
    df_num = df.groupby("species", as_index=False)[numcols].agg(agg)

    if taxonomy_col in df.columns:
        df_tax = (
            df.groupby("species", as_index=False)[taxonomy_col]
            .agg(lambda x: x.dropna().iloc[0] if x.dropna().shape[0] else np.nan)
        )
        df_num = df_num.merge(df_tax, on="species", how="left")

    return df_num


def fit_predict_fold(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    pca_dim: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    # Scale on train only
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    # PCA on train only (no leakage)
    if pca_dim == 0:
        Xtr_p, Xte_p = Xtr_s, Xte_s
    else:
        ncomp = min(pca_dim, Xtr_s.shape[1], max(1, Xtr_s.shape[0] - 1))
        pca = PCA(n_components=ncomp, random_state=seed, svd_solver="randomized")
        Xtr_p = pca.fit_transform(Xtr_s)
        Xte_p = pca.transform(Xte_s)

    m = Ridge(alpha=alpha, random_state=seed)
    m.fit(Xtr_p, ytr)
    return m.predict(Xte_p)


def inner_oof_spearman(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    inner_cv,
    pca_dim: int,
    alpha: float,
    seed: int,
) -> float:
    """
    Compute inner-CV out-of-fold predictions and return Spearman correlation.
    """
    yhat = np.full(len(y), np.nan, dtype=float)
    splitter = inner_cv.split(X, y, groups=groups) if groups is not None else inner_cv.split(X, y)
    for tr, te in splitter:
        yhat[te] = fit_predict_fold(X[tr], y[tr], X[te], pca_dim, alpha, seed)
    sp = spearmanr(y, yhat).correlation
    return float(sp) if np.isfinite(sp) else -np.inf


def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    outer_cv,
    inner_cv,
    pca_grid: List[int],
    alpha_grid: List[float],
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Nested CV:
      - For each outer fold, tune (pca_dim, alpha) by inner OOF Spearman on outer-train.
      - Fit on full outer-train with best hyperparams, predict outer-test.
    Returns:
      yhat_oof, chosen_params_df
    """
    yhat = np.full(len(y), np.nan, dtype=float)
    chosen = []

    outer_splitter = outer_cv.split(X, y, groups=groups) if groups is not None else outer_cv.split(X, y)
    for ofold, (tr, te) in enumerate(outer_splitter):
        Xtr, ytr = X[tr], y[tr]
        Xte = X[te]
        gtr = groups[tr] if groups is not None else None

        best = None
        best_sp = -np.inf

        for pca_dim in pca_grid:
            for alpha in alpha_grid:
                sp = inner_oof_spearman(Xtr, ytr, gtr, inner_cv, pca_dim, alpha, seed)
                if sp > best_sp:
                    best_sp = sp
                    best = (pca_dim, alpha)

        pca_dim, alpha = best
        chosen.append({"outer_fold": ofold, "pca_dim": pca_dim, "alpha": alpha, "inner_spearman": best_sp})

        yhat[te] = fit_predict_fold(Xtr, ytr, Xte, pca_dim, alpha, seed)

    return yhat, pd.DataFrame(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flamholz_csv", required=True)
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--emb_npy", required=True)  # dict npy: {"ids":..., "emb":...}
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--targets", nargs="+", default=KIN_DEFAULT)
    ap.add_argument("--log10", action="store_true", default=True)

    ap.add_argument("--wt_only", action="store_true", default=False)
    ap.add_argument("--native_only", action="store_true", default=False)
    ap.add_argument("--temp_C", type=float, default=None)
    ap.add_argument("--ph_min", type=float, default=None)
    ap.add_argument("--ph_max", type=float, default=None)

    ap.add_argument("--aggregate", choices=["median", "mean"], default="median")
    ap.add_argument("--cv", choices=["kfold", "taxonomy_groupkfold"], default="kfold")
    ap.add_argument("--k_outer", type=int, default=5)
    ap.add_argument("--k_inner", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pca_grid", default="0,8,16,32,64")
    ap.add_argument("--alpha_grid", default="0.1,1,10,100")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    fl = pd.read_csv(args.flamholz_csv)
    if "species" not in fl.columns:
        raise SystemExit("Flamholz CSV must include a 'species' column.")

    # WT filter
    if args.wt_only:
        if "primary" in fl.columns:
            fl = fl[fl["primary"] == 1].copy()
        if "mutant" in fl.columns:
            fl = fl[fl["mutant"] == 0].copy()

    # Native-only filter (robust)
    if args.native_only and "heterologous_expression" in fl.columns:
        fl = fl[parse_falseish(fl["heterologous_expression"])].copy()

    # temp_C filter
    if args.temp_C is not None and "temp_C" in fl.columns:
        tc = pd.to_numeric(fl["temp_C"], errors="coerce")
        fl = fl[tc == float(args.temp_C)].copy()

    # pH range filter
    if "pH" in fl.columns and (args.ph_min is not None or args.ph_max is not None):
        ph = pd.to_numeric(fl["pH"], errors="coerce")
        if args.ph_min is not None:
            fl = fl[ph >= float(args.ph_min)].copy()
        if args.ph_max is not None:
            fl = fl[ph <= float(args.ph_max)].copy()

    # Choose kinetic columns present
    numcols = [c for c in args.targets if c in fl.columns]
    if not numcols:
        raise SystemExit(f"No requested targets found in Flamholz CSV. Requested={args.targets}")

    # Aggregate to one row per species
    fl_agg = aggregate_by_species(fl, numcols=numcols, agg=args.aggregate, taxonomy_col="taxonomy")

    # Load meta + embeddings dict
    meta = pd.read_csv(args.meta_csv)
    emb_obj = np.load(args.emb_npy, allow_pickle=True).item()
    if "ids" not in emb_obj or "emb" not in emb_obj:
        raise SystemExit("emb_npy must be a dict with keys 'ids' and 'emb'")

    emb_ids = [str(x) for x in emb_obj["ids"].tolist()]
    X_all = np.asarray(emb_obj["emb"], dtype=float)

    if "label" not in meta.columns or "species" not in meta.columns:
        raise SystemExit("meta_csv must contain columns: label, species")

    lab2species = dict(zip(meta["label"].astype(str), meta["species"].astype(str)))
    species_for_id = [lab2species.get(l, "") for l in emb_ids]

    emb_df = pd.DataFrame(X_all, columns=[f"e{i}" for i in range(X_all.shape[1])])
    emb_df["species"] = species_for_id

    merged = fl_agg.merge(emb_df, on="species", how="inner")
    feat_cols = [c for c in merged.columns if c.startswith("e")]

    # Hyperparameter grids
    pca_grid = [int(x) for x in args.pca_grid.split(",") if x.strip() != ""]
    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip() != ""]

    results = []

    for target in args.targets:
        if target not in merged.columns:
            continue

        y_raw = pd.to_numeric(merged[target], errors="coerce").to_numpy(dtype=float)
        if args.log10:
            y = safe_log10(y_raw)
        else:
            y = y_raw

        keep = np.isfinite(y)
        sub = merged.loc[keep].reset_index(drop=True)
        y = y[keep]
        if len(sub) < max(args.k_outer, 10):
            print(f"[warn] {target}: too few samples after filtering (n={len(sub)}); skipping.")
            continue

        X = sub[feat_cols].to_numpy(dtype=float)

        # CV splitter
        if args.cv == "taxonomy_groupkfold":
            if "taxonomy" not in sub.columns:
                print(f"[warn] {target}: taxonomy_groupkfold requested but taxonomy missing; skipping.")
                continue
            groups = sub["taxonomy"].astype(str).fillna("NA").to_numpy()
            n_groups = len(np.unique(groups))
            if n_groups < 3:
                print(f"[warn] {target}: too few taxonomy groups (n_groups={n_groups}); skipping.")
                continue
            outer = GroupKFold(n_splits=min(args.k_outer, n_groups))
            inner = GroupKFold(n_splits=min(args.k_inner, n_groups))
        else:
            groups = None
            outer = KFold(n_splits=args.k_outer, shuffle=True, random_state=args.seed)
            inner = KFold(n_splits=args.k_inner, shuffle=True, random_state=args.seed)

        yhat, chosen = nested_cv(X, y, groups, outer, inner, pca_grid, alpha_grid, args.seed)

        sp = float(spearmanr(y, yhat).correlation)
        r2 = float(r2_score(y, yhat))
        e = float(rmse(y, yhat))

        # Save artifacts
        chosen.to_csv(os.path.join(args.out_dir, f"chosen_params_{target}.csv"), index=False)
        pd.DataFrame({"species": sub["species"], "y_true": y, "y_pred": yhat}).to_csv(
            os.path.join(args.out_dir, f"pred_{target}.csv"), index=False
        )

        results.append({
            "target": target,
            "n": int(len(y)),
            "spearman": sp,
            "r2": r2,
            "rmse": e,
            "cv": args.cv,
            "log10": bool(args.log10),
            "wt_only": bool(args.wt_only),
            "native_only": bool(args.native_only),
            "temp_C": args.temp_C,
            "ph_min": args.ph_min,
            "ph_max": args.ph_max,
            "aggregate": args.aggregate,
        })

        print(f"[done] {target}: n={len(y)} spearman={sp:.3f} r2={r2:.3f} rmse={e:.3f}")

    out_path = os.path.join(args.out_dir, "summary_nestedcv.csv")
    out = pd.DataFrame(results)

    if out.empty:
        print("[warn] No targets produced results under these filters. Writing empty summary_nestedcv.csv")
        out.to_csv(out_path, index=False)
        return

    out = out.sort_values("spearman", ascending=False)
    out.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
