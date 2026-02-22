#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR

import xgboost as xgb

try:
    from tabpfn import TabPFNRegressor
except Exception:
    TabPFNRegressor = None


def safe_spearman(y, p):
    c = spearmanr(y, p).correlation
    return float(c) if np.isfinite(c) else np.nan


def topk_precision(y_true, y_pred, frac=0.05):
    n = len(y_true)
    k = max(1, int(np.ceil(frac * n)))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    return len(true_top & pred_top) / k


def topk_enrich_diff(y_true, y_pred, frac=0.05):
    n = len(y_true)
    k = max(1, int(np.ceil(frac * n)))
    pred_top = np.argsort(y_pred)[-k:]
    return float(np.mean(y_true[pred_top]) - np.mean(y_true))


def metric_row(y_true, y_pred):
    return {
        "spearman": safe_spearman(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "top5_precision": float(topk_precision(y_true, y_pred, 0.05)),
        "top5_enrich_diff": float(topk_enrich_diff(y_true, y_pred, 0.05)),
    }


def bootstrap_percentile_ci(values: np.ndarray, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return np.nan, np.nan
    boots = np.array([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def quantile_balance_indices(y: np.ndarray, n_bins: int = 5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    q = np.asarray(q)
    counts = [np.sum(q == b) for b in np.unique(q)]
    keep_n = int(min(counts))
    keep = []
    for b in np.unique(q):
        idx = np.where(q == b)[0]
        keep.extend(rng.choice(idx, size=keep_n, replace=False).tolist())
    return np.array(sorted(keep), dtype=int)


def transform(X, tr, te, pca_dim):
    scaler = StandardScaler()
    pca = PCA(n_components=pca_dim, random_state=42)
    Xtr = pca.fit_transform(scaler.fit_transform(X[tr]))
    Xte = pca.transform(scaler.transform(X[te]))
    return Xtr.astype(np.float32), Xte.astype(np.float32)


def fit_predict_xgb(Xtr, ytr, Xte, seed=42):
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dte = xgb.DMatrix(Xte)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 10.0,
        "tree_method": "hist",
        "seed": seed,
    }
    bst = xgb.train(params, dtr, num_boost_round=1200, verbose_eval=False)
    return bst.predict(dte)


def fit_predict_tabpfn(Xtr, ytr, Xte, device="cpu"):
    if TabPFNRegressor is None:
        raise RuntimeError("tabpfn is not installed")
    reg = TabPFNRegressor(device=device)
    reg.fit(Xtr, ytr)
    return reg.predict(Xte)


def fit_predict_svr(Xtr, ytr, Xte, seed=42, use_linear_large_n=True):
    if use_linear_large_n and len(ytr) > 8000:
        model = LinearSVR(C=1.0, epsilon=0.1, random_state=seed, max_iter=5000)
        model.fit(Xtr, ytr)
        return model.predict(Xte), {"svr_variant": "LinearSVR", "C": 1.0, "epsilon": 0.1}

    grid = list(ParameterGrid({"kernel": ["rbf", "linear"], "C": [0.1, 1.0, 10.0], "epsilon": [0.01, 0.1, 0.2]}))
    Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.15, random_state=seed)
    best = None
    for g in grid:
        m = SVR(**g)
        m.fit(Xtr2, ytr2)
        p = m.predict(Xva)
        s = safe_spearman(yva, p)
        if best is None or s > best[0]:
            best = (s, g)
    final = SVR(**best[1])
    final.fit(Xtr, ytr)
    return final.predict(Xte), {"svr_variant": "SVR", **best[1]}


def fit_predict_mlp(Xtr, ytr, Xte, seed=42):
    grid = list(ParameterGrid({
        "hidden_layer_sizes": [(256, 128), (512, 256), (256, 128, 64)],
        "learning_rate_init": [1e-3, 3e-4],
        "alpha": [1e-5, 1e-4],
    }))
    Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.15, random_state=seed)
    best = None
    for g in grid:
        m = MLPRegressor(
            hidden_layer_sizes=g["hidden_layer_sizes"],
            learning_rate_init=g["learning_rate_init"],
            alpha=g["alpha"],
            max_iter=400,
            early_stopping=True,
            random_state=seed,
        )
        m.fit(Xtr2, ytr2)
        p = m.predict(Xva)
        s = safe_spearman(yva, p)
        if best is None or s > best[0]:
            best = (s, g)
    final = MLPRegressor(**best[1], max_iter=600, early_stopping=True, random_state=seed)
    final.fit(Xtr, ytr)
    return final.predict(Xte), best[1]


def evaluate_split(X, y, split_name, tr, te, pca_dim, seed, tabpfn_device, use_balance=False):
    Xtr, Xte = transform(X, tr, te, pca_dim)
    ytr, yte = y[tr], y[te]
    balance_strategy = "none"
    if use_balance:
        keep = quantile_balance_indices(ytr, n_bins=5, seed=seed)
        Xtr, ytr = Xtr[keep], ytr[keep]
        balance_strategy = "quantile_subsample_q5"

    rows = []
    pred = fit_predict_xgb(Xtr, ytr, Xte, seed=seed)
    rows.append({"model": "xgb", **metric_row(yte, pred)})

    if TabPFNRegressor is not None:
        try:
            pred = fit_predict_tabpfn(Xtr, ytr, Xte, device=tabpfn_device)
            rows.append({"model": "tabpfn", **metric_row(yte, pred)})
        except Exception:
            pass

    pred, cfg = fit_predict_svr(Xtr, ytr, Xte, seed=seed)
    rows.append({"model": "svr", **metric_row(yte, pred), **{f"hp_{k}": v for k, v in cfg.items()}})

    pred, cfg = fit_predict_mlp(Xtr, ytr, Xte, seed=seed)
    rows.append({"model": "mlp", **metric_row(yte, pred), **{f"hp_{k}": v for k, v in cfg.items()}})

    for r in rows:
        r.update({
            "split": split_name,
            "pca_dim": pca_dim,
            "seed": seed,
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "balance_strategy": balance_strategy,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npy", required=True)
    ap.add_argument("--labels_csv", default="code/AI_CHEM_code/rubisco_datasets_merged.csv")
    ap.add_argument("--out_dir", default="results/results_model_extensions")
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--dms_target", default="dms_enrichment_mean")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tabpfn_device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    emb = np.load(args.emb_npy, allow_pickle=True).item()
    ids = emb["ids"].astype(str)
    Xall = emb["emb"].astype(np.float32)
    df = pd.read_csv(args.labels_csv).set_index("variant_id").loc[ids].reset_index()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    runs = []

    # DMS within-position + position-holdout
    dms_m = df["dataset_id"].eq("DMS").to_numpy()
    dms = df.loc[dms_m].copy()
    Xd = Xall[dms_m]
    yd = pd.to_numeric(dms[args.dms_target], errors="coerce").to_numpy(dtype=np.float32)
    pos = pd.to_numeric(dms["position_external"], errors="coerce").to_numpy()
    keep = np.isfinite(yd) & np.isfinite(pos)
    Xd, yd, pos = Xd[keep], yd[keep], pos[keep].astype(int)

    for seed in seeds:
        tr, te = train_test_split(np.arange(len(yd)), test_size=0.2, random_state=seed, stratify=pos)
        runs.extend(evaluate_split(Xd, yd, "dms_within_position", tr, te, args.pca_dim, seed, args.tabpfn_device, use_balance=False))
        runs.extend(evaluate_split(Xd, yd, "dms_within_position", tr, te, args.pca_dim, seed, args.tabpfn_device, use_balance=True))

    gkf = GroupKFold(n_splits=min(5, len(np.unique(pos))))
    for fold, (tr, te) in enumerate(gkf.split(Xd, yd, groups=pos), start=1):
        runs.extend(evaluate_split(Xd, yd, f"dms_pos_holdout_fold{fold}", tr, te, args.pca_dim, 42 + fold, args.tabpfn_device, use_balance=False))
        runs.extend(evaluate_split(Xd, yd, f"dms_pos_holdout_fold{fold}", tr, te, args.pca_dim, 42 + fold, args.tabpfn_device, use_balance=True))

    # HOFF depth holdout
    hm = df["dataset_id"].eq("HOFF").to_numpy()
    h = df.loc[hm].copy()
    Xh = Xall[hm]
    yh = pd.to_numeric(h["hoff_delta_O2_minus_N2"], errors="coerce").to_numpy(dtype=np.float32)
    nmut = pd.to_numeric(h["n_mut"], errors="coerce").to_numpy()
    keep = np.isfinite(yh) & np.isfinite(nmut)
    Xh, yh, nmut = Xh[keep], yh[keep], nmut[keep].astype(int)
    tr = np.where(nmut <= 4)[0]
    te = np.where(nmut >= 6)[0]
    if len(te) < 10:
        te = np.where(nmut >= 5)[0]
    runs.extend(evaluate_split(Xh, yh, "hoff_depth_holdout", tr, te, args.pca_dim, 42, args.tabpfn_device, use_balance=False))
    runs.extend(evaluate_split(Xh, yh, "hoff_depth_holdout", tr, te, args.pca_dim, 42, args.tabpfn_device, use_balance=True))

    # Cross-organism random split if available
    cross_mask = df["dataset_id"].isin(["CYANO_DOUBLING", "FLAMHOLZ", "DOUBLING", "KINETICS"]).to_numpy()
    if np.any(cross_mask):
        cdf = df.loc[cross_mask].copy()
        Xc = Xall[cross_mask]
        target_col = "target_value" if "target_value" in cdf.columns else None
        if target_col is not None:
            yc = pd.to_numeric(cdf[target_col], errors="coerce").to_numpy(dtype=np.float32)
            keep = np.isfinite(yc)
            Xc, yc = Xc[keep], yc[keep]
            for seed in seeds:
                tr, te = train_test_split(np.arange(len(yc)), test_size=0.2, random_state=seed)
                runs.extend(evaluate_split(Xc, yc, "cross_organism_random", tr, te, args.pca_dim, seed, args.tabpfn_device, use_balance=False))

    runs_df = pd.DataFrame(runs)
    runs_path = os.path.join(args.out_dir, "runs_extended_models.csv")
    runs_df.to_csv(runs_path, index=False)

    summary = []
    for (split, model, bal), sub in runs_df.groupby(["split", "model", "balance_strategy"], dropna=False):
        s_ci = bootstrap_percentile_ci(sub["spearman"].to_numpy())
        p_ci = bootstrap_percentile_ci(sub["top5_precision"].to_numpy())
        summary.append({
            "split": split,
            "model": model,
            "balance_strategy": bal,
            "n": len(sub),
            "spearman_mean": sub["spearman"].mean(),
            "spearman_ci95_low": s_ci[0],
            "spearman_ci95_high": s_ci[1],
            "top5_precision_mean": sub["top5_precision"].mean(),
            "top5_precision_ci95_low": p_ci[0],
            "top5_precision_ci95_high": p_ci[1],
            "rmse_mean": sub["rmse"].mean(),
            "top5_enrich_diff_mean": sub["top5_enrich_diff"].mean(),
        })
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(args.out_dir, "table_model_balance_bootstrap_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Wrote", runs_path)
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
