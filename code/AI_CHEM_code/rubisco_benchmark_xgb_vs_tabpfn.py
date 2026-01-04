#!/usr/bin/env python3
import argparse, os, json, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

import xgboost as xgb
from tabpfn import TabPFNRegressor

try:
    import torch
except Exception:
    torch = None


# -------------------------
# Metrics
# -------------------------
def safe_spearman(y, p) -> float:
    c = spearmanr(y, p).correlation
    return float(c) if np.isfinite(c) else np.nan

def topk_overlap_precision(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.05) -> float:
    n = len(y_true)
    k = max(1, int(np.ceil(frac * n)))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    return len(true_top & pred_top) / k

def topk_mean_true(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.05) -> float:
    n = len(y_true)
    k = max(1, int(np.ceil(frac * n)))
    pred_top = np.argsort(y_pred)[-k:]
    return float(np.mean(y_true[pred_top]))

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {
        "spearman": safe_spearman(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "mse": float(mean_squared_error(y_true, y_pred)),
        "top5_precision": float(topk_overlap_precision(y_true, y_pred, 0.05)),
        "top5_mean_true": float(topk_mean_true(y_true, y_pred, 0.05)),
        "mean_true": float(np.mean(y_true)),
    }
    out["top5_enrich_diff"] = out["top5_mean_true"] - out["mean_true"]
    if abs(out["mean_true"]) > 1e-12:
        out["top5_enrich_ratio"] = out["top5_mean_true"] / out["mean_true"]
    else:
        out["top5_enrich_ratio"] = np.nan
    return out


# -------------------------
# Models
# -------------------------
@dataclass
class XGBParams:
    max_depth: int = 6
    reg_lambda: float = 10.0
    eta: float = 0.03
    subsample: float = 0.85
    colsample: float = 0.85
    min_child_weight: float = 1.0
    num_boost_round: int = 8000
    early_stop: int = 200
    val_frac: float = 0.10
    nthread: int = 16
    seed: int = 0

def fit_predict_xgb(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, params: XGBParams) -> np.ndarray:
    rng = np.random.default_rng(params.seed)
    idx = np.arange(len(ytr))
    tr_idx, va_idx = train_test_split(idx, test_size=params.val_frac, random_state=int(rng.integers(1e9)))

    dtr = xgb.DMatrix(Xtr[tr_idx], label=ytr[tr_idx])
    dva = xgb.DMatrix(Xtr[va_idx], label=ytr[va_idx])
    dte = xgb.DMatrix(Xte)

    p = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": params.eta,
        "max_depth": int(params.max_depth),
        "min_child_weight": float(params.min_child_weight),
        "subsample": float(params.subsample),
        "colsample_bytree": float(params.colsample),
        "lambda": float(params.reg_lambda),
        "tree_method": "hist",
        "seed": int(params.seed),
        "nthread": int(params.nthread),
    }

    bst = xgb.train(
        p, dtr,
        num_boost_round=int(params.num_boost_round),
        evals=[(dva, "val")],
        early_stopping_rounds=int(params.early_stop),
        verbose_eval=False,
    )
    bi = bst.best_iteration
    if bi is None:
        pred = bst.predict(dte)
    else:
        pred = bst.predict(dte, iteration_range=(0, bi + 1))
    return pred.astype(np.float32)

def make_tabpfn(device: str, ignore_limits: bool):
    try:
        return TabPFNRegressor(device=device, ignore_pretraining_limits=ignore_limits)
    except TypeError:
        return TabPFNRegressor(device=device)

def fit_predict_tabpfn(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
    device: str, ignore_limits: bool,
    train_cap: int, seed: int
) -> np.ndarray:
    n = len(ytr)
    idx = np.arange(n)
    if train_cap and train_cap > 0 and n > train_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=train_cap, replace=False)

    reg = make_tabpfn(device, ignore_limits)
    reg.fit(Xtr[idx], ytr[idx])
    return reg.predict(Xte).astype(np.float32)


# -------------------------
# Feature transform: fit on train only
# -------------------------
def transform_train_test(
    Xemb: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    pca_dim: int,
    Xnum: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xemb[tr_idx])
    Xte_s = scaler.transform(Xemb[te_idx])

    pca = PCA(n_components=int(pca_dim), random_state=0, svd_solver="randomized")
    Xtr_p = pca.fit_transform(Xtr_s).astype(np.float32)
    Xte_p = pca.transform(Xte_s).astype(np.float32)

    if Xnum is not None:
        scn = StandardScaler()
        Xtr_n = scn.fit_transform(Xnum[tr_idx]).astype(np.float32)
        Xte_n = scn.transform(Xnum[te_idx]).astype(np.float32)
        Xtr_p = np.hstack([Xtr_p, Xtr_n]).astype(np.float32)
        Xte_p = np.hstack([Xte_p, Xte_n]).astype(np.float32)

    return Xtr_p, Xte_p


# -------------------------
# Dataset extraction
# -------------------------
def get_dms(df: pd.DataFrame, Xemb_all: np.ndarray, target: str):
    mask = (df["dataset_id"].values == "DMS")
    d = df.loc[mask].copy()
    X = Xemb_all[mask]
    y = pd.to_numeric(d[target], errors="coerce").to_numpy(dtype=np.float32)
    pos = pd.to_numeric(d["position_external"], errors="coerce").to_numpy()
    keep = np.isfinite(y) & np.isfinite(pos)
    d = d.loc[keep].copy()
    X = X[keep]
    y = y[keep]
    pos = pos[keep].astype(int)
    ids = d["variant_id"].astype(str).to_numpy()
    return ids, X, y, pos

def get_hoff(df: pd.DataFrame, Xemb_all: np.ndarray, target: str, add_nmut_features: bool):
    mask = (df["dataset_id"].values == "HOFF")
    h = df.loc[mask].copy()
    X = Xemb_all[mask]

    if "has_stop" in h.columns:
        good = ~h["has_stop"].fillna(False).astype(bool).to_numpy()
        h = h.loc[good].copy()
        X = X[good]

    y = pd.to_numeric(h[target], errors="coerce").to_numpy(dtype=np.float32)
    nmut = pd.to_numeric(h["n_mut"], errors="coerce").to_numpy()
    keep = np.isfinite(y) & np.isfinite(nmut)
    h = h.loc[keep].copy()
    X = X[keep]
    y = y[keep]
    nmut = nmut[keep].astype(int)

    Xnum = None
    if add_nmut_features:
        nm = nmut.astype(np.float32)
        Xnum = np.vstack([nm, nm**2]).T.astype(np.float32)

    ids = h["variant_id"].astype(str).to_numpy()
    return ids, X, y, nmut, Xnum


# -------------------------
# Benchmark runners
# -------------------------
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npy", default="esm2_t33_650m_full.npy")
    ap.add_argument("--labels_csv", default="rubisco_datasets_merged.csv")
    ap.add_argument("--out_dir", default="results_rubisco_benchmark_xgb_tabpfn")

    ap.add_argument("--datasets", choices=["DMS", "HOFF", "BOTH"], default="BOTH")
    ap.add_argument("--dms_target", default="dms_enrichment_mean")
    ap.add_argument("--hoff_target", default="hoff_delta_O2_minus_N2")

    ap.add_argument("--pca_dims", default="64,128,256")
    ap.add_argument("--split_seeds", default="0,1,2")         # for DMS within-position
    ap.add_argument("--model_seeds", default="0,1,2,3,4")     # for XGB val split + TabPFN subsample
    ap.add_argument("--tabpfn_caps", default="0,5000,2000")   # 0 means full train
    ap.add_argument("--tabpfn_device", default="cuda", help="cuda|cpu|auto")
    ap.add_argument("--tabpfn_ignore_limits", action="store_true")

    # XGB params
    ap.add_argument("--xgb_max_depth", type=int, default=6)
    ap.add_argument("--xgb_reg_lambda", type=float, default=10.0)
    ap.add_argument("--xgb_num_round", type=int, default=8000)
    ap.add_argument("--xgb_early_stop", type=int, default=200)
    ap.add_argument("--xgb_val_frac", type=float, default=0.10)
    ap.add_argument("--xgb_nthread", type=int, default=16)

    # Hoffmann options
    ap.add_argument("--hoff_add_nmut_features", action="store_true")

    # Optional: save per-test predictions (many files)
    ap.add_argument("--save_predictions", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "predictions"), exist_ok=True)

    # GPU check if requested
    if args.tabpfn_device == "cuda":
        if torch is None:
            raise RuntimeError("torch not importable; cannot use --tabpfn_device cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Run this on a GPU node or set --tabpfn_device cpu/auto.")

    pca_dims = parse_int_list(args.pca_dims)
    split_seeds = parse_int_list(args.split_seeds)
    model_seeds = parse_int_list(args.model_seeds)
    tabpfn_caps = parse_int_list(args.tabpfn_caps)

    # Load embeddings and labels aligned
    embd = np.load(args.emb_npy, allow_pickle=True).item()
    ids_all = embd["ids"].astype(str)
    Xemb_all = embd["emb"].astype(np.float32)

    df = pd.read_csv(args.labels_csv, low_memory=False)
    df = df.set_index("variant_id").loc[ids_all].reset_index()

    runs = []
    split_manifest = {"DMS": {}, "HOFF": {}}

    # ---------------- DMS ----------------
    if args.datasets in ("DMS", "BOTH"):
        d_ids, d_X, d_y, d_pos = get_dms(df, Xemb_all, args.dms_target)

        # within-position splits across split_seeds
        for sseed in split_seeds:
            tr, te = train_test_split(np.arange(len(d_y)), test_size=0.2, random_state=sseed, stratify=d_pos)
            split_manifest["DMS"][f"within_seed{sseed}"] = {"n_train": int(len(tr)), "n_test": int(len(te))}

            for pca_dim in pca_dims:
                Xtr_p, Xte_p = transform_train_test(d_X, tr, te, pca_dim=pca_dim, Xnum=None)
                ytr, yte = d_y[tr], d_y[te]

                # XGB across model_seeds (val-split randomness)
                for mseed in model_seeds:
                    t0 = time.time()
                    pred = fit_predict_xgb(Xtr_p, ytr, Xte_p, XGBParams(
                        max_depth=args.xgb_max_depth,
                        reg_lambda=args.xgb_reg_lambda,
                        num_boost_round=args.xgb_num_round,
                        early_stop=args.xgb_early_stop,
                        val_frac=args.xgb_val_frac,
                        nthread=args.xgb_nthread,
                        seed=mseed
                    ))
                    met = eval_metrics(yte, pred)
                    runs.append({
                        "dataset":"DMS", "target":args.dms_target, "split":"within_position",
                        "split_id":f"within_seed{sseed}", "fold":-1,
                        "model":"xgb", "pca_dim":pca_dim, "model_seed":mseed, "tabpfn_cap":np.nan,
                        "n_train":int(len(tr)), "n_test":int(len(te)), "seconds":time.time()-t0, **met
                    })
                    if args.save_predictions:
                        outp = pd.DataFrame({"variant_id": d_ids[te], "y_true": yte, "y_pred": pred})
                        outp.to_parquet(os.path.join(args.out_dir, "predictions",
                                                     f"DMS_within_seed{sseed}_xgb_pca{pca_dim}_seed{mseed}.parquet"),
                                        index=False)

                # TabPFN across caps and model_seeds
                for cap in tabpfn_caps:
                    for mseed in model_seeds:
                        t0 = time.time()
                        pred = fit_predict_tabpfn(Xtr_p, ytr, Xte_p,
                                                  device=args.tabpfn_device,
                                                  ignore_limits=args.tabpfn_ignore_limits,
                                                  train_cap=cap,
                                                  seed=mseed)
                        met = eval_metrics(yte, pred)
                        runs.append({
                            "dataset":"DMS", "target":args.dms_target, "split":"within_position",
                            "split_id":f"within_seed{sseed}", "fold":-1,
                            "model":"tabpfn", "pca_dim":pca_dim, "model_seed":mseed, "tabpfn_cap":cap,
                            "n_train":int(len(tr)), "n_test":int(len(te)), "seconds":time.time()-t0, **met
                        })
                        if args.save_predictions:
                            outp = pd.DataFrame({"variant_id": d_ids[te], "y_true": yte, "y_pred": pred})
                            outp.to_parquet(os.path.join(args.out_dir, "predictions",
                                                         f"DMS_within_seed{sseed}_tabpfn_pca{pca_dim}_cap{cap}_seed{mseed}.parquet"),
                                            index=False)

        # position-holdout CV (GroupKFold)
        uniq = np.unique(d_pos)
        n_splits = min(5, len(uniq))
        gkf = GroupKFold(n_splits=n_splits)
        for fold_i, (tr, te) in enumerate(gkf.split(np.zeros(len(d_y)), d_y, groups=d_pos), start=1):
            split_manifest["DMS"][f"poscv_fold{fold_i}"] = {"n_train": int(len(tr)), "n_test": int(len(te))}
            for pca_dim in pca_dims:
                Xtr_p, Xte_p = transform_train_test(d_X, tr, te, pca_dim=pca_dim, Xnum=None)
                ytr, yte = d_y[tr], d_y[te]

                for mseed in model_seeds:
                    t0 = time.time()
                    pred = fit_predict_xgb(Xtr_p, ytr, Xte_p, XGBParams(
                        max_depth=args.xgb_max_depth,
                        reg_lambda=args.xgb_reg_lambda,
                        num_boost_round=args.xgb_num_round,
                        early_stop=args.xgb_early_stop,
                        val_frac=args.xgb_val_frac,
                        nthread=args.xgb_nthread,
                        seed=mseed
                    ))
                    met = eval_metrics(yte, pred)
                    runs.append({
                        "dataset":"DMS", "target":args.dms_target, "split":"pos_holdout",
                        "split_id":f"poscv_fold{fold_i}", "fold":fold_i,
                        "model":"xgb", "pca_dim":pca_dim, "model_seed":mseed, "tabpfn_cap":np.nan,
                        "n_train":int(len(tr)), "n_test":int(len(te)), "seconds":time.time()-t0, **met
                    })

                for cap in tabpfn_caps:
                    for mseed in model_seeds:
                        t0 = time.time()
                        pred = fit_predict_tabpfn(Xtr_p, ytr, Xte_p,
                                                  device=args.tabpfn_device,
                                                  ignore_limits=args.tabpfn_ignore_limits,
                                                  train_cap=cap,
                                                  seed=mseed)
                        met = eval_metrics(yte, pred)
                        runs.append({
                            "dataset":"DMS", "target":args.dms_target, "split":"pos_holdout",
                            "split_id":f"poscv_fold{fold_i}", "fold":fold_i,
                            "model":"tabpfn", "pca_dim":pca_dim, "model_seed":mseed, "tabpfn_cap":cap,
                            "n_train":int(len(tr)), "n_test":int(len(te)), "seconds":time.time()-t0, **met
                        })

    # ---------------- HOFFMANN ----------------
    if args.datasets in ("HOFF", "BOTH"):
        h_ids, h_X, h_y, h_nmut, h_Xnum = get_hoff(df, Xemb_all, args.hoff_target, add_nmut_features=args.hoff_add_nmut_features)

        # depth holdout split fixed by n_mut, but val fallback uses seed
        base_tr = np.where(h_nmut <= 4)[0]
        base_va = np.where(h_nmut == 5)[0]
        base_te = np.where(h_nmut >= 6)[0]
        if len(base_te) < 50:
            base_te = np.where(h_nmut >= 5)[0]

        for mseed in model_seeds:
            # validation fallback if n_mut==5 sparse
            if len(base_va) < 50:
                tr_idx, va_idx = train_test_split(base_tr, test_size=0.1, random_state=mseed)
            else:
                tr_idx, va_idx = base_tr, base_va
            te_idx = base_te
            split_manifest["HOFF"][f"depth_seed{mseed}"] = {"n_train": int(len(tr_idx)), "n_test": int(len(te_idx))}

            for pca_dim in pca_dims:
                Xtr_p, Xte_p = transform_train_test(h_X, tr_idx, te_idx, pca_dim=pca_dim, Xnum=h_Xnum)
                ytr, yte = h_y[tr_idx], h_y[te_idx]

                # XGB (use same mseed as internal val split seed)
                t0 = time.time()
                pred = fit_predict_xgb(Xtr_p, ytr, Xte_p, XGBParams(
                    max_depth=args.xgb_max_depth,
                    reg_lambda=args.xgb_reg_lambda,
                    num_boost_round=args.xgb_num_round,
                    early_stop=args.xgb_early_stop,
                    val_frac=args.xgb_val_frac,
                    nthread=args.xgb_nthread,
                    seed=mseed
                ))
                met = eval_metrics(yte, pred)
                runs.append({
                    "dataset":"HOFF", "target":args.hoff_target, "split":"depth_holdout",
                    "split_id":f"depth_seed{mseed}", "fold":-1,
                    "model":"xgb", "pca_dim":pca_dim, "model_seed":mseed, "tabpfn_cap":np.nan,
                    "n_train":int(len(tr_idx)), "n_test":int(len(te_idx)), "seconds":time.time()-t0, **met
                })

                # TabPFN across caps
                for cap in tabpfn_caps:
                    t0 = time.time()
                    pred = fit_predict_tabpfn(Xtr_p, ytr, Xte_p,
                                              device=args.tabpfn_device,
                                              ignore_limits=args.tabpfn_ignore_limits,
                                              train_cap=cap,
                                              seed=mseed)
                    met = eval_metrics(yte, pred)
                    runs.append({
                        "dataset":"HOFF", "target":args.hoff_target, "split":"depth_holdout",
                        "split_id":f"depth_seed{mseed}", "fold":-1,
                        "model":"tabpfn", "pca_dim":pca_dim, "model_seed":mseed, "tabpfn_cap":cap,
                        "n_train":int(len(tr_idx)), "n_test":int(len(te_idx)), "seconds":time.time()-t0, **met
                    })

    runs_df = pd.DataFrame(runs)
    runs_path = os.path.join(args.out_dir, "runs.csv")
    runs_df.to_csv(runs_path, index=False)

    # Summary for stats: mean/std/count over replicates (seeds and folds)
    group_cols = ["dataset","target","split","model","pca_dim","tabpfn_cap"]
    summ = (runs_df.groupby(group_cols)
                  .agg(
                      spearman_mean=("spearman","mean"),
                      spearman_std=("spearman","std"),
                      r2_mean=("r2","mean"),
                      r2_std=("r2","std"),
                      top5_precision_mean=("top5_precision","mean"),
                      top5_precision_std=("top5_precision","std"),
                      n_runs=("spearman","count"),
                  )
                  .reset_index())
    summ_path = os.path.join(args.out_dir, "summary.csv")
    summ.to_csv(summ_path, index=False)

    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump(split_manifest, f, indent=2)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Wrote:", runs_path)
    print("Wrote:", summ_path)
    print("Wrote:", os.path.join(args.out_dir, "splits.json"))

    # Print a short summary for quick inspection
    print("\n=== Quick view: best mean spearman by dataset/split/model ===")
    q = (summ.sort_values("spearman_mean", ascending=False)
             .groupby(["dataset","split","model"])
             .head(3))
    print(q[["dataset","split","model","pca_dim","tabpfn_cap","spearman_mean","spearman_std","n_runs"]].to_string(index=False))


if __name__ == "__main__":
    main()
