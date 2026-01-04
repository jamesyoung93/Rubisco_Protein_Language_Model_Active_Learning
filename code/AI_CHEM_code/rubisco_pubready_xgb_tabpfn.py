#!/usr/bin/env python3
import argparse, os, json, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, ttest_rel, wilcoxon

import xgboost as xgb
from tabpfn import TabPFNRegressor

import matplotlib.pyplot as plt

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
    out["top5_enrich_ratio"] = out["top5_mean_true"] / out["mean_true"] if abs(out["mean_true"]) > 1e-12 else np.nan
    return out


# -------------------------
# Models
# -------------------------
class XGBParams:
    def __init__(self, max_depth=6, reg_lambda=10.0, eta=0.03, subsample=0.85, colsample=0.85,
                 min_child_weight=1.0, num_boost_round=8000, early_stop=200, val_frac=0.10, nthread=16, seed=0):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.eta = eta
        self.subsample = subsample
        self.colsample = colsample
        self.min_child_weight = min_child_weight
        self.num_boost_round = num_boost_round
        self.early_stop = early_stop
        self.val_frac = val_frac
        self.nthread = nthread
        self.seed = seed

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
    pred = bst.predict(dte) if bi is None else bst.predict(dte, iteration_range=(0, bi + 1))
    return pred.astype(np.float32)

def make_tabpfn(device: str, ignore_limits: bool):
    try:
        return TabPFNRegressor(device=device, ignore_pretraining_limits=ignore_limits)
    except TypeError:
        return TabPFNRegressor(device=device)

def fit_predict_tabpfn(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                       device: str, ignore_limits: bool, train_cap: int, seed: int) -> np.ndarray:
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
def transform_train_test(Xemb: np.ndarray, tr_idx: np.ndarray, te_idx: np.ndarray,
                         pca_dim: int, Xnum: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
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

def get_hoff(df: pd.DataFrame, Xemb_all: np.ndarray, add_nmut_features: bool):
    mask = (df["dataset_id"].values == "HOFF")
    h = df.loc[mask].copy()
    X = Xemb_all[mask]
    if "has_stop" in h.columns:
        good = ~h["has_stop"].fillna(False).astype(bool).to_numpy()
        h = h.loc[good].copy()
        X = X[good]
    h["n_mut"] = pd.to_numeric(h["n_mut"], errors="coerce")
    nmut = h["n_mut"].to_numpy()
    keep = np.isfinite(nmut)
    h = h.loc[keep].copy()
    X = X[keep]
    nmut = nmut[keep].astype(int)

    Xnum = None
    if add_nmut_features:
        nm = nmut.astype(np.float32)
        Xnum = np.vstack([nm, nm**2]).T.astype(np.float32)

    ids = h["variant_id"].astype(str).to_numpy()
    return ids, X, h, nmut, Xnum


# -------------------------
# Summaries and tests (publication-ready)
# -------------------------
def bootstrap_ci_mean(diff: np.ndarray, n_boot=20000, seed=0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return (np.nan, np.nan)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(diff, size=n, replace=True)
        boots.append(np.mean(samp))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)

def make_pub_outputs(runs_csv: str, out_dir: str):
    runs = pd.read_csv(runs_csv)

    # Define independent unit key
    unit_cols = ["dataset", "task_name", "target", "split", "split_id"]
    cfg_cols  = ["model", "pca_dim", "tabpfn_cap"]

    # Collapse model_seed replicates within each independent unit+config+model
    agg = (runs.groupby(unit_cols + cfg_cols, dropna=False)
                .agg(
                    spearman=("spearman", "mean"),
                    r2=("r2", "mean"),
                    top5_precision=("top5_precision", "mean"),
                    top5_enrich_diff=("top5_enrich_diff", "mean"),
                    n_runs=("spearman", "count"),
                    n_train=("n_train", "first"),
                    n_test=("n_test", "first"),
                )
                .reset_index())

    unit_path = os.path.join(out_dir, "unit_level.csv")
    agg.to_csv(unit_path, index=False)

    # Choose a fixed config to report (publication-friendly)
    # TabPFN: pca=128 cap=0; XGB: pca=128
    def fixed_select(dfsub):
        # tabpfn
        t = dfsub[(dfsub["model"] == "tabpfn") & (dfsub["pca_dim"] == 128) & (dfsub["tabpfn_cap"] == 0)].copy()
        # xgb
        x = dfsub[(dfsub["model"] == "xgb") & (dfsub["pca_dim"] == 128)].copy()
        # for xgb, tabpfn_cap is NaN; normalize so merge works
        x["tabpfn_cap"] = -1
        t["tabpfn_cap"] = -1
        return x, t

    # Also choose best config per model per task (optional “best-of-grid”)
    best_cfg_rows = []
    for key, dfsub in agg.groupby(["dataset","task_name","target","split"], dropna=False):
        for model in ["xgb","tabpfn"]:
            d = dfsub[dfsub["model"] == model].copy()
            if len(d) == 0:
                continue
            # maximize mean spearman across units
            g = d.groupby(["pca_dim","tabpfn_cap"], dropna=False)["spearman"].mean().reset_index()
            g = g.sort_values("spearman", ascending=False).iloc[0]
            best_cfg_rows.append({
                "dataset": key[0], "task_name": key[1], "target": key[2], "split": key[3],
                "model": model, "best_pca_dim": int(g["pca_dim"]),
                "best_tabpfn_cap": None if (model=="xgb") else (int(g["tabpfn_cap"]) if np.isfinite(g["tabpfn_cap"]) else None),
                "best_mean_spearman": float(g["spearman"]),
            })
    best_cfg = pd.DataFrame(best_cfg_rows)
    best_cfg.to_csv(os.path.join(out_dir, "best_config_by_task.csv"), index=False)

    # Summary table across independent units for fixed config (and best config)
    summary_rows = []
    paired_rows = []

    for (dataset, task_name, target, split), dfsub in agg.groupby(["dataset","task_name","target","split"], dropna=False):
        # ---- Fixed comparison ----
        x_fixed = dfsub[(dfsub.model=="xgb") & (dfsub.pca_dim==128)].copy()
        t_fixed = dfsub[(dfsub.model=="tabpfn") & (dfsub.pca_dim==128) & (dfsub.tabpfn_cap==0)].copy()

        # Join on unit (dataset/task/target/split/split_id)
        join_cols = ["dataset","task_name","target","split","split_id"]
        xj = x_fixed[join_cols + ["spearman","top5_precision","top5_enrich_diff"]].rename(columns={
            "spearman":"spearman_xgb",
            "top5_precision":"top5_precision_xgb",
            "top5_enrich_diff":"top5_enrich_diff_xgb",
        })
        tj = t_fixed[join_cols + ["spearman","top5_precision","top5_enrich_diff"]].rename(columns={
            "spearman":"spearman_tabpfn",
            "top5_precision":"top5_precision_tabpfn",
            "top5_enrich_diff":"top5_enrich_diff_tabpfn",
        })
        j = xj.merge(tj, on=join_cols, how="inner")
        if len(j) >= 3:
            diff = (j["spearman_tabpfn"] - j["spearman_xgb"]).to_numpy()
            t_p = float(ttest_rel(j["spearman_tabpfn"], j["spearman_xgb"]).pvalue)
            try:
                w_p = float(wilcoxon(diff).pvalue)
            except Exception:
                w_p = np.nan
            lo, hi = bootstrap_ci_mean(diff, n_boot=5000, seed=0)

            paired_rows.append({
                "dataset": dataset, "task_name": task_name, "target": target, "split": split,
                "n_units": int(len(j)),
                "mean_diff_spearman_tab_minus_xgb": float(np.mean(diff)),
                "median_diff_spearman": float(np.median(diff)),
                "ci95_low": lo, "ci95_high": hi,
                "paired_t_pvalue": t_p,
                "wilcoxon_pvalue": w_p,
            })

            # Figures (paired scatter + delta hist)
            fig_prefix = f"{dataset}_{task_name}_{target}_{split}".replace("/","_")
            # scatter
            plt.figure()
            plt.scatter(j["spearman_xgb"], j["spearman_tabpfn"])
            mn = min(j["spearman_xgb"].min(), j["spearman_tabpfn"].min())
            mx = max(j["spearman_xgb"].max(), j["spearman_tabpfn"].max())
            plt.plot([mn,mx],[mn,mx], linestyle="--")
            plt.xlabel("Spearman XGB (pca=128)")
            plt.ylabel("Spearman TabPFN (pca=128 cap=0)")
            plt.title(f"Paired units: {dataset} {task_name} {split} (n={len(j)})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"paired_scatter_{fig_prefix}.png"), dpi=200)
            plt.close()

            # delta hist
            plt.figure()
            plt.hist(diff, bins=12)
            plt.xlabel("Spearman(TabPFN) - Spearman(XGB)")
            plt.title(f"Delta Spearman: {dataset} {task_name} {split}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"delta_hist_{fig_prefix}.png"), dpi=200)
            plt.close()

        # Summary stats for fixed configs (unpaired, descriptive)
        def summarize_block(d, label):
            if len(d)==0:
                return
            summary_rows.append({
                "dataset":dataset, "task_name":task_name, "target":target, "split":split,
                "config_label":label,
                "model": d["model"].iloc[0],
                "pca_dim": int(d["pca_dim"].iloc[0]),
                "tabpfn_cap": None if d["model"].iloc[0]=="xgb" else int(d["tabpfn_cap"].iloc[0]),
                "n_units": int(d[join_cols].drop_duplicates().shape[0]),
                "spearman_mean": float(d["spearman"].mean()),
                "spearman_std": float(d["spearman"].std(ddof=1)) if len(d)>1 else np.nan,
                "top5_precision_mean": float(d["top5_precision"].mean()),
                "top5_enrich_diff_mean": float(d["top5_enrich_diff"].mean()),
            })

        summarize_block(x_fixed.assign(model="xgb"), "fixed")
        summarize_block(t_fixed.assign(model="tabpfn"), "fixed")

        # ---- Best-of-grid descriptive summary (no paired test, since configs differ) ----
        for model in ["xgb","tabpfn"]:
            bc = best_cfg[(best_cfg.dataset==dataset)&(best_cfg.task_name==task_name)&(best_cfg.target==target)&(best_cfg.split==split)&(best_cfg.model==model)]
            if len(bc)==0:
                continue
            pca_best = int(bc["best_pca_dim"].iloc[0])
            cap_best = bc["best_tabpfn_cap"].iloc[0]
            if model=="xgb":
                d_best = dfsub[(dfsub.model=="xgb")&(dfsub.pca_dim==pca_best)]
            else:
                d_best = dfsub[(dfsub.model=="tabpfn")&(dfsub.pca_dim==pca_best)&(dfsub.tabpfn_cap==cap_best)]
            summarize_block(d_best.assign(model=model), "best_grid")

    summary_df = pd.DataFrame(summary_rows)
    paired_df = pd.DataFrame(paired_rows)

    summary_df.to_csv(os.path.join(out_dir, "summary_pub.csv"), index=False)
    paired_df.to_csv(os.path.join(out_dir, "paired_tests.csv"), index=False)

    print("Wrote:", unit_path)
    print("Wrote:", os.path.join(out_dir, "summary_pub.csv"))
    print("Wrote:", os.path.join(out_dir, "paired_tests.csv"))
    print("Wrote plots: paired_scatter_*.png and delta_hist_*.png")


# -------------------------
# RUN mode: generate runs_raw.csv for (1) DMS 3 targets and (2) Hoff direct + derived delta
# -------------------------
def run_benchmark(args):
    if args.tabpfn_device == "cuda":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Run on GPU node or set --tabpfn_device cpu/auto.")

    os.makedirs(args.out_dir, exist_ok=True)

    embd = np.load(args.emb_npy, allow_pickle=True).item()
    ids_all = embd["ids"].astype(str)
    Xemb_all = embd["emb"].astype(np.float32)

    df = pd.read_csv(args.labels_csv, low_memory=False)
    df = df.set_index("variant_id").loc[ids_all].reset_index()

    pca_dims = [int(x) for x in args.pca_dims.split(",") if x.strip()]
    split_seeds = [int(x) for x in args.split_seeds.split(",") if x.strip()]
    model_seeds = [int(x) for x in args.model_seeds.split(",") if x.strip()]
    tabpfn_caps = [int(x) for x in args.tabpfn_caps.split(",") if x.strip()]

    runs = []

    # ----- DMS targets -----
    if args.datasets in ("DMS","BOTH"):
        dms_targets = [t.strip() for t in args.dms_targets.split(",") if t.strip()]
        for tgt in dms_targets:
            d_ids, d_X, d_y, d_pos = get_dms(df, Xemb_all, tgt)

            # within-position
            for sseed in split_seeds:
                tr, te = train_test_split(np.arange(len(d_y)), test_size=0.2, random_state=sseed, stratify=d_pos)
                for pca_dim in pca_dims:
                    Xtr_p, Xte_p = transform_train_test(d_X, tr, te, pca_dim=pca_dim, Xnum=None)
                    ytr, yte = d_y[tr], d_y[te]

                    # XGB (vary model_seed)
                    for mseed in model_seeds:
                        t0 = time.time()
                        pred = fit_predict_xgb(Xtr_p, ytr, Xte_p, XGBParams(
                            max_depth=args.xgb_max_depth, reg_lambda=args.xgb_reg_lambda,
                            num_boost_round=args.xgb_num_round, early_stop=args.xgb_early_stop,
                            val_frac=args.xgb_val_frac, nthread=args.xgb_nthread, seed=mseed
                        ))
                        met = eval_metrics(yte, pred)
                        runs.append({
                            "dataset":"DMS","task_name":"DMS","target":tgt,"split":"within_position","split_id":f"within_seed{sseed}",
                            "model":"xgb","pca_dim":pca_dim,"tabpfn_cap":np.nan,"model_seed":mseed,
                            "n_train":len(tr),"n_test":len(te), **met
                        })

                    # TabPFN
                    for cap in tabpfn_caps:
                        # cap==0 deterministic -> one run
                        seeds_for = [0] if cap==0 else model_seeds
                        for mseed in seeds_for:
                            t0 = time.time()
                            pred = fit_predict_tabpfn(Xtr_p, ytr, Xte_p,
                                                      device=args.tabpfn_device,
                                                      ignore_limits=args.tabpfn_ignore_limits,
                                                      train_cap=cap, seed=mseed)
                            met = eval_metrics(yte, pred)
                            runs.append({
                                "dataset":"DMS","task_name":"DMS","target":tgt,"split":"within_position","split_id":f"within_seed{sseed}",
                                "model":"tabpfn","pca_dim":pca_dim,"tabpfn_cap":cap,"model_seed":mseed,
                                "n_train":len(tr),"n_test":len(te), **met
                            })

            # pos-holdout CV
            uniq = np.unique(d_pos)
            n_splits = min(5, len(uniq))
            gkf = GroupKFold(n_splits=n_splits)
            for fold_i, (tr, te) in enumerate(gkf.split(np.zeros(len(d_y)), d_y, groups=d_pos), start=1):
                for pca_dim in pca_dims:
                    Xtr_p, Xte_p = transform_train_test(d_X, tr, te, pca_dim=pca_dim, Xnum=None)
                    ytr, yte = d_y[tr], d_y[te]

                    for mseed in model_seeds:
                        pred = fit_predict_xgb(Xtr_p, ytr, Xte_p, XGBParams(
                            max_depth=args.xgb_max_depth, reg_lambda=args.xgb_reg_lambda,
                            num_boost_round=args.xgb_num_round, early_stop=args.xgb_early_stop,
                            val_frac=args.xgb_val_frac, nthread=args.xgb_nthread, seed=mseed
                        ))
                        met = eval_metrics(yte, pred)
                        runs.append({
                            "dataset":"DMS","task_name":"DMS","target":tgt,"split":"pos_holdout","split_id":f"poscv_fold{fold_i}",
                            "model":"xgb","pca_dim":pca_dim,"tabpfn_cap":np.nan,"model_seed":mseed,
                            "n_train":len(tr),"n_test":len(te), **met
                        })

                    for cap in tabpfn_caps:
                        seeds_for = [fold_i] if cap==0 else model_seeds
                        for mseed in seeds_for:
                            pred = fit_predict_tabpfn(Xtr_p, ytr, Xte_p,
                                                      device=args.tabpfn_device,
                                                      ignore_limits=args.tabpfn_ignore_limits,
                                                      train_cap=cap, seed=mseed)
                            met = eval_metrics(yte, pred)
                            runs.append({
                                "dataset":"DMS","task_name":"DMS","target":tgt,"split":"pos_holdout","split_id":f"poscv_fold{fold_i}",
                                "model":"tabpfn","pca_dim":pca_dim,"tabpfn_cap":cap,"model_seed":mseed,
                                "n_train":len(tr),"n_test":len(te), **met
                            })

    # ----- HOFF direct + derived delta -----
    if args.datasets in ("HOFF","BOTH"):
        h_ids, h_X, h_df, h_nmut, h_Xnum = get_hoff(df, Xemb_all, add_nmut_features=args.hoff_add_nmut_features)

        # True labels
        y_delta = pd.to_numeric(h_df["hoff_delta_O2_minus_N2"], errors="coerce").to_numpy(dtype=np.float32)
        y_o2 = pd.to_numeric(h_df["hoff_fitness_O2"], errors="coerce").to_numpy(dtype=np.float32)
        y_n2 = pd.to_numeric(h_df["hoff_fitness_N2"], errors="coerce").to_numpy(dtype=np.float32)

        keep = np.isfinite(y_delta) & np.isfinite(y_o2) & np.isfinite(y_n2)
        h_X = h_X[keep]
        h_Xnum2 = h_Xnum[keep] if h_Xnum is not None else None
        y_delta = y_delta[keep]
        y_o2 = y_o2[keep]
        y_n2 = y_n2[keep]
        nmut = h_nmut[keep]

        base_tr = np.where(nmut <= 4)[0]
        base_va = np.where(nmut == 5)[0]
        base_te = np.where(nmut >= 6)[0]
        if len(base_te) < 50:
            base_te = np.where(nmut >= 5)[0]

        for mseed in model_seeds:
            if len(base_va) < 50:
                tr_idx, va_idx = train_test_split(base_tr, test_size=0.1, random_state=mseed)
            else:
                tr_idx, va_idx = base_tr, base_va
            te_idx = base_te

            for pca_dim in pca_dims:
                Xtr_p, Xte_p = transform_train_test(h_X, tr_idx, te_idx, pca_dim=pca_dim, Xnum=h_Xnum2)
                ytr_delta, yte_delta = y_delta[tr_idx], y_delta[te_idx]

                # Direct delta
                pred_xgb = fit_predict_xgb(Xtr_p, ytr_delta, Xte_p, XGBParams(
                    max_depth=args.xgb_max_depth, reg_lambda=args.xgb_reg_lambda,
                    num_boost_round=args.xgb_num_round, early_stop=args.xgb_early_stop,
                    val_frac=args.xgb_val_frac, nthread=args.xgb_nthread, seed=mseed
                ))
                met = eval_metrics(yte_delta, pred_xgb)
                runs.append({
                    "dataset":"HOFF","task_name":"HOFF_delta_direct","target":"hoff_delta_O2_minus_N2","split":"depth_holdout","split_id":f"depth_seed{mseed}",
                    "model":"xgb","pca_dim":pca_dim,"tabpfn_cap":np.nan,"model_seed":mseed,
                    "n_train":len(tr_idx),"n_test":len(te_idx), **met
                })

                for cap in tabpfn_caps:
                    seeds_for = [mseed] if cap>0 else [0]
                    for s2 in seeds_for:
                        pred_tab = fit_predict_tabpfn(Xtr_p, ytr_delta, Xte_p,
                                                      device=args.tabpfn_device,
                                                      ignore_limits=args.tabpfn_ignore_limits,
                                                      train_cap=cap, seed=s2)
                        met = eval_metrics(yte_delta, pred_tab)
                        runs.append({
                            "dataset":"HOFF","task_name":"HOFF_delta_direct","target":"hoff_delta_O2_minus_N2","split":"depth_holdout","split_id":f"depth_seed{mseed}",
                            "model":"tabpfn","pca_dim":pca_dim,"tabpfn_cap":cap,"model_seed":s2,
                            "n_train":len(tr_idx),"n_test":len(te_idx), **met
                        })

                # Derived delta = pred(O2) - pred(N2)
                # XGB
                pred_o2 = fit_predict_xgb(Xtr_p, y_o2[tr_idx], Xte_p, XGBParams(
                    max_depth=args.xgb_max_depth, reg_lambda=args.xgb_reg_lambda,
                    num_boost_round=args.xgb_num_round, early_stop=args.xgb_early_stop,
                    val_frac=args.xgb_val_frac, nthread=args.xgb_nthread, seed=mseed
                ))
                pred_n2 = fit_predict_xgb(Xtr_p, y_n2[tr_idx], Xte_p, XGBParams(
                    max_depth=args.xgb_max_depth, reg_lambda=args.xgb_reg_lambda,
                    num_boost_round=args.xgb_num_round, early_stop=args.xgb_early_stop,
                    val_frac=args.xgb_val_frac, nthread=args.xgb_nthread, seed=mseed
                ))
                pred_delta_derived = pred_o2 - pred_n2
                met = eval_metrics(yte_delta, pred_delta_derived)
                runs.append({
                    "dataset":"HOFF","task_name":"HOFF_delta_derived","target":"hoff_delta_O2_minus_N2","split":"depth_holdout","split_id":f"depth_seed{mseed}",
                    "model":"xgb","pca_dim":pca_dim,"tabpfn_cap":np.nan,"model_seed":mseed,
                    "n_train":len(tr_idx),"n_test":len(te_idx), **met
                })

                # TabPFN derived
                for cap in tabpfn_caps:
                    seeds_for = [mseed] if cap>0 else [0]
                    for s2 in seeds_for:
                        pred_o2_t = fit_predict_tabpfn(Xtr_p, y_o2[tr_idx], Xte_p,
                                                       device=args.tabpfn_device,
                                                       ignore_limits=args.tabpfn_ignore_limits,
                                                       train_cap=cap, seed=s2+123)
                        pred_n2_t = fit_predict_tabpfn(Xtr_p, y_n2[tr_idx], Xte_p,
                                                       device=args.tabpfn_device,
                                                       ignore_limits=args.tabpfn_ignore_limits,
                                                       train_cap=cap, seed=s2+456)
                        pred_delta_t = pred_o2_t - pred_n2_t
                        met = eval_metrics(yte_delta, pred_delta_t)
                        runs.append({
                            "dataset":"HOFF","task_name":"HOFF_delta_derived","target":"hoff_delta_O2_minus_N2","split":"depth_holdout","split_id":f"depth_seed{mseed}",
                            "model":"tabpfn","pca_dim":pca_dim,"tabpfn_cap":cap,"model_seed":s2,
                            "n_train":len(tr_idx),"n_test":len(te_idx), **met
                        })

    runs_df = pd.DataFrame(runs)
    out_path = os.path.join(args.out_dir, "runs_raw.csv")
    runs_df.to_csv(out_path, index=False)
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Wrote:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run","summarize"], required=True)

    ap.add_argument("--emb_npy", default="esm2_t33_650m_full.npy")
    ap.add_argument("--labels_csv", default="rubisco_datasets_merged.csv")

    ap.add_argument("--out_dir", default="results_pubready_xgb_tabpfn")
    ap.add_argument("--in_runs", default="", help="If set in summarize mode, path to runs_raw.csv (else out_dir/runs_raw.csv)")

    ap.add_argument("--datasets", choices=["DMS","HOFF","BOTH"], default="BOTH")

    # DMS
    ap.add_argument("--dms_targets", default="dms_enrichment_mean,dms_KmCO2_logfit,dms_VmaxRatio_logfit")
    ap.add_argument("--pca_dims", default="64,128,256")
    ap.add_argument("--split_seeds", default="0,1,2")
    ap.add_argument("--model_seeds", default="0,1,2,3,4")
    ap.add_argument("--tabpfn_caps", default="0,5000,2000")
    ap.add_argument("--tabpfn_device", default="cuda")
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

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "run":
        run_benchmark(args)
    else:
        runs_csv = args.in_runs if args.in_runs else os.path.join(args.out_dir, "runs_raw.csv")
        if not os.path.exists(runs_csv):
            raise RuntimeError(f"runs file not found: {runs_csv}")
        make_pub_outputs(runs_csv, args.out_dir)

if __name__ == "__main__":
    main()
