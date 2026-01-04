#!/usr/bin/env python3
import argparse, os, json, math, time, hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import xgboost as xgb
from tabpfn import TabPFNRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

# Torch (optional; required for CUDA availability check)
try:
    import torch
except Exception:
    torch = None


# --------------------------
# Utilities
# --------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_spearman(y, p) -> float:
    c = spearmanr(y, p).correlation
    return float(c) if np.isfinite(c) else np.nan

def stable_int_hash(s: str, mod: int = 2**31 - 1) -> int:
    """Deterministic integer hash (avoids Python's randomized hash() across processes)."""
    try:
        h = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).digest()
    except TypeError:
        # older Python/OpenSSL builds
        h = hashlib.md5(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False) % int(mod)

def _trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility wrapper: NumPy >= 2.0 uses trapezoid; fall back if needed."""
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))

def sanitize_preds(y_pred: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Replace non-finite predictions and clip to the (finite) training range."""
    if y_pred.size == 0:
        return y_pred
    if np.all(np.isfinite(y_pred)):
        return y_pred

    yt = np.asarray(y_train)
    yt = yt[np.isfinite(yt)]
    if yt.size == 0:
        # last-resort: just make finite
        return np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0).astype(y_pred.dtype, copy=False)

    med = float(np.median(yt))
    lo = float(np.min(yt))
    hi = float(np.max(yt))
    out = np.nan_to_num(y_pred, nan=med, posinf=hi, neginf=lo)
    out = np.clip(out, lo, hi)
    return out.astype(y_pred.dtype, copy=False)

def build_top_set(y: np.ndarray, frac: float, mask: Optional[np.ndarray] = None) -> set:
    idx = np.arange(len(y))
    if mask is not None:
        idx = idx[mask]
    k = max(1, int(np.ceil(frac * len(idx))))
    sub = y[idx]
    top_idx = idx[np.argsort(sub)[-k:]]
    return set(top_idx.tolist())

def topk_recovery(discovered: set, top_set: set) -> float:
    return len(discovered & top_set) / max(1, len(top_set))

def median_min_dist(X: np.ndarray, new_idx: np.ndarray, ref_idx: np.ndarray) -> float:
    if len(ref_idx) == 0 or len(new_idx) == 0:
        return np.nan
    A = X[new_idx]
    R = X[ref_idx]
    a2 = np.sum(A*A, axis=1, keepdims=True)
    r2 = np.sum(R*R, axis=1, keepdims=True).T
    d2 = a2 + r2 - 2.0 * (A @ R.T)
    d2 = np.maximum(d2, 0.0)
    md = np.sqrt(np.min(d2, axis=1))
    return float(np.median(md))

def position_coverage(pos: np.ndarray, discovered_idx: np.ndarray) -> float:
    all_pos = set(pos.tolist())
    seen = set(pos[discovered_idx].tolist())
    return len(seen) / max(1, len(all_pos))

def auc_best_so_far(n_assayed: np.ndarray, best: np.ndarray) -> float:
    # trapezoid AUC over assays axis; assumes sorted n_assayed
    if len(n_assayed) < 2:
        return np.nan
    order = np.argsort(n_assayed)
    x = n_assayed[order].astype(float)
    y = best[order].astype(float)
    return _trapezoid_area(y, x)

def make_tabpfn(device: str, ignore_limits: bool):
    try:
        return TabPFNRegressor(device=device, ignore_pretraining_limits=ignore_limits)
    except TypeError:
        return TabPFNRegressor(device=device)


# --------------------------
# Feature prep (unsupervised on full pool; allowed in AL)
# --------------------------
@dataclass
class Feats:
    X_model: np.ndarray   # used for training/prediction
    X_novel: np.ndarray   # used for novelty distances
    meta: pd.DataFrame

def build_features(
    Xemb: np.ndarray,
    meta: pd.DataFrame,
    pca_dim: int,
    add_nmut: bool,
) -> Feats:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xemb.astype(np.float32))
    pca = PCA(n_components=int(pca_dim), random_state=0, svd_solver="randomized")
    Xp = pca.fit_transform(Xs).astype(np.float32)

    X_novel = Xp

    if add_nmut:
        nm = pd.to_numeric(meta["n_mut"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        Xnum = np.vstack([nm, nm**2]).T
        scn = StandardScaler()
        Xnum_s = scn.fit_transform(Xnum).astype(np.float32)
        X_model = np.hstack([Xp, Xnum_s]).astype(np.float32)
    else:
        X_model = Xp

    return Feats(X_model=X_model, X_novel=X_novel, meta=meta)


# --------------------------
# Surrogate training: XGB ensemble
# --------------------------
@dataclass
class XGBEnsembleCfg:
    ensemble_size: int = 5
    num_boost_round: int = 2000
    early_stop: int = 100
    val_frac: float = 0.10
    max_depth: int = 6
    reg_lambda: float = 10.0
    eta: float = 0.03
    subsample: float = 0.85
    colsample: float = 0.85
    nthread: int = 16

def xgb_ensemble_predict(
    X: np.ndarray,
    y: np.ndarray,
    labeled_idx: np.ndarray,
    pool_idx: np.ndarray,
    w: Optional[np.ndarray],
    cfg: XGBEnsembleCfg,
    rng: np.random.Generator,
    seed_base: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(pool_idx) == 0:
        return np.array([]), np.array([])

    preds = np.zeros((cfg.ensemble_size, len(pool_idx)), dtype=np.float32)
    dpool = xgb.DMatrix(X[pool_idx])

    for m in range(cfg.ensemble_size):
        rs = int((seed_base + 10007*m) % (2**31 - 1))
        # bootstrap
        boot = rng.choice(labeled_idx, size=len(labeled_idx), replace=True)
        tr_b, va_b = train_test_split(boot, test_size=cfg.val_frac, random_state=rs)

        dtr = xgb.DMatrix(X[tr_b], label=y[tr_b], weight=(w[tr_b] if w is not None else None))
        dva = xgb.DMatrix(X[va_b], label=y[va_b], weight=(w[va_b] if w is not None else None))

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": cfg.eta,
            "max_depth": int(cfg.max_depth),
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample,
            "lambda": cfg.reg_lambda,
            "tree_method": "hist",
            "seed": rs,
            "nthread": int(cfg.nthread),
        }
        bst = xgb.train(
            params, dtr,
            num_boost_round=int(cfg.num_boost_round),
            evals=[(dva, "val")],
            early_stopping_rounds=int(cfg.early_stop),
            verbose_eval=False,
        )
        bi = bst.best_iteration
        pred = bst.predict(dpool) if bi is None else bst.predict(dpool, iteration_range=(0, bi + 1))
        preds[m, :] = pred.astype(np.float32)

    return preds.mean(axis=0), preds.std(axis=0)


# --------------------------
# Surrogate training: TabPFN ensemble (subsample ensemble for uncertainty)
# --------------------------
@dataclass
class TabPFNEnsembleCfg:
    ensemble_size: int = 5
    train_cap: int = 5000
    device: str = "cuda"
    ignore_limits: bool = True

def tabpfn_ensemble_predict(
    X: np.ndarray,
    y: np.ndarray,
    labeled_idx: np.ndarray,
    pool_idx: np.ndarray,
    cfg: TabPFNEnsembleCfg,
    rng: np.random.Generator,
    seed_base: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(pool_idx) == 0:
        return np.array([]), np.array([])
    preds = np.zeros((cfg.ensemble_size, len(pool_idx)), dtype=np.float32)

    # Extract labeled training set once
    Xlab = X[labeled_idx]
    ylab = y[labeled_idx]
    n = len(ylab)

    for m in range(cfg.ensemble_size):
        rs = int((seed_base + 20011*m) % (2**31 - 1))
        idx = np.arange(n)
        if cfg.train_cap and cfg.train_cap > 0 and n > cfg.train_cap:
            rr = np.random.default_rng(rs)
            idx = rr.choice(idx, size=cfg.train_cap, replace=False)

        reg = make_tabpfn(cfg.device, cfg.ignore_limits)
        reg.fit(Xlab[idx], ylab[idx])

        # Suppress numpy runtime warnings inside TabPFN transforms; sanitize after predict.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred = reg.predict(X[pool_idx]).astype(np.float32)

        pred = sanitize_preds(pred, ylab[idx])
        preds[m, :] = pred

    return preds.mean(axis=0), preds.std(axis=0)


# --------------------------
# Acquisition policies
# --------------------------
def pick_batch(strategy: str, pool_idx: np.ndarray, mean_obj: np.ndarray, std_obj: np.ndarray,
               rng: np.random.Generator, batch_size: int, beta: float) -> np.ndarray:
    if len(pool_idx) == 0:
        return np.array([], dtype=int)

    if strategy == "random":
        return rng.choice(pool_idx, size=min(batch_size, len(pool_idx)), replace=False).astype(int)

    if strategy == "greedy":
        score = mean_obj
    elif strategy == "uncertainty":
        score = std_obj
    elif strategy == "ucb":
        score = mean_obj + beta * std_obj
    elif strategy == "thompson":
        score = mean_obj + std_obj * rng.standard_normal(len(mean_obj))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    k = min(batch_size, len(pool_idx))
    top_local = np.argpartition(score, -k)[-k:]
    top_local = top_local[np.argsort(score[top_local])[::-1]]
    return pool_idx[top_local].astype(int)


# --------------------------
# Simulation
# --------------------------
@dataclass
class SimCfg:
    n_reps: int = 20
    init_n: int = 200
    batch_size: int = 48
    max_rounds: int = 25
    beta: float = 1.0
    novelty_cutoff_frac: float = 0.20
    novelty_patience: int = 3
    novelty_min_rounds: int = 5
    dms_pos_coverage_stop: float = 0.95
    top_fracs: Tuple[float, float] = (0.01, 0.05)

def simulate_dataset(
    dataset_name: str,
    X_model: np.ndarray,
    X_novel: np.ndarray,
    y_obj_true: np.ndarray,
    # Hoffmann constraint (true + threshold); None for DMS
    y_constraint_true: Optional[np.ndarray],
    constraint_threshold_true: Optional[float],
    # evaluation top sets mask (feasible subset)
    feasible_mask: np.ndarray,
    # DMS coverage
    dms_positions: Optional[np.ndarray],
    strategies: List[str],
    surrogate: str,  # "xgb" or "tabpfn"
    simcfg: SimCfg,
    xgb_cfg_obj: Optional[XGBEnsembleCfg],
    xgb_cfg_c: Optional[XGBEnsembleCfg],
    tab_cfg_obj: Optional[TabPFNEnsembleCfg],
    tab_cfg_c: Optional[TabPFNEnsembleCfg],
    # weights for XGB (optional)
    w_obj: Optional[np.ndarray],
    w_c: Optional[np.ndarray],
    constraint_mode: str,
    constraint_lambda: float,
    seed0: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      rounds_df: per-round traces
      run_df: per-run summaries (time-to-hit, AUC, stop reason)
    """
    n = len(y_obj_true)
    pool_all = np.arange(n)

    top_sets = {frac: build_top_set(y_obj_true, frac=frac, mask=feasible_mask) for frac in simcfg.top_fracs}

    rounds_rows = []
    run_rows = []

    for rep in range(simcfg.n_reps):
        base_seed = seed0 + 100000*rep
        rng_init = np.random.default_rng(base_seed)
        init = rng_init.choice(pool_all, size=min(simcfg.init_n, n), replace=False).astype(int)

        # identical init across strategies (fair within surrogate)
        init_set = set(init.tolist())

        for strategy in strategies:
            seed_offset = stable_int_hash(f"{dataset_name}|{surrogate}|{strategy}", mod=100000)
            rng = np.random.default_rng(base_seed + seed_offset)

            labeled = set(init_set)
            discovered = set(init_set)

            baseline_novel = None
            low_novel_streak = 0
            stop_reason = ""

            # record round 0
            disc_arr = np.array(sorted(discovered), dtype=int)
            feasible_disc = disc_arr
            if y_constraint_true is not None and constraint_threshold_true is not None:
                feasible_disc = disc_arr[y_constraint_true[disc_arr] >= constraint_threshold_true]

            best_so_far = float(np.max(y_obj_true[feasible_disc])) if len(feasible_disc) else np.nan
            cov = position_coverage(dms_positions, disc_arr) if dms_positions is not None else np.nan
            rec = {f"recovery_top_{int(frac*100)}pct": topk_recovery(discovered, top_sets[frac]) for frac in simcfg.top_fracs}

            rounds_rows.append({
                "dataset": dataset_name,
                "surrogate": surrogate,
                "strategy": strategy,
                "rep": rep,
                "round": 0,
                "n_assayed": int(len(disc_arr)),
                "best_so_far": best_so_far,
                "novelty_median": np.nan,
                "baseline_novelty": np.nan,
                "low_novelty_streak": 0,
                "stop_reason": "",
                "dms_pos_coverage": cov,
                **rec
            })

            # track first-hit times
            first_hit = {frac: None for frac in simcfg.top_fracs}

            for t in range(1, simcfg.max_rounds + 1):
                labeled_idx = np.array(sorted(labeled), dtype=int)
                pool_idx = pool_all[~np.isin(pool_all, labeled_idx)]

                if len(pool_idx) == 0:
                    stop_reason = "pool_exhausted"
                    break

                # ---------- choose batch ----------
                if strategy == "random":
                    chosen = pick_batch("random", pool_idx, None, None, rng, simcfg.batch_size, simcfg.beta)

                else:
                    # get surrogate predictions
                    if surrogate == "xgb":
                        mean_obj, std_obj = xgb_ensemble_predict(
                            X_model, y_obj_true, labeled_idx, pool_idx, w_obj, xgb_cfg_obj, rng, base_seed + 1000*t
                        )
                        if y_constraint_true is not None:
                            mean_c, std_c = xgb_ensemble_predict(
                                X_model, y_constraint_true, labeled_idx, pool_idx, w_c, xgb_cfg_c, rng, base_seed + 2000*t
                            )
                        else:
                            mean_c = std_c = None

                    elif surrogate == "tabpfn":
                        mean_obj, std_obj = tabpfn_ensemble_predict(
                            X_model, y_obj_true, labeled_idx, pool_idx, tab_cfg_obj, rng, base_seed + 3000*t
                        )
                        if y_constraint_true is not None:
                            mean_c, std_c = tabpfn_ensemble_predict(
                                X_model, y_constraint_true, labeled_idx, pool_idx, tab_cfg_c, rng, base_seed + 4000*t
                            )
                        else:
                            mean_c = std_c = None
                    else:
                        raise ValueError("surrogate must be xgb or tabpfn")

                    # constraint handling in acquisition (only Hoffmann)
                    if y_constraint_true is not None and constraint_threshold_true is not None:
                        if constraint_mode == "hard":
                            feas = (mean_c >= constraint_threshold_true)
                            cand_local = np.where(feas)[0]
                            if len(cand_local) < simcfg.batch_size:
                                # fallback: widen to top by predicted constraint
                                k = min(len(pool_idx), max(simcfg.batch_size*5, simcfg.batch_size))
                                cand_local = np.argsort(mean_c)[-k:]
                            pool2 = pool_idx[cand_local]
                            m2 = mean_obj[cand_local]
                            s2 = std_obj[cand_local]
                            chosen = pick_batch(strategy, pool2, m2, s2, rng, simcfg.batch_size, simcfg.beta)
                        elif constraint_mode == "soft":
                            penalty = np.maximum(0.0, constraint_threshold_true - mean_c)
                            m_adj = mean_obj - constraint_lambda * penalty
                            chosen = pick_batch(strategy, pool_idx, m_adj, std_obj, rng, simcfg.batch_size, simcfg.beta)
                        else:
                            raise ValueError("constraint_mode must be hard or soft")
                    else:
                        chosen = pick_batch(strategy, pool_idx, mean_obj, std_obj, rng, simcfg.batch_size, simcfg.beta)

                chosen = np.array(chosen, dtype=int)
                if len(chosen) == 0:
                    stop_reason = "no_candidates"
                    break

                # novelty vs already labeled
                nov = median_min_dist(X_novel, chosen, labeled_idx)
                if baseline_novel is None and np.isfinite(nov):
                    baseline_novel = nov

                if baseline_novel is not None and np.isfinite(nov) and t >= simcfg.novelty_min_rounds:
                    if nov < simcfg.novelty_cutoff_frac * baseline_novel:
                        low_novel_streak += 1
                    else:
                        low_novel_streak = 0

                # update sets
                for i in chosen.tolist():
                    labeled.add(int(i))
                    discovered.add(int(i))

                disc_arr = np.array(sorted(discovered), dtype=int)

                # evaluation: feasible by TRUE constraint if present
                feasible_disc = disc_arr
                feas_rate = np.nan
                if y_constraint_true is not None and constraint_threshold_true is not None:
                    feas_mask_disc = (y_constraint_true[disc_arr] >= constraint_threshold_true)
                    feasible_disc = disc_arr[feas_mask_disc]
                    # feasible rate in the newly chosen batch
                    feas_rate = float(np.mean(y_constraint_true[chosen] >= constraint_threshold_true))

                best_so_far = float(np.max(y_obj_true[feasible_disc])) if len(feasible_disc) else np.nan
                cov = position_coverage(dms_positions, disc_arr) if dms_positions is not None else np.nan
                rec = {f"recovery_top_{int(frac*100)}pct": topk_recovery(discovered, top_sets[frac]) for frac in simcfg.top_fracs}

                # first-hit times
                for frac in simcfg.top_fracs:
                    if first_hit[frac] is None and rec[f"recovery_top_{int(frac*100)}pct"] > 0:
                        first_hit[frac] = int(len(disc_arr))

                rounds_rows.append({
                    "dataset": dataset_name,
                    "surrogate": surrogate,
                    "strategy": strategy,
                    "rep": rep,
                    "round": t,
                    "n_assayed": int(len(disc_arr)),
                    "best_so_far": best_so_far,
                    "novelty_median": nov,
                    "baseline_novelty": baseline_novel if baseline_novel is not None else np.nan,
                    "low_novelty_streak": int(low_novel_streak),
                    "stop_reason": "",
                    "dms_pos_coverage": cov,
                    "feasible_rate_batch": feas_rate,
                    **rec
                })

                # stop rules
                if dms_positions is not None and np.isfinite(cov) and cov >= simcfg.dms_pos_coverage_stop:
                    stop_reason = "dms_pos_coverage_reached"
                    break
                if low_novel_streak >= simcfg.novelty_patience:
                    stop_reason = "novelty_exhausted"
                    break

            # mark stop reason on last row for this run
            if stop_reason:
                for i in range(len(rounds_rows)-1, -1, -1):
                    r = rounds_rows[i]
                    if r["dataset"]==dataset_name and r["surrogate"]==surrogate and r["strategy"]==strategy and r["rep"]==rep:
                        rounds_rows[i]["stop_reason"] = stop_reason
                        break

            # run-level summary
            sub = [r for r in rounds_rows if r["dataset"]==dataset_name and r["surrogate"]==surrogate and r["strategy"]==strategy and r["rep"]==rep]
            sub = sorted(sub, key=lambda x: x["n_assayed"])
            n_ass = np.array([r["n_assayed"] for r in sub], dtype=int)
            best = np.array([r["best_so_far"] for r in sub], dtype=float)

            run_rows.append({
                "dataset": dataset_name,
                "surrogate": surrogate,
                "strategy": strategy,
                "rep": rep,
                "stop_reason": stop_reason,
                "n_final": int(n_ass[-1]) if len(n_ass) else 0,
                "auc_best_so_far": auc_best_so_far(n_ass, best),
                "assays_to_hit_top1pct": first_hit[0.01],
                "assays_to_hit_top5pct": first_hit[0.05],
            })

    return pd.DataFrame(rounds_rows), pd.DataFrame(run_rows)


# --------------------------
# Plotting + summaries
# --------------------------
def plot_trajectories(rounds_df: pd.DataFrame, outdir: str, dataset: str):
    ensure_dir(outdir)
    sub = rounds_df[rounds_df["dataset"]==dataset].copy()
    if len(sub)==0:
        return

    metrics = ["best_so_far", "recovery_top_1pct", "recovery_top_5pct", "novelty_median"]
    for surrogate in sorted(sub["surrogate"].unique()):
        for metric in metrics:
            plt.figure()
            for strat in sorted(sub["strategy"].unique()):
                ss = sub[(sub["surrogate"]==surrogate)&(sub["strategy"]==strat)].copy()
                if len(ss)==0:
                    continue

                # align by n_assayed grid
                grid = sorted(ss["n_assayed"].unique())
                reps = sorted(ss["rep"].unique())
                mat = np.full((len(reps), len(grid)), np.nan, dtype=float)
                for i,r in enumerate(reps):
                    rs = ss[ss["rep"]==r].sort_values("n_assayed")
                    vals = dict(zip(rs["n_assayed"], rs[metric]))
                    last = np.nan
                    for j,a in enumerate(grid):
                        if a in vals:
                            last = vals[a]
                        mat[i,j] = last
                mean = np.nanmean(mat, axis=0)
                std  = np.nanstd(mat, axis=0)
                plt.plot(grid, mean, label=strat)
                plt.fill_between(grid, mean-std, mean+std, alpha=0.2)

            plt.xlabel("Assays (cumulative)")
            plt.ylabel(metric)
            plt.title(f"{dataset} | {surrogate} | {metric}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{dataset}_{surrogate}_{metric}.png"), dpi=200)
            plt.close()

def summarize_runs(run_df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    ensure_dir(outdir)
    # summarize time-to-hit with medians and bootstrap CI
    rows = []
    rng = np.random.default_rng(0)

    def boot_ci(arr, n_boot=5000):
        arr = np.array([x for x in arr if x is not None and not (isinstance(x,float) and np.isnan(x))], dtype=float)
        if len(arr)==0:
            return (np.nan, np.nan)
        boots = []
        for _ in range(n_boot):
            samp = rng.choice(arr, size=len(arr), replace=True)
            boots.append(np.nanmedian(samp))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(lo), float(hi)

    for (dataset, surrogate, strategy), g in run_df.groupby(["dataset","surrogate","strategy"]):
        arr1 = g["assays_to_hit_top1pct"].dropna().to_numpy(dtype=float)
        arr5 = g["assays_to_hit_top5pct"].dropna().to_numpy(dtype=float)
        med1 = float(np.nanmedian(arr1)) if len(arr1) else np.nan
        med5 = float(np.nanmedian(arr5)) if len(arr5) else np.nan
        lo1, hi1 = boot_ci(arr1)
        lo5, hi5 = boot_ci(arr5)

        rows.append({
            "dataset": dataset,
            "surrogate": surrogate,
            "strategy": strategy,
            "n_reps": int(len(g)),
            "median_assays_to_hit_top1pct": med1,
            "ci95_top1pct_low": lo1,
            "ci95_top1pct_high": hi1,
            "median_assays_to_hit_top5pct": med5,
            "ci95_top5pct_low": lo5,
            "ci95_top5pct_high": hi5,
            "auc_best_so_far_mean": float(g["auc_best_so_far"].mean()),
            "auc_best_so_far_std": float(g["auc_best_so_far"].std(ddof=1)) if len(g)>1 else np.nan,
            "stop_reason_mode": g["stop_reason"].mode().iloc[0] if len(g["stop_reason"].mode()) else "",
        })

    summ = pd.DataFrame(rows).sort_values(["dataset","surrogate","strategy"])
    summ.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    return summ

def _read_concat_dedupe(path: str, new_df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    if new_df is None or len(new_df) == 0:
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()
    if not os.path.exists(path):
        return new_df
    try:
        old = pd.read_csv(path)
        comb = pd.concat([old, new_df], ignore_index=True)
        comb = comb.drop_duplicates(subset=subset, keep="last")
        return comb
    except Exception:
        return new_df


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npy", default="esm2_t33_650m_full.npy")
    ap.add_argument("--labels_csv", default="rubisco_datasets_merged.csv")
    ap.add_argument("--out_dir", default="results_active_learning_pubready")

    ap.add_argument("--datasets", choices=["DMS","HOFF","BOTH"], default="BOTH")
    ap.add_argument("--surrogates", choices=["XGB","TABPFN","BOTH"], default="BOTH")
    ap.add_argument("--strategies", default="random,greedy,uncertainty,ucb,thompson")

    # DMS
    ap.add_argument("--dms_target", default="dms_enrichment_mean")
    ap.add_argument("--dms_pca_dim", type=int, default=128)

    # Hoffmann
    ap.add_argument("--hoff_objective", choices=["delta_direct","delta_derived"], default="delta_direct")
    ap.add_argument("--hoff_pca_dim", type=int, default=128)
    ap.add_argument("--hoff_add_nmut_features", action="store_true")
    ap.add_argument("--hoff_n2_margin", type=float, default=0.10)
    ap.add_argument("--constraint_mode", choices=["hard","soft"], default="hard")
    ap.add_argument("--constraint_lambda", type=float, default=5.0)

    # Simulation
    ap.add_argument("--n_reps", type=int, default=20)
    ap.add_argument("--init_n", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--max_rounds", type=int, default=25)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--novelty_cutoff_frac", type=float, default=0.20)
    ap.add_argument("--novelty_patience", type=int, default=3)
    ap.add_argument("--novelty_min_rounds", type=int, default=5)
    ap.add_argument("--dms_pos_coverage_stop", type=float, default=0.95)

    # XGB ensemble settings
    ap.add_argument("--xgb_ens", type=int, default=5)
    ap.add_argument("--xgb_rounds", type=int, default=2000)
    ap.add_argument("--xgb_early_stop", type=int, default=100)
    ap.add_argument("--xgb_val_frac", type=float, default=0.10)
    ap.add_argument("--xgb_max_depth", type=int, default=6)
    ap.add_argument("--xgb_reg_lambda", type=float, default=10.0)
    ap.add_argument("--xgb_nthread", type=int, default=16)

    # TabPFN ensemble settings
    ap.add_argument("--tabpfn_ens", type=int, default=5)
    ap.add_argument("--tabpfn_cap", type=int, default=5000)
    ap.add_argument("--tabpfn_device", default="cuda")
    ap.add_argument("--tabpfn_ignore_limits", action="store_true")

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "plots"))

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    surrogates = []
    if args.surrogates in ("XGB","BOTH"): surrogates.append("xgb")
    if args.surrogates in ("TABPFN","BOTH"): surrogates.append("tabpfn")

    # GPU check if using TabPFN cuda
    if "tabpfn" in surrogates and args.tabpfn_device == "cuda":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("TabPFN requires CUDA here, but CUDA is not available. Run on GPU node or set --tabpfn_device cpu.")

    # Load embeddings + labels aligned
    embd = np.load(args.emb_npy, allow_pickle=True).item()
    ids_all = embd["ids"].astype(str)
    Xemb_all = embd["emb"].astype(np.float32)

    df = pd.read_csv(args.labels_csv, low_memory=False)
    df = df.set_index("variant_id").loc[ids_all].reset_index()

    # Config objects
    simcfg = SimCfg(
        n_reps=args.n_reps,
        init_n=args.init_n,
        batch_size=args.batch_size,
        max_rounds=args.max_rounds,
        beta=args.beta,
        novelty_cutoff_frac=args.novelty_cutoff_frac,
        novelty_patience=args.novelty_patience,
        novelty_min_rounds=args.novelty_min_rounds,
        dms_pos_coverage_stop=args.dms_pos_coverage_stop,
    )

    xgb_cfg_obj = XGBEnsembleCfg(
        ensemble_size=args.xgb_ens,
        num_boost_round=args.xgb_rounds,
        early_stop=args.xgb_early_stop,
        val_frac=args.xgb_val_frac,
        max_depth=args.xgb_max_depth,
        reg_lambda=args.xgb_reg_lambda,
        nthread=args.xgb_nthread,
    )
    xgb_cfg_c = xgb_cfg_obj

    tab_cfg_obj = TabPFNEnsembleCfg(
        ensemble_size=args.tabpfn_ens,
        train_cap=args.tabpfn_cap,
        device=args.tabpfn_device,
        ignore_limits=args.tabpfn_ignore_limits,
    )
    tab_cfg_c = tab_cfg_obj

    all_rounds = []
    all_runs = []

    # ---------- DMS ----------
    if args.datasets in ("DMS","BOTH"):
        mask = (df["dataset_id"].values == "DMS")
        d = df.loc[mask].copy()
        Xd = Xemb_all[mask]
        y = pd.to_numeric(d[args.dms_target], errors="coerce").to_numpy(dtype=np.float32)
        pos = pd.to_numeric(d["position_external"], errors="coerce").to_numpy()
        keep = np.isfinite(y) & np.isfinite(pos)
        d = d.loc[keep].copy()
        Xd = Xd[keep]
        y = y[keep]
        pos = pos[keep].astype(int)

        feats = build_features(Xd, d.assign(n_mut=1), pca_dim=args.dms_pca_dim, add_nmut=False)
        feasible_mask = np.ones(len(y), dtype=bool)

        for surrogate in surrogates:
            r_df, run_df = simulate_dataset(
                dataset_name=f"DMS_{args.dms_target}",
                X_model=feats.X_model,
                X_novel=feats.X_novel,
                y_obj_true=y,
                y_constraint_true=None,
                constraint_threshold_true=None,
                feasible_mask=feasible_mask,
                dms_positions=pos,
                strategies=strategies,
                surrogate=surrogate,
                simcfg=simcfg,
                xgb_cfg_obj=xgb_cfg_obj,
                xgb_cfg_c=None,
                tab_cfg_obj=tab_cfg_obj,
                tab_cfg_c=None,
                w_obj=None,
                w_c=None,
                constraint_mode=args.constraint_mode,
                constraint_lambda=args.constraint_lambda,
                seed0=0
            )
            all_rounds.append(r_df)
            all_runs.append(run_df)

    # ---------- HOFF ----------
    if args.datasets in ("HOFF","BOTH"):
        mask = (df["dataset_id"].values == "HOFF")
        h = df.loc[mask].copy()
        Xh = Xemb_all[mask]

        if "has_stop" in h.columns:
            has_stop = h["has_stop"].astype("boolean").fillna(False)
            good = (~has_stop).to_numpy(dtype=bool)
            h = h.loc[good].copy()
            Xh = Xh[good]

        # Coerce required columns to numeric once, then filter consistently.
        h["n_mut"] = pd.to_numeric(h["n_mut"], errors="coerce")
        h["hoff_delta_O2_minus_N2"] = pd.to_numeric(h["hoff_delta_O2_minus_N2"], errors="coerce")
        h["hoff_fitness_O2"] = pd.to_numeric(h["hoff_fitness_O2"], errors="coerce")
        h["hoff_fitness_N2"] = pd.to_numeric(h["hoff_fitness_N2"], errors="coerce")

        keep = (
            np.isfinite(h["n_mut"].to_numpy())
            & np.isfinite(h["hoff_delta_O2_minus_N2"].to_numpy())
            & np.isfinite(h["hoff_fitness_O2"].to_numpy())
            & np.isfinite(h["hoff_fitness_N2"].to_numpy())
        )
        h = h.loc[keep].copy()
        Xh = Xh[keep]

        nmut = h["n_mut"].to_numpy(dtype=np.float32)
        y_delta = h["hoff_delta_O2_minus_N2"].to_numpy(dtype=np.float32)
        y_o2 = h["hoff_fitness_O2"].to_numpy(dtype=np.float32)
        y_n2 = h["hoff_fitness_N2"].to_numpy(dtype=np.float32)

        # True constraint threshold from WT (n_mut==0)
        wt_mask = (nmut == 0)
        wt_n2 = float(np.nanmedian(y_n2[wt_mask])) if np.any(wt_mask) else np.nan
        if np.isfinite(wt_n2):
            c_thresh = float(wt_n2 - args.hoff_n2_margin)
        else:
            c_thresh = float(np.nanmedian(y_n2))

        # objective truth and feasible set
        if args.hoff_objective == "delta_direct":
            y_obj = y_delta
        else:
            # keep y_obj_true = true delta for evaluation
            y_obj = y_delta

        feasible_mask = (y_n2 >= c_thresh)

        feats = build_features(Xh, h.assign(n_mut=nmut), pca_dim=args.hoff_pca_dim, add_nmut=args.hoff_add_nmut_features)

        # weights for XGB (optional; keep simple uniform here for AL fairness)
        w = None

        for surrogate in surrogates:
            r_df, run_df = simulate_dataset(
                dataset_name=f"HOFF_{args.hoff_objective}",
                X_model=feats.X_model,
                X_novel=feats.X_novel,
                y_obj_true=y_obj,
                y_constraint_true=y_n2,
                constraint_threshold_true=c_thresh,
                feasible_mask=feasible_mask,
                dms_positions=None,
                strategies=strategies,
                surrogate=surrogate,
                simcfg=simcfg,
                xgb_cfg_obj=xgb_cfg_obj,
                xgb_cfg_c=xgb_cfg_c,
                tab_cfg_obj=tab_cfg_obj,
                tab_cfg_c=tab_cfg_c,
                w_obj=w,
                w_c=w,
                constraint_mode=args.constraint_mode,
                constraint_lambda=args.constraint_lambda,
                seed0=777
            )
            all_rounds.append(r_df)
            all_runs.append(run_df)

        # save constraint info (same regardless of surrogate)
        with open(os.path.join(args.out_dir, "hoff_constraint.json"), "w") as f:
            json.dump({
                "wt_n2_obs": None if not np.isfinite(wt_n2) else float(wt_n2),
                "n2_margin": float(args.hoff_n2_margin),
                "constraint_threshold": float(c_thresh),
                "feasible_frac": float(np.mean(feasible_mask)),
            }, f, indent=2)

    rounds_df_new = pd.concat(all_rounds, ignore_index=True) if all_rounds else pd.DataFrame()
    run_df_new = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()

    rounds_path = os.path.join(args.out_dir, "rounds.csv")
    run_path = os.path.join(args.out_dir, "run_level.csv")

    # If splitting surrogates across separate jobs into the SAME out_dir, merge/dedupe instead of overwriting.
    rounds_df = _read_concat_dedupe(rounds_path, rounds_df_new, subset=["dataset","surrogate","strategy","rep","round"])
    run_df = _read_concat_dedupe(run_path, run_df_new, subset=["dataset","surrogate","strategy","rep"])

    rounds_df.to_csv(rounds_path, index=False)
    run_df.to_csv(run_path, index=False)

    # Write per-surrogate config so split runs don't overwrite each other.
    with open(os.path.join(args.out_dir, f"config_{args.surrogates}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # plots + summary (use merged dfs so second job produces combined outputs)
    if len(rounds_df):
        for dataset in sorted(rounds_df["dataset"].unique()):
            plot_trajectories(rounds_df, os.path.join(args.out_dir, "plots"), dataset)

    if len(run_df):
        summarize_runs(run_df, args.out_dir)

    print("Wrote:", rounds_path)
    print("Wrote:", run_path)
    print("Wrote:", os.path.join(args.out_dir, "summary.csv"))
    print("Plots:", os.path.join(args.out_dir, "plots"))

if __name__ == "__main__":
    main()
