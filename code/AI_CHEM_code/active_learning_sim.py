#!/usr/bin/env python3
import argparse, os, json, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# -----------------------------
# Utility
# -----------------------------
def safe_spearman(y, p) -> float:
    c = spearmanr(y, p).correlation
    return float(c) if np.isfinite(c) else np.nan

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def median_min_dist(X: np.ndarray, new_idx: np.ndarray, ref_idx: np.ndarray) -> float:
    """
    Median Euclidean distance from each new point to its nearest neighbor in ref set.
    Computed in a safe way for small batch sizes.
    """
    if len(ref_idx) == 0 or len(new_idx) == 0:
        return np.nan
    A = X[new_idx]          # [B, D]
    R = X[ref_idx]          # [N, D]
    # ||a-r||^2 = ||a||^2 + ||r||^2 - 2 a·r
    a2 = np.sum(A*A, axis=1, keepdims=True)      # [B,1]
    r2 = np.sum(R*R, axis=1, keepdims=True).T    # [1,N]
    d2 = a2 + r2 - 2.0 * (A @ R.T)               # [B,N]
    d2 = np.maximum(d2, 0.0)
    md = np.sqrt(np.min(d2, axis=1))
    return float(np.median(md))

def topk_recovery(discovered_set: set, top_set: set) -> float:
    if len(top_set) == 0:
        return np.nan
    return len(discovered_set & top_set) / len(top_set)

def build_top_set(y: np.ndarray, frac: float, mask: Optional[np.ndarray] = None) -> set:
    idx = np.arange(len(y))
    if mask is not None:
        idx = idx[mask]
    k = max(1, int(math.ceil(frac * len(idx))))
    # top k among masked subset
    sub = y[idx]
    order = np.argsort(sub)
    top_idx = idx[order[-k:]]
    return set(top_idx.tolist())

def position_coverage(pos: np.ndarray, discovered_idx: np.ndarray) -> float:
    all_pos = set(pos.tolist())
    seen = set(pos[discovered_idx].tolist())
    return len(seen) / max(1, len(all_pos))

def load_wt_embedding(wt_path: str, wt_id: str) -> np.ndarray:
    d = np.load(wt_path, allow_pickle=True).item()
    ids = d["ids"].astype(str)
    emb = d["emb"]
    m = np.where(ids == wt_id)[0]
    if len(m) == 0:
        raise RuntimeError(f"WT id {wt_id} not found in {wt_path}")
    return emb[m[0]].astype(np.float32)


# -----------------------------
# XGBoost training / prediction
# -----------------------------
@dataclass
class XGBConfig:
    pca_dim: int = 128
    max_depth: int = 6
    reg_lambda: float = 1.0
    eta: float = 0.03
    subsample: float = 0.85
    colsample: float = 0.85
    min_child_weight: float = 1.0
    num_boost_round: int = 2500
    early_stopping: int = 100
    nthread: int = 16
    val_frac: float = 0.10

def train_predict_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    labeled_idx: np.ndarray,
    pool_idx: np.ndarray,
    w: Optional[np.ndarray],
    cfg: XGBConfig,
    ensemble_size: int,
    rng: np.random.Generator,
    seed_base: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns mean, std, per_member_preds for pool_idx.
    Bootstrap ensemble with early stopping.
    """
    if len(pool_idx) == 0:
        return np.array([]), np.array([]), np.zeros((ensemble_size, 0), dtype=np.float32)

    preds = np.zeros((ensemble_size, len(pool_idx)), dtype=np.float32)

    # Pre-create pool DMatrix once per round
    dpool = xgb.DMatrix(X[pool_idx])

    for m in range(ensemble_size):
        # bootstrap sample
        boot = rng.choice(labeled_idx, size=len(labeled_idx), replace=True)
        # split boot into train/val
        rs = int((seed_base + 10007*m) % (2**31 - 1))
        tr_b, va_b = train_test_split(boot, test_size=cfg.val_frac, random_state=rs)

        dtr = xgb.DMatrix(X[tr_b], label=y[tr_b], weight=(w[tr_b] if w is not None else None))
        dva = xgb.DMatrix(X[va_b], label=y[va_b], weight=(w[va_b] if w is not None else None))

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": cfg.eta,
            "max_depth": int(cfg.max_depth),
            "min_child_weight": float(cfg.min_child_weight),
            "subsample": float(cfg.subsample),
            "colsample_bytree": float(cfg.colsample),
            "lambda": float(cfg.reg_lambda),
            "tree_method": "hist",
            "seed": rs,
            "nthread": int(cfg.nthread),
        }

        bst = xgb.train(
            params,
            dtr,
            num_boost_round=int(cfg.num_boost_round),
            evals=[(dva, "val")],
            early_stopping_rounds=int(cfg.early_stopping),
            verbose_eval=False,
        )

        bi = bst.best_iteration
        if bi is None:
            pred = bst.predict(dpool)
        else:
            pred = bst.predict(dpool, iteration_range=(0, bi + 1))
        preds[m, :] = pred.astype(np.float32)

    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std, preds


# -----------------------------
# Feature preparation
# -----------------------------
@dataclass
class Features:
    X_model: np.ndarray     # features for learning
    X_novel: np.ndarray     # features for novelty distance (PCA-only)
    meta: pd.DataFrame

def build_features_from_embeddings(
    emb: np.ndarray,
    meta: pd.DataFrame,
    pca_dim: int,
    add_nmut: bool,
    wt_vec: Optional[np.ndarray] = None,
) -> Features:
    """
    Unsupervised transform on the full pool (allowed in AL; no labels).
    PCA is fit on all sequences.
    """
    X = emb.astype(np.float32)
    if wt_vec is not None:
        X = (X - wt_vec[None, :]).astype(np.float32)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=int(pca_dim), random_state=0, svd_solver="randomized")
    Xp = pca.fit_transform(Xs).astype(np.float32)

    X_novel = Xp

    if add_nmut:
        nm = pd.to_numeric(meta["n_mut"], errors="coerce").fillna(0).to_numpy().astype(np.float32)
        Xnum = np.vstack([nm, nm**2]).T
        scn = StandardScaler()
        Xnum_s = scn.fit_transform(Xnum).astype(np.float32)
        X_model = np.hstack([Xp, Xnum_s]).astype(np.float32)
    else:
        X_model = Xp.astype(np.float32)

    return Features(X_model=X_model, X_novel=X_novel, meta=meta)


# -----------------------------
# Acquisition
# -----------------------------
def pick_batch(
    strategy: str,
    pool_idx: np.ndarray,
    mean_obj: np.ndarray,
    std_obj: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
    beta: float = 1.0,
) -> np.ndarray:
    if len(pool_idx) == 0:
        return np.array([], dtype=int)
    if strategy == "random":
        choice = rng.choice(pool_idx, size=min(batch_size, len(pool_idx)), replace=False)
        return choice.astype(int)

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

    # select top by score
    k = min(batch_size, len(pool_idx))
    top_local = np.argpartition(score, -k)[-k:]
    # sort those k by score descending
    top_local = top_local[np.argsort(score[top_local])[::-1]]
    return pool_idx[top_local].astype(int)


# -----------------------------
# Simulation core
# -----------------------------
@dataclass
class SimConfig:
    init_n: int = 200
    batch_size: int = 48
    max_rounds: int = 30
    ensemble_size: int = 5
    n_reps: int = 10
    beta: float = 1.0
    top_fracs: Tuple[float, float] = (0.01, 0.05)
    novelty_cutoff_frac: float = 0.20
    novelty_patience: int = 3
    novelty_min_rounds: int = 5
    dms_pos_coverage_stop: float = 0.95

def simulate(
    dataset_name: str,
    X: np.ndarray,
    Xnovel: np.ndarray,
    y_obj: np.ndarray,
    pool_mask: np.ndarray,
    strategies: List[str],
    simcfg: SimConfig,
    modelcfg_obj: XGBConfig,
    w_obj: Optional[np.ndarray],
    # optional constraint (Hoffmann)
    y_constraint: Optional[np.ndarray] = None,
    modelcfg_constraint: Optional[XGBConfig] = None,
    w_constraint: Optional[np.ndarray] = None,
    constraint_thresh_true: Optional[float] = None,
    constraint_mode: str = "hard",          # hard or soft
    constraint_lambda: float = 5.0,         # only for soft
    # DMS position coverage stop
    dms_positions: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Active learning simulation.
    pool_mask: boolean mask of candidates eligible in the pool (e.g. finite labels).
    """
    n = len(y_obj)
    all_idx = np.arange(n)
    pool_all = all_idx[pool_mask]

    # Precompute true top sets for eval
    if y_constraint is not None and constraint_thresh_true is not None:
        feasible_truth = (y_constraint >= constraint_thresh_true) & pool_mask
    else:
        feasible_truth = pool_mask.copy()

    top_sets = {frac: build_top_set(y_obj, frac=frac, mask=feasible_truth) for frac in simcfg.top_fracs}

    rows = []
    for rep in range(simcfg.n_reps):
        rep_seed = 100000 + rep
        rng0 = np.random.default_rng(rep_seed)
        init = rng0.choice(pool_all, size=min(simcfg.init_n, len(pool_all)), replace=False).astype(int)

        for strategy in strategies:
            # Use a fresh RNG per (rep,strategy) but same initial set for fairness
            rng = np.random.default_rng(rep_seed + (hash(strategy) % 10000))

            labeled = set(init.tolist())
            discovered = set(init.tolist())
            stop_reason = ""
            low_novelty_streak = 0
            baseline_novelty = None

            # DMS coverage
            if dms_positions is not None:
                cov = position_coverage(dms_positions, np.array(sorted(discovered), dtype=int))
            else:
                cov = np.nan

            # Metrics at round 0
            discovered_arr = np.array(sorted(discovered), dtype=int)
            if y_constraint is not None and constraint_thresh_true is not None:
                feas_disc = discovered_arr[y_constraint[discovered_arr] >= constraint_thresh_true]
                best_so_far = float(np.max(y_obj[feas_disc])) if len(feas_disc) else np.nan
            else:
                best_so_far = float(np.max(y_obj[discovered_arr])) if len(discovered_arr) else np.nan

            rec = {f"recovery_top_{int(frac*100)}pct": topk_recovery(discovered, top_sets[frac]) for frac in simcfg.top_fracs}

            rows.append({
                "dataset": dataset_name,
                "strategy": strategy,
                "rep": rep,
                "round": 0,
                "n_assayed": len(discovered),
                "best_so_far": best_so_far,
                "novelty_median": np.nan,
                "baseline_novelty": np.nan,
                "low_novelty_streak": 0,
                "stop_reason": "",
                "dms_pos_coverage": cov,
                **rec,
            })

            # rounds
            for t in range(1, simcfg.max_rounds + 1):
                labeled_idx = np.array(sorted(labeled), dtype=int)
                pool_idx = pool_all[~np.isin(pool_all, labeled_idx)]

                if len(pool_idx) == 0:
                    stop_reason = "pool_exhausted"
                    break

                # For random, no model required
                if strategy == "random":
                    # For Hoffmann constraints, we still need a constraint handling rule
                    if y_constraint is not None and constraint_thresh_true is not None:
                        # Use true constraint threshold only to filter candidates? In real life you don't have it.
                        # For acquisition, we approximate by ignoring constraint for random selection (conservative baseline).
                        chosen = rng.choice(pool_idx, size=min(simcfg.batch_size, len(pool_idx)), replace=False).astype(int)
                    else:
                        chosen = rng.choice(pool_idx, size=min(simcfg.batch_size, len(pool_idx)), replace=False).astype(int)

                    mean_obj = std_obj = preds_obj = None
                    mean_c = std_c = preds_c = None

                else:
                    # Train objective predictions (single model for greedy; ensemble for others)
                    ens = 1 if strategy == "greedy" else simcfg.ensemble_size
                    mean_obj, std_obj, _ = train_predict_ensemble(
                        X=X,
                        y=y_obj,
                        labeled_idx=labeled_idx,
                        pool_idx=pool_idx,
                        w=w_obj,
                        cfg=modelcfg_obj,
                        ensemble_size=ens,
                        rng=rng,
                        seed_base=rep_seed + 1000*t,
                    )

                    # Constraint model (Hoffmann) if provided
                    if y_constraint is not None and modelcfg_constraint is not None and constraint_thresh_true is not None:
                        mean_c, std_c, _ = train_predict_ensemble(
                            X=X,
                            y=y_constraint,
                            labeled_idx=labeled_idx,
                            pool_idx=pool_idx,
                            w=w_constraint,
                            cfg=modelcfg_constraint,
                            ensemble_size=ens,
                            rng=rng,
                            seed_base=rep_seed + 2000*t,
                        )

                        # Apply constraint in acquisition
                        if constraint_mode == "hard":
                            feas = (mean_c >= constraint_thresh_true)
                            cand_local = np.where(feas)[0]
                            if len(cand_local) < simcfg.batch_size:
                                # fallback: take top by predicted constraint
                                cand_local = np.argsort(mean_c)[-min(len(mean_c), max(simcfg.batch_size*3, simcfg.batch_size)):]
                            pool2 = pool_idx[cand_local]
                            m2 = mean_obj[cand_local]
                            s2 = std_obj[cand_local]
                            chosen = pick_batch(strategy, pool2, m2, s2, rng, simcfg.batch_size, beta=simcfg.beta)
                        elif constraint_mode == "soft":
                            # penalty for violating constraint
                            penalty = np.maximum(0.0, constraint_thresh_true - mean_c)
                            m_adj = mean_obj - constraint_lambda * penalty
                            chosen = pick_batch(strategy, pool_idx, m_adj, std_obj, rng, simcfg.batch_size, beta=simcfg.beta)
                        else:
                            raise ValueError("constraint_mode must be hard or soft")
                    else:
                        chosen = pick_batch(strategy, pool_idx, mean_obj, std_obj, rng, simcfg.batch_size, beta=simcfg.beta)

                chosen = np.array(chosen, dtype=int)
                if len(chosen) == 0:
                    stop_reason = "no_candidates"
                    break

                # novelty
                prev_labeled = labeled_idx.copy()
                nov = median_min_dist(Xnovel, chosen, prev_labeled)
                if baseline_novelty is None and np.isfinite(nov):
                    baseline_novelty = nov

                if baseline_novelty is not None and np.isfinite(nov) and t >= simcfg.novelty_min_rounds:
                    if nov < simcfg.novelty_cutoff_frac * baseline_novelty:
                        low_novelty_streak += 1
                    else:
                        low_novelty_streak = 0

                # update labeled/discovered
                for i in chosen.tolist():
                    labeled.add(int(i))
                    discovered.add(int(i))

                discovered_arr = np.array(sorted(discovered), dtype=int)

                # DMS coverage
                if dms_positions is not None:
                    cov = position_coverage(dms_positions, discovered_arr)
                else:
                    cov = np.nan

                # Evaluate best so far (truth-based for Hoffmann constraint)
                if y_constraint is not None and constraint_thresh_true is not None:
                    feas_disc = discovered_arr[y_constraint[discovered_arr] >= constraint_thresh_true]
                    best_so_far = float(np.max(y_obj[feas_disc])) if len(feas_disc) else np.nan
                else:
                    best_so_far = float(np.max(y_obj[discovered_arr])) if len(discovered_arr) else np.nan

                rec = {f"recovery_top_{int(frac*100)}pct": topk_recovery(discovered, top_sets[frac]) for frac in simcfg.top_fracs}

                rows.append({
                    "dataset": dataset_name,
                    "strategy": strategy,
                    "rep": rep,
                    "round": t,
                    "n_assayed": len(discovered),
                    "best_so_far": best_so_far,
                    "novelty_median": nov,
                    "baseline_novelty": baseline_novelty if baseline_novelty is not None else np.nan,
                    "low_novelty_streak": low_novelty_streak,
                    "stop_reason": "",
                    "dms_pos_coverage": cov,
                    **rec,
                })

                # stop checks
                if dms_positions is not None and np.isfinite(cov) and cov >= simcfg.dms_pos_coverage_stop:
                    stop_reason = "dms_pos_coverage_reached"
                    break

                if low_novelty_streak >= simcfg.novelty_patience:
                    stop_reason = "novelty_exhausted"
                    break

            # mark final row for this run with stop reason
            # (find last row for this rep/strategy)
            if stop_reason:
                for i in range(len(rows)-1, -1, -1):
                    if rows[i]["dataset"] == dataset_name and rows[i]["strategy"] == strategy and rows[i]["rep"] == rep:
                        rows[i]["stop_reason"] = stop_reason
                        break

    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------
def plot_trajectories(df: pd.DataFrame, outdir: str, dataset: str):
    if plt is None:
        print("matplotlib not available; skipping plots.")
        return

    ensure_dir(outdir)
    sub = df[df["dataset"] == dataset].copy()
    if len(sub) == 0:
        return

    metrics = ["best_so_far", "recovery_top_1pct", "recovery_top_5pct", "novelty_median"]
    # Determine assay grid
    assay_grid = sorted(sub["n_assayed"].unique().tolist())

    def summarize(metric: str):
        out = []
        for strat in sorted(sub["strategy"].unique()):
            ssub = sub[sub["strategy"] == strat]
            reps = sorted(ssub["rep"].unique())
            mat = np.full((len(reps), len(assay_grid)), np.nan, dtype=float)

            for r_i, r in enumerate(reps):
                rsub = ssub[ssub["rep"] == r].sort_values("n_assayed")
                vals = dict(zip(rsub["n_assayed"].tolist(), rsub[metric].tolist()))
                last = np.nan
                for j, a in enumerate(assay_grid):
                    if a in vals:
                        last = vals[a]
                    mat[r_i, j] = last
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            out.append((strat, mean, std))
        return out

    for metric in metrics:
        summary = summarize(metric)
        plt.figure()
        for strat, mean, std in summary:
            plt.plot(assay_grid, mean, label=strat)
            # shaded uncertainty
            plt.fill_between(assay_grid, mean-std, mean+std, alpha=0.2)
        plt.xlabel("Assays (cumulative)")
        plt.ylabel(metric)
        plt.title(f"{dataset}: {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{dataset}_{metric}.png"), dpi=200)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=".", help="Directory containing embedding and CSV files")
    ap.add_argument("--emb_file", default="esm2_t33_650m_full.npy")
    ap.add_argument("--labels_file", default="rubisco_datasets_merged.csv")
    ap.add_argument("--wt_file", default="esm2_wt_embeddings.npy")
    ap.add_argument("--outdir", default="results_active_learning")

    ap.add_argument("--dataset", choices=["DMS", "HOFF", "BOTH"], default="BOTH")
    ap.add_argument("--strategies", default="random,greedy,uncertainty,ucb,thompson")
    ap.add_argument("--n_reps", type=int, default=10)
    ap.add_argument("--init_n", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--max_rounds", type=int, default=25)
    ap.add_argument("--ensemble_size", type=int, default=5)
    ap.add_argument("--beta", type=float, default=1.0)

    ap.add_argument("--novelty_cutoff_frac", type=float, default=0.20)
    ap.add_argument("--novelty_patience", type=int, default=3)
    ap.add_argument("--novelty_min_rounds", type=int, default=5)

    # DMS settings
    ap.add_argument("--dms_target", default="dms_enrichment_mean")
    ap.add_argument("--dms_pos_coverage_stop", type=float, default=0.95)
    ap.add_argument("--dms_pca_dim", type=int, default=128)

    # Hoffmann settings
    ap.add_argument("--hoff_feature_mode", choices=["mean", "delta"], default="mean")
    ap.add_argument("--hoff_pca_dim", type=int, default=64)
    ap.add_argument("--hoff_objective", default="hoff_delta_O2_minus_N2")
    ap.add_argument("--hoff_constraint", default="hoff_fitness_N2")
    ap.add_argument("--hoff_n2_margin", type=float, default=0.10)
    ap.add_argument("--hoff_constraint_mode", choices=["hard", "soft"], default="hard")
    ap.add_argument("--hoff_constraint_lambda", type=float, default=5.0)

    # Model hyperparams (shared; set sensible defaults)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--reg_lambda", type=float, default=10.0)
    ap.add_argument("--num_boost_round", type=int, default=2500)
    ap.add_argument("--early_stopping", type=int, default=100)
    ap.add_argument("--nthread", type=int, default=16)

    args = ap.parse_args()

    os.chdir(args.workdir)
    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "plots"))

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    # Load embeddings + labels aligned
    embd = np.load(args.emb_file, allow_pickle=True).item()
    ids = embd["ids"].astype(str)
    emb = embd["emb"].astype(np.float32)

    df = pd.read_csv(args.labels_file, low_memory=False)
    df = df.set_index("variant_id").loc[ids].reset_index()

    # Global configs
    simcfg = SimConfig(
        init_n=args.init_n,
        batch_size=args.batch_size,
        max_rounds=args.max_rounds,
        ensemble_size=args.ensemble_size,
        n_reps=args.n_reps,
        beta=args.beta,
        novelty_cutoff_frac=args.novelty_cutoff_frac,
        novelty_patience=args.novelty_patience,
        novelty_min_rounds=args.novelty_min_rounds,
        dms_pos_coverage_stop=args.dms_pos_coverage_stop,
    )

    # Model config template
    def make_modelcfg(pca_dim: int) -> XGBConfig:
        return XGBConfig(
            pca_dim=pca_dim,
            max_depth=args.max_depth,
            reg_lambda=args.reg_lambda,
            num_boost_round=args.num_boost_round,
            early_stopping=args.early_stopping,
            nthread=args.nthread,
            val_frac=0.10
        )

    all_results = []

    # -----------------------------
    # DMS
    # -----------------------------
    if args.dataset in ("DMS", "BOTH"):
        dms_mask = (df["dataset_id"].values == "DMS")
        dms = df.loc[dms_mask].copy()
        Xemb = emb[dms_mask]

        y = pd.to_numeric(dms[args.dms_target], errors="coerce").to_numpy(dtype=np.float32)
        pos = pd.to_numeric(dms["position_external"], errors="coerce").to_numpy()
        finite = np.isfinite(y) & np.isfinite(pos)

        dms = dms.loc[finite].copy()
        Xemb = Xemb[finite]
        y = y[finite]
        pos = pos[finite].astype(int)

        # Features (PCA fit on all DMS sequences)
        feats = build_features_from_embeddings(
            emb=Xemb,
            meta=dms.assign(n_mut=1),  # n_mut not used here; just placeholder
            pca_dim=args.dms_pca_dim,
            add_nmut=False,
            wt_vec=None
        )
        modelcfg = make_modelcfg(args.dms_pca_dim)

        res = simulate(
            dataset_name="DMS",
            X=feats.X_model,
            Xnovel=feats.X_novel,
            y_obj=y,
            pool_mask=np.ones(len(y), dtype=bool),
            strategies=strategies,
            simcfg=simcfg,
            modelcfg_obj=modelcfg,
            w_obj=None,
            y_constraint=None,
            modelcfg_constraint=None,
            w_constraint=None,
            constraint_thresh_true=None,
            dms_positions=pos,
        )
        all_results.append(res)

    # -----------------------------
    # HOFFMANN
    # -----------------------------
    if args.dataset in ("HOFF", "BOTH"):
        h_mask = (df["dataset_id"].values == "HOFF")
        hoff = df.loc[h_mask].copy()
        Xemb = emb[h_mask]

        # Filter stops if present
        if "has_stop" in hoff.columns:
            good = ~hoff["has_stop"].fillna(False).astype(bool).to_numpy()
            hoff = hoff.loc[good].copy()
            Xemb = Xemb[good]

        y_obj = pd.to_numeric(hoff[args.hoff_objective], errors="coerce").to_numpy(dtype=np.float32)
        y_c   = pd.to_numeric(hoff[args.hoff_constraint], errors="coerce").to_numpy(dtype=np.float32)

        # Compute true constraint threshold from WT (n_mut==0), else median-based fallback
        hoff["n_mut"] = pd.to_numeric(hoff["n_mut"], errors="coerce")
        wt_rows = hoff[hoff["n_mut"] == 0]
        wt_n2 = np.nanmedian(pd.to_numeric(wt_rows[args.hoff_constraint], errors="coerce").to_numpy()) if len(wt_rows) else np.nan
        if np.isfinite(wt_n2):
            c_thresh = float(wt_n2 - args.hoff_n2_margin)
        else:
            c_thresh = float(np.nanmedian(y_c))
        # Pool mask requires objective and constraint to be finite
        finite = np.isfinite(y_obj) & np.isfinite(y_c)
        hoff = hoff.loc[finite].copy()
        Xemb = Xemb[finite]
        y_obj = y_obj[finite]
        y_c = y_c[finite]
        nmut = pd.to_numeric(hoff["n_mut"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

        # feature mode (mean vs delta)
        wt_vec = None
        if args.hoff_feature_mode == "delta":
            wt_vec = load_wt_embedding(args.wt_file, "HOFF|WT")

        # Weights (optional; use same for objective and constraint)
        # If barcode/SE cols are missing, weights are roughly uniform.
        def make_weights(meta: pd.DataFrame) -> np.ndarray:
            n = len(meta)
            def num(col):
                if col in meta.columns:
                    return pd.to_numeric(meta[col], errors="coerce").to_numpy()
                return np.full(n, np.nan)
            nb_o2 = num("hoff_n_barcodes_O2")
            nb_n2 = num("hoff_n_barcodes_N2")
            se_o2 = num("hoff_lfcSE_last_mean_O2")
            se_n2 = num("hoff_lfcSE_last_mean_N2")

            nb_o2 = np.where(np.isfinite(nb_o2), nb_o2, 1.0)
            nb_n2 = np.where(np.isfinite(nb_n2), nb_n2, 1.0)
            nb_o2 = np.clip(nb_o2, 1.0, None)
            nb_n2 = np.clip(nb_n2, 1.0, None)
            w_bar = np.log1p(np.minimum(nb_o2, nb_n2))

            if np.isfinite(se_o2).any() or np.isfinite(se_n2).any():
                fill_o2 = np.nanmedian(se_o2) if np.isfinite(se_o2).any() else 1.0
                fill_n2 = np.nanmedian(se_n2) if np.isfinite(se_n2).any() else 1.0
                se_o2 = np.where(np.isfinite(se_o2), se_o2, fill_o2)
                se_n2 = np.where(np.isfinite(se_n2), se_n2, fill_n2)
                w_se = 1.0 / (se_o2**2 + se_n2**2 + 1e-6)
                med = np.nanmedian(w_se) if np.isfinite(w_se).any() else 1.0
                w_se = w_se / (med + 1e-12)
            else:
                w_se = np.ones(n, dtype=float)

            w = w_bar * w_se
            w = np.where(np.isfinite(w), w, 1.0)
            w = w / (w.mean() + 1e-12)
            w = np.clip(w, 0.25, 4.0)
            return w.astype(np.float32)

        w = make_weights(hoff)

        # Build features with n_mut + n_mut^2
        feats = build_features_from_embeddings(
            emb=Xemb,
            meta=hoff.assign(n_mut=nmut),
            pca_dim=args.hoff_pca_dim,
            add_nmut=True,
            wt_vec=wt_vec
        )

        modelcfg_obj = make_modelcfg(args.hoff_pca_dim)
        modelcfg_c = make_modelcfg(args.hoff_pca_dim)

        res = simulate(
            dataset_name="HOFF",
            X=feats.X_model,
            Xnovel=feats.X_novel,
            y_obj=y_obj,
            pool_mask=np.ones(len(y_obj), dtype=bool),
            strategies=strategies,
            simcfg=simcfg,
            modelcfg_obj=modelcfg_obj,
            w_obj=w,
            y_constraint=y_c,
            modelcfg_constraint=modelcfg_c,
            w_constraint=w,
            constraint_thresh_true=c_thresh,
            constraint_mode=args.hoff_constraint_mode,
            constraint_lambda=args.hoff_constraint_lambda,
            dms_positions=None,
        )
        all_results.append(res)

        # Save the constraint threshold used
        with open(os.path.join(args.outdir, "hoff_constraint_threshold.json"), "w") as f:
            json.dump({"constraint_column": args.hoff_constraint,
                       "wt_n2_obs": None if not np.isfinite(wt_n2) else float(wt_n2),
                       "n2_margin": float(args.hoff_n2_margin),
                       "constraint_threshold": float(c_thresh)}, f, indent=2)

    # Combine and save
    out_df = pd.concat(all_results, ignore_index=True)
    out_csv = os.path.join(args.outdir, "active_learning_rounds.csv")
    out_df.to_csv(out_csv, index=False)

    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump({
            "dataset": args.dataset,
            "strategies": strategies,
            "simcfg": simcfg.__dict__,
            "xgb": {
                "pca_dim_dms": args.dms_pca_dim,
                "pca_dim_hoff": args.hoff_pca_dim,
                "max_depth": args.max_depth,
                "reg_lambda": args.reg_lambda,
                "num_boost_round": args.num_boost_round,
                "early_stopping": args.early_stopping,
                "nthread": args.nthread,
            },
            "hoff": {
                "feature_mode": args.hoff_feature_mode,
                "objective": args.hoff_objective,
                "constraint": args.hoff_constraint,
                "constraint_mode": args.hoff_constraint_mode,
                "constraint_lambda": args.hoff_constraint_lambda,
                "n2_margin": args.hoff_n2_margin,
            },
            "dms": {"target": args.dms_target},
        }, f, indent=2)

    print("Wrote:", out_csv)

    # Plots
    plot_dir = os.path.join(args.outdir, "plots")
    plot_trajectories(out_df, plot_dir, "DMS")
    plot_trajectories(out_df, plot_dir, "HOFF")
    if plt is not None:
        print("Plots written to:", plot_dir)
    else:
        print("No plots written (matplotlib missing).")

if __name__ == "__main__":
    main()
