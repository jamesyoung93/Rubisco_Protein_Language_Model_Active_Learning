import os, json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from xgboost import XGBRegressor

# -----------------------------
# CONFIG (edit these if desired)
# -----------------------------
EMB_FILE = "esm2_t33_650m_full.npy"
WT_EMB_FILE = "esm2_wt_embeddings.npy"
LABELS_FILE = "rubisco_datasets_merged.csv"
OUTDIR = "results_full_eval/hoffmann_o2"

SEEDS = [0, 1, 2, 3, 4]

# Control how harsh the position-holdout is:
TEST_FRAC_LOW = 0.20
TEST_FRAC_HIGH = 0.35
TEST_FRAC_TARGET = (TEST_FRAC_LOW + TEST_FRAC_HIGH) / 2

VAL_FRAC = 0.10

# Model settings
N_JOBS = 16
PCA_DIM = 128
EARLY_STOP = 200
N_ESTIMATORS = 8000

# Candidate selection defaults
MAX_N_MUT_FOR_CANDIDATES = 8          # set to None to disable
N2_MARGIN_BELOW_WT = 0.10             # constraint: predicted N2 >= WT_N2 - margin
O2_MARGIN_BELOW_WT = None             # set e.g. 0.10 to enforce O2 floor too
SHORTLIST_N = 300                     # size of shortlist csv

# -----------------------------
# Helpers
# -----------------------------
def safe_spear(y, p):
    c = spearmanr(y, p).correlation
    return float(c) if np.isfinite(c) else np.nan

def topk_precision_and_enrich(y, p, frac=0.05):
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    true_top = set(np.argsort(y)[-k:])
    pred_top = set(np.argsort(p)[-k:])
    precision = len(true_top & pred_top) / k
    enrich = float(np.mean(y[list(pred_top)]) / (np.mean(y) + 1e-12))
    return float(precision), float(enrich)

def parse_pos_list(s) -> List[int]:
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    out = []
    for tok in s.split(";"):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return out

@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    holdout_positions: List[int]
    test_frac: float

def make_pos_holdout_split(pos_lists: List[List[int]], seed: int,
                           frac_low: float, frac_high: float, frac_target: float,
                           val_frac: float) -> Split:
    rng = np.random.default_rng(seed)
    all_pos = sorted({p for lst in pos_lists for p in lst})
    if len(all_pos) == 0:
        raise RuntimeError("No mutation positions found to construct position holdout.")

    perm = rng.permutation(all_pos)

    # Precompute for each variant: the earliest rank among its positions in perm
    rank_map = {p: i for i, p in enumerate(perm)}
    max_rank = len(perm) + 1
    min_rank = np.full(len(pos_lists), max_rank, dtype=int)
    for i, lst in enumerate(pos_lists):
        if not lst:
            continue
        mr = max_rank
        for p in lst:
            r = rank_map.get(p, max_rank)
            if r < mr:
                mr = r
        min_rank[i] = mr

    # Evaluate achievable test fractions for prefixes k=1..len(perm)
    fracs = np.empty(len(perm), dtype=float)
    for k in range(1, len(perm) + 1):
        fracs[k - 1] = float((min_rank < k).mean())

    # Choose k within band closest to target; else closest overall
    in_band = np.where((fracs >= frac_low) & (fracs <= frac_high))[0]
    if len(in_band) > 0:
        k_idx = in_band[np.argmin(np.abs(fracs[in_band] - frac_target))]
    else:
        k_idx = int(np.argmin(np.abs(fracs - frac_target)))

    k = k_idx + 1
    hold = perm[:k].tolist()
    test_mask = (min_rank < k)

    test_idx = np.where(test_mask)[0]
    train_pool = np.where(~test_mask)[0]

    # val from train_pool
    if len(train_pool) < 10:
        raise RuntimeError("Train pool too small after holdout; adjust test frac band.")
    tr_idx, va_idx = train_test_split(train_pool, test_size=val_frac, random_state=seed)

    return Split(
        train_idx=tr_idx,
        val_idx=va_idx,
        test_idx=test_idx,
        holdout_positions=hold,
        test_frac=float(test_mask.mean()),
    )

def make_depth_holdout_split(nmut: np.ndarray, seed: int, val_frac: float) -> Split:
    tr_mask = (nmut <= 4)
    va_mask = (nmut == 5)
    te_mask = (nmut >= 6)

    # Fallback if val sparse
    if va_mask.sum() < 50:
        tr_idx_all = np.where(tr_mask)[0]
        tr_idx, va_idx = train_test_split(tr_idx_all, test_size=val_frac, random_state=seed)
        tr_mask2 = np.zeros_like(tr_mask); tr_mask2[tr_idx] = True
        va_mask2 = np.zeros_like(tr_mask); va_mask2[va_idx] = True
        tr_mask, va_mask = tr_mask2, va_mask2

    # Fallback if test sparse
    if te_mask.sum() < 50:
        te_mask = (nmut >= 5)

    return Split(
        train_idx=np.where(tr_mask)[0],
        val_idx=np.where(va_mask)[0],
        test_idx=np.where(te_mask)[0],
        holdout_positions=[],
        test_frac=float(te_mask.mean()),
    )

def fit_xgb_pca(Xtr, ytr, Xva, yva, Xte, pca_dim: int, n_jobs: int) -> np.ndarray:
    scaler = StandardScaler()
    pca = PCA(n_components=pca_dim, random_state=0, svd_solver="randomized")

    Xtr_t = pca.fit_transform(scaler.fit_transform(Xtr))
    Xva_t = pca.transform(scaler.transform(Xva))
    Xte_t = pca.transform(scaler.transform(Xte))

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=N_ESTIMATORS,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=n_jobs,
        random_state=0,
        tree_method="hist",
        early_stopping_rounds=EARLY_STOP,
    )
    model.fit(Xtr_t, ytr, eval_set=[(Xva_t, yva)], verbose=False)
    return model.predict(Xte_t)

def fit_final_pipeline(X, y, pca_dim: int, seed: int, n_jobs: int):
    idx = np.where(np.isfinite(y))[0]
    tr_idx, va_idx = train_test_split(idx, test_size=VAL_FRAC, random_state=seed)

    scaler = StandardScaler()
    pca = PCA(n_components=pca_dim, random_state=seed, svd_solver="randomized")

    Xtr_t = pca.fit_transform(scaler.fit_transform(X[tr_idx]))
    Xva_t = pca.transform(scaler.transform(X[va_idx]))

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=N_ESTIMATORS,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=n_jobs,
        random_state=seed,
        tree_method="hist",
        early_stopping_rounds=EARLY_STOP,
    )
    model.fit(Xtr_t, y[tr_idx], eval_set=[(Xva_t, y[va_idx])], verbose=False)

    # Predict all
    Xall_t = pca.transform(scaler.transform(X))
    pred_all = model.predict(Xall_t)
    return pred_all

def eval_metrics(yte, pte) -> Dict[str, float]:
    out = {
        "n_test": int(len(yte)),
        "r2": float(r2_score(yte, pte)) if len(yte) >= 2 else np.nan,
        "spearman": safe_spear(yte, pte),
    }
    p5, e5 = topk_precision_and_enrich(yte, pte, 0.05)
    out["precision_at_5pct"] = p5
    out["enrichment_at_5pct"] = e5
    return out

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load embeddings
    emb = np.load(EMB_FILE, allow_pickle=True).item()
    X_all = emb["emb"].astype(np.float32)
    ids_all = emb["ids"].astype(str)

    # Load labels aligned to embedding order
    df = pd.read_csv(LABELS_FILE, low_memory=False)
    df = df.set_index("variant_id").loc[ids_all].reset_index()

    # HOFF subset
    hoff_mask = (df["dataset_id"].values == "HOFF")
    hoff = df.loc[hoff_mask].copy()
    X_hoff = X_all[hoff_mask]

    # Filter stop variants
    if "has_stop" in hoff.columns:
        good = ~hoff["has_stop"].fillna(False).astype(bool).to_numpy()
        hoff = hoff.loc[good].copy()
        X_hoff = X_hoff[good]

    hoff["n_mut"] = pd.to_numeric(hoff["n_mut"], errors="coerce")
    nmut = hoff["n_mut"].astype(int).to_numpy()

    # Parse mut positions (for holdout)
    pos_lists = hoff["mut_positions"].apply(parse_pos_list).tolist()

    # WT embedding
    wt = np.load(WT_EMB_FILE, allow_pickle=True).item()
    wt_ids = wt["ids"].astype(str)
    wt_emb = wt["emb"].astype(np.float32)
    try:
        wt_hoff = wt_emb[np.where(wt_ids == "HOFF|WT")[0][0]]
    except Exception:
        raise RuntimeError("Could not find HOFF|WT in esm2_wt_embeddings.npy")

    # Feature modes to compare (concat intentionally omitted due to prior collapse)
    FEATURES: Dict[str, np.ndarray] = {
        "mean": X_hoff,
        "delta": (X_hoff - wt_hoff[None, :]),
    }

    # Targets
    y_delta = pd.to_numeric(hoff["hoff_delta_O2_minus_N2"], errors="coerce").to_numpy()
    y_n2    = pd.to_numeric(hoff["hoff_fitness_N2"], errors="coerce").to_numpy()
    y_o2    = pd.to_numeric(hoff["hoff_fitness_O2"], errors="coerce").to_numpy()

    targets = {
        "delta_O2_minus_N2": y_delta,
        "fitness_N2": y_n2,
        "fitness_O2": y_o2,
    }

    # -------------------------
    # CV evaluation (pos-holdout across seeds + depth holdout once per seed)
    # -------------------------
    metrics_rows = []
    holdout_by_seed = {}

    for seed in SEEDS:
        pos_split = make_pos_holdout_split(
            pos_lists, seed=seed,
            frac_low=TEST_FRAC_LOW, frac_high=TEST_FRAC_HIGH, frac_target=TEST_FRAC_TARGET,
            val_frac=VAL_FRAC
        )
        depth_split = make_depth_holdout_split(nmut, seed=seed, val_frac=VAL_FRAC)

        holdout_by_seed[str(seed)] = {
            "pos_holdout_positions": pos_split.holdout_positions,
            "pos_test_frac": pos_split.test_frac,
            "depth_test_frac": depth_split.test_frac,
        }

        for feat_name, F in FEATURES.items():
            for tgt_name, y in targets.items():
                # restrict indices to finite labels for the target
                fin = np.isfinite(y)

                def sub(idx):
                    idx = np.asarray(idx)
                    return idx[fin[idx]]

                # ---- position holdout eval ----
                tr = sub(pos_split.train_idx)
                va = sub(pos_split.val_idx)
                te = sub(pos_split.test_idx)

                if len(tr) > 200 and len(va) > 50 and len(te) > 200:
                    pred = fit_xgb_pca(F[tr], y[tr], F[va], y[va], F[te], pca_dim=PCA_DIM, n_jobs=N_JOBS)
                    m = eval_metrics(y[te], pred)
                    m.update({
                        "seed": seed,
                        "split": "pos_holdout",
                        "feature_mode": feat_name,
                        "target": tgt_name,
                        "test_frac": pos_split.test_frac,
                        "holdout_positions": len(pos_split.holdout_positions),
                    })
                    metrics_rows.append(m)

                # ---- depth holdout eval ----
                tr = sub(depth_split.train_idx)
                va = sub(depth_split.val_idx)
                te = sub(depth_split.test_idx)

                if len(tr) > 200 and len(va) > 50 and len(te) > 200:
                    pred = fit_xgb_pca(F[tr], y[tr], F[va], y[va], F[te], pca_dim=PCA_DIM, n_jobs=N_JOBS)
                    m = eval_metrics(y[te], pred)
                    m.update({
                        "seed": seed,
                        "split": "depth_holdout",
                        "feature_mode": feat_name,
                        "target": tgt_name,
                        "test_frac": depth_split.test_frac,
                        "holdout_positions": 0,
                    })
                    metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(OUTDIR, "hoffmann_cv_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    with open(os.path.join(OUTDIR, "holdout_positions_by_seed.json"), "w") as f:
        json.dump(holdout_by_seed, f, indent=2)

    print("Wrote:", metrics_path)
    print("Wrote:", os.path.join(OUTDIR, "holdout_positions_by_seed.json"))

    # -------------------------
    # Pick best feature mode for primary target using pos-holdout Spearman
    # -------------------------
    primary = metrics_df[(metrics_df["split"] == "pos_holdout") & (metrics_df["target"] == "delta_O2_minus_N2")]
    summary = primary.groupby("feature_mode")[["spearman", "enrichment_at_5pct", "precision_at_5pct"]].mean(numeric_only=True)
    print("\n=== Feature-mode summary (pos-holdout, primary target) ===")
    print(summary.to_string())

    if len(summary) == 0:
        raise RuntimeError("No CV results for primary target; check labels and filtering.")

    best_mode = summary["spearman"].idxmax()
    with open(os.path.join(OUTDIR, "selected_feature_mode.txt"), "w") as f:
        f.write(best_mode + "\n")
    print("\nSelected feature_mode:", best_mode)

    # -------------------------
    # Train final models on all Hoffmann data (with internal val) and rank candidates
    # -------------------------
    F_best = FEATURES[best_mode]

    pred_delta = fit_final_pipeline(F_best, y_delta, pca_dim=PCA_DIM, seed=SEEDS[0], n_jobs=N_JOBS)
    pred_n2    = fit_final_pipeline(F_best, y_n2,    pca_dim=PCA_DIM, seed=SEEDS[0], n_jobs=N_JOBS)
    pred_o2    = fit_final_pipeline(F_best, y_o2,    pca_dim=PCA_DIM, seed=SEEDS[0], n_jobs=N_JOBS)

    out = hoff.copy()
    out["variant_str"] = out["variant_id"].astype(str).str.replace("^HOFF\\|", "", regex=True)
    out["pred_delta_O2_minus_N2"] = pred_delta
    out["pred_fitness_N2"] = pred_n2
    out["pred_fitness_O2"] = pred_o2

    # WT observed (if present)
    wt_rows = out[pd.to_numeric(out["n_mut"], errors="coerce") == 0]
    wt_n2_obs = pd.to_numeric(wt_rows.get("hoff_fitness_N2"), errors="coerce").median() if len(wt_rows) else np.nan
    wt_o2_obs = pd.to_numeric(wt_rows.get("hoff_fitness_O2"), errors="coerce").median() if len(wt_rows) else np.nan

    # Constraints
    keep = np.ones(len(out), dtype=bool)

    if MAX_N_MUT_FOR_CANDIDATES is not None:
        keep &= (pd.to_numeric(out["n_mut"], errors="coerce").to_numpy() <= MAX_N_MUT_FOR_CANDIDATES)

    if np.isfinite(wt_n2_obs):
        n2_thresh = float(wt_n2_obs - N2_MARGIN_BELOW_WT)
    else:
        # fallback: require at least median predicted N2
        n2_thresh = float(np.quantile(out["pred_fitness_N2"].to_numpy(), 0.50))

    keep &= (out["pred_fitness_N2"].to_numpy() >= n2_thresh)

    if O2_MARGIN_BELOW_WT is not None and np.isfinite(wt_o2_obs):
        o2_thresh = float(wt_o2_obs - O2_MARGIN_BELOW_WT)
        keep &= (out["pred_fitness_O2"].to_numpy() >= o2_thresh)
    else:
        o2_thresh = None

    out["keep_constraint"] = keep
    out["n2_threshold_used"] = n2_thresh
    out["o2_threshold_used"] = o2_thresh if o2_thresh is not None else np.nan

    # Rank (primary delta, tie-break O2 then N2)
    out_ranked = out.sort_values(
        by=["keep_constraint", "pred_delta_O2_minus_N2", "pred_fitness_O2", "pred_fitness_N2"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    ranked_path = os.path.join(OUTDIR, "hoffmann_candidates_ranked.csv")
    out_ranked.to_csv(ranked_path, index=False)

    shortlist = out_ranked[out_ranked["keep_constraint"]].head(SHORTLIST_N).copy()
    shortlist_path = os.path.join(OUTDIR, "hoffmann_candidates_shortlist.csv")
    shortlist.to_csv(shortlist_path, index=False)

    cfg = {
        "selected_feature_mode": best_mode,
        "test_frac_band": [TEST_FRAC_LOW, TEST_FRAC_HIGH],
        "test_frac_target": TEST_FRAC_TARGET,
        "seeds": SEEDS,
        "MAX_N_MUT_FOR_CANDIDATES": MAX_N_MUT_FOR_CANDIDATES,
        "N2_MARGIN_BELOW_WT": N2_MARGIN_BELOW_WT,
        "O2_MARGIN_BELOW_WT": O2_MARGIN_BELOW_WT,
        "n2_threshold_used": n2_thresh,
        "o2_threshold_used": o2_thresh,
        "wt_n2_obs": float(wt_n2_obs) if np.isfinite(wt_n2_obs) else None,
        "wt_o2_obs": float(wt_o2_obs) if np.isfinite(wt_o2_obs) else None,
    }
    with open(os.path.join(OUTDIR, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("\nWrote:", ranked_path)
    print("Wrote:", shortlist_path)
    print("Wrote:", os.path.join(OUTDIR, "run_config.json"))

if __name__ == "__main__":
    main()
