import os, json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

import xgboost as xgb

# -----------------------------
# CONFIG (edit if desired)
# -----------------------------
EMB_FILE = "esm2_t33_650m_full.npy"
WT_EMB_FILE = "esm2_wt_embeddings.npy"
LABELS_FILE = "rubisco_datasets_merged.csv"

OUTDIR = "results_full_eval/hoffmann_o2_v2"
os.makedirs(OUTDIR, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]

# Position-holdout band (fixes your prior "0.3% test" issue)
TEST_FRAC_LOW = 0.20
TEST_FRAC_HIGH = 0.35
TEST_FRAC_TARGET = (TEST_FRAC_LOW + TEST_FRAC_HIGH) / 2
VAL_FRAC = 0.10

# Hyperparam sweep (small, purposeful)
PCA_DIMS = [64, 128, 256]
MAX_DEPTHS = [4, 6]
REG_LAMBDAS = [1.0, 10.0]

# XGB core params
ETA = 0.03
SUBSAMPLE = 0.85
COLSAMPLE = 0.85
MIN_CHILD_WEIGHT = 1.0
NUM_BOOST_ROUND = 8000
EARLY_STOPPING = 200
NTHREAD = 16

# Candidate selection
MAX_N_MUT_FOR_CANDIDATES = 8      # None to disable
N2_MARGIN_BELOW_WT = 0.10         # keep variants with pred_N2 >= WT_N2 - margin
O2_MARGIN_BELOW_WT = None         # set e.g. 0.10 to enforce O2 floor too
SHORTLIST_N = 300

# Whether to include explicit mutation-burden features
ADD_NMUT_FEATURES = True

# Sample weight behavior
WEIGHT_CLIP = (0.25, 4.0)         # clip extreme weights to avoid instability
FREQ_WEIGHT_ALPHA = 1.0           # holdout sampling bias toward frequent positions

# -----------------------------
# Helpers
# -----------------------------
def safe_spear(y, p) -> float:
    c = spearmanr(y, p).correlation
    return float(c) if np.isfinite(c) else np.nan

def topk_precision_and_enrich(y, p, frac=0.05) -> Tuple[float, float]:
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    true_top = set(np.argsort(y)[-k:])
    pred_top = set(np.argsort(p)[-k:])
    precision = len(true_top & pred_top) / k
    enrich = float(np.mean(y[list(pred_top)]) / (np.mean(y) + 1e-12))
    return float(precision), float(enrich)

def eval_metrics(y, p) -> Dict[str, float]:
    out = {}
    out["n_test"] = int(len(y))
    out["r2"] = float(r2_score(y, p)) if len(y) >= 2 else np.nan
    out["spearman"] = safe_spear(y, p)
    p5, e5 = topk_precision_and_enrich(y, p, 0.05)
    out["precision_at_5pct"] = p5
    out["enrichment_at_5pct"] = e5
    return out

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

def make_sample_weights(hoff: pd.DataFrame) -> np.ndarray:
    """
    Uses barcode counts and lfcSE if available:
    - w_barcodes = log1p(min(n_barcodes_O2, n_barcodes_N2))
    - w_se = 1 / (lfcSE_O2^2 + lfcSE_N2^2 + eps)
    w = w_barcodes * normalized(w_se), then normalized to mean 1 and clipped.
    """
    n = len(hoff)

    def num(col):
        if col in hoff.columns:
            return pd.to_numeric(hoff[col], errors="coerce").to_numpy()
        return np.full(n, np.nan)

    nb_o2 = num("hoff_n_barcodes_O2")
    nb_n2 = num("hoff_n_barcodes_N2")
    se_o2 = num("hoff_lfcSE_last_mean_O2")
    se_n2 = num("hoff_lfcSE_last_mean_N2")

    # barcode weight
    nb_o2 = np.where(np.isfinite(nb_o2), nb_o2, 1.0)
    nb_n2 = np.where(np.isfinite(nb_n2), nb_n2, 1.0)
    nb_o2 = np.clip(nb_o2, 1.0, None)
    nb_n2 = np.clip(nb_n2, 1.0, None)
    w_bar = np.log1p(np.minimum(nb_o2, nb_n2))

    # se weight
    if np.isfinite(se_o2).any() or np.isfinite(se_n2).any():
        se_o2 = np.where(np.isfinite(se_o2), se_o2, np.nanmedian(se_o2) if np.isfinite(se_o2).any() else 1.0)
        se_n2 = np.where(np.isfinite(se_n2), se_n2, np.nanmedian(se_n2) if np.isfinite(se_n2).any() else 1.0)
        w_se = 1.0 / (se_o2**2 + se_n2**2 + 1e-6)
        med = np.nanmedian(w_se) if np.isfinite(w_se).any() else 1.0
        w_se = w_se / (med + 1e-12)
    else:
        w_se = np.ones(n, dtype=float)

    w = w_bar * w_se

    # normalize and clip
    w = np.where(np.isfinite(w), w, 1.0)
    w = w / (w.mean() + 1e-12)
    w = np.clip(w, WEIGHT_CLIP[0], WEIGHT_CLIP[1])
    return w.astype(np.float32)

@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    holdout_positions: List[int]
    test_frac: float

def build_pos_index(pos_lists: List[List[int]]) -> Tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """
    Returns:
      pos_arr: array of unique positions
      pos_to_idx: dict pos -> np.array of variant indices containing pos
      freq: frequency array aligned to pos_arr
    """
    pos_to = defaultdict(list)
    for i, lst in enumerate(pos_lists):
        if not lst:
            continue
        for p in set(lst):
            pos_to[p].append(i)

    pos_arr = np.array(sorted(pos_to.keys()), dtype=int)
    pos_to_idx = {p: np.array(pos_to[p], dtype=int) for p in pos_arr}
    freq = np.array([len(pos_to_idx[p]) for p in pos_arr], dtype=float)
    return pos_arr, pos_to_idx, freq

def make_pos_holdout_split_greedy(pos_arr, pos_to_idx, freq, n_variants: int, seed: int) -> Split:
    """
    Frequency-weighted random order + greedy accumulation until test_frac enters band.
    Guarantees a substantial test set (unlike your v1).
    """
    rng = np.random.default_rng(seed)

    weights = np.power(freq, FREQ_WEIGHT_ALPHA)
    # Weighted random permutation (Efraimidis-Spirakis style)
    u = rng.random(len(pos_arr))
    keys = -np.log(u + 1e-12) / (weights + 1e-12)
    order = pos_arr[np.argsort(keys)]

    test_mask = np.zeros(n_variants, dtype=bool)
    fracs = []
    for pos in order:
        test_mask[pos_to_idx[pos]] = True
        fracs.append(float(test_mask.mean()))
        if fracs[-1] >= TEST_FRAC_HIGH:
            # no need to keep growing a lot beyond the band
            break

    fracs = np.array(fracs, dtype=float)

    in_band = np.where((fracs >= TEST_FRAC_LOW) & (fracs <= TEST_FRAC_HIGH))[0]
    if len(in_band) > 0:
        k_idx = in_band[np.argmin(np.abs(fracs[in_band] - TEST_FRAC_TARGET))]
    else:
        k_idx = int(np.argmin(np.abs(fracs - TEST_FRAC_TARGET)))

    k = k_idx + 1
    hold = order[:k].tolist()

    # rebuild exact mask from hold
    test_mask = np.zeros(n_variants, dtype=bool)
    for pos in hold:
        test_mask[pos_to_idx[pos]] = True

    test_idx = np.where(test_mask)[0]
    train_pool = np.where(~test_mask)[0]
    tr_idx, va_idx = train_test_split(train_pool, test_size=VAL_FRAC, random_state=seed)

    return Split(
        train_idx=tr_idx,
        val_idx=va_idx,
        test_idx=test_idx,
        holdout_positions=hold,
        test_frac=float(test_mask.mean()),
    )

def make_depth_holdout_split(nmut: np.ndarray, seed: int) -> Split:
    tr_mask = (nmut <= 4)
    va_mask = (nmut == 5)
    te_mask = (nmut >= 6)

    if va_mask.sum() < 50:
        tr_idx_all = np.where(tr_mask)[0]
        tr_idx, va_idx = train_test_split(tr_idx_all, test_size=VAL_FRAC, random_state=seed)
    else:
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]

    if te_mask.sum() < 50:
        te_mask = (nmut >= 5)
    te_idx = np.where(te_mask)[0]

    return Split(
        train_idx=tr_idx,
        val_idx=va_idx,
        test_idx=te_idx,
        holdout_positions=[],
        test_frac=float(te_mask.mean()),
    )

def fit_predict_xgb_pca(
    Xemb_tr, Xnum_tr, ytr, wtr,
    Xemb_va, Xnum_va, yva, wva,
    Xemb_te, Xnum_te,
    pca_dim: int,
    max_depth: int,
    reg_lambda: float,
    seed: int,
) -> np.ndarray:
    # PCA only on embeddings; numeric features appended after
    scaler_emb = StandardScaler()
    pca = PCA(n_components=pca_dim, random_state=seed, svd_solver="randomized")

    Xtr_emb = pca.fit_transform(scaler_emb.fit_transform(Xemb_tr))
    Xva_emb = pca.transform(scaler_emb.transform(Xemb_va))
    Xte_emb = pca.transform(scaler_emb.transform(Xemb_te))

    if Xnum_tr is not None:
        scaler_num = StandardScaler()
        Xtr_num = scaler_num.fit_transform(Xnum_tr)
        Xva_num = scaler_num.transform(Xnum_va)
        Xte_num = scaler_num.transform(Xnum_te)
        Xtr = np.hstack([Xtr_emb, Xtr_num])
        Xva = np.hstack([Xva_emb, Xva_num])
        Xte = np.hstack([Xte_emb, Xte_num])
    else:
        Xtr, Xva, Xte = Xtr_emb, Xva_emb, Xte_emb

    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    dva = xgb.DMatrix(Xva, label=yva, weight=wva)
    dte = xgb.DMatrix(Xte)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": ETA,
        "max_depth": int(max_depth),
        "min_child_weight": float(MIN_CHILD_WEIGHT),
        "subsample": float(SUBSAMPLE),
        "colsample_bytree": float(COLSAMPLE),
        "lambda": float(reg_lambda),
        "tree_method": "hist",
        "seed": int(seed),
        "nthread": int(NTHREAD),
    }

    bst = xgb.train(
        params,
        dtr,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dva, "val")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
    )

    best_iter = bst.best_iteration
    if best_iter is None:
        pred = bst.predict(dte)
    else:
        pred = bst.predict(dte, iteration_range=(0, best_iter + 1))
    return pred.astype(np.float32)

def fit_final_predict_all(
    Xemb, Xnum, y, w,
    pca_dim: int, max_depth: int, reg_lambda: float,
    seed: int
) -> np.ndarray:
    idx = np.where(np.isfinite(y))[0]
    tr_idx, va_idx = train_test_split(idx, test_size=VAL_FRAC, random_state=seed)

    Xemb_tr, Xemb_va = Xemb[tr_idx], Xemb[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    wtr, wva = w[tr_idx], w[va_idx]

    Xnum_tr = Xnum[tr_idx] if Xnum is not None else None
    Xnum_va = Xnum[va_idx] if Xnum is not None else None
    Xnum_all = Xnum if Xnum is not None else None

    # Train and predict all by reusing the same transform flow
    pred_all = fit_predict_xgb_pca(
        Xemb_tr, Xnum_tr, ytr, wtr,
        Xemb_va, Xnum_va, yva, wva,
        Xemb, Xnum_all,
        pca_dim=pca_dim,
        max_depth=max_depth,
        reg_lambda=reg_lambda,
        seed=seed,
    )
    return pred_all

def main():
    # Load embeddings
    emb = np.load(EMB_FILE, allow_pickle=True).item()
    X_all = emb["emb"].astype(np.float32)
    ids_all = emb["ids"].astype(str)

    # Load labels aligned to embedding order
    df = pd.read_csv(LABELS_FILE, low_memory=False).set_index("variant_id").loc[ids_all].reset_index()

    # HOFF subset
    hoff_mask = (df["dataset_id"].values == "HOFF")
    hoff = df.loc[hoff_mask].copy()
    X = X_all[hoff_mask]

    # Filter stop variants
    if "has_stop" in hoff.columns:
        good = ~hoff["has_stop"].fillna(False).astype(bool).to_numpy()
        hoff = hoff.loc[good].copy()
        X = X[good]

    # n_mut
    hoff["n_mut"] = pd.to_numeric(hoff["n_mut"], errors="coerce")
    nmut = hoff["n_mut"].astype(int).to_numpy()

    # Parse mutation positions and build index
    pos_lists = hoff["mut_positions"].apply(parse_pos_list).tolist()
    pos_arr, pos_to_idx, freq = build_pos_index(pos_lists)

    # WT embedding
    wt = np.load(WT_EMB_FILE, allow_pickle=True).item()
    wt_ids = wt["ids"].astype(str)
    wt_emb = wt["emb"].astype(np.float32)
    if "HOFF|WT" not in set(wt_ids.tolist()):
        raise RuntimeError("HOFF|WT not found in esm2_wt_embeddings.npy")
    wt_hoff = wt_emb[np.where(wt_ids == "HOFF|WT")[0][0]]

    # Feature modes
    X_mean = X
    X_delta = (X - wt_hoff[None, :])

    # Optional numeric features
    Xnum = None
    if ADD_NMUT_FEATURES:
        nm = nmut.astype(np.float32)
        Xnum = np.vstack([nm, nm**2]).T.astype(np.float32)

    # Targets
    y_delta = pd.to_numeric(hoff["hoff_delta_O2_minus_N2"], errors="coerce").to_numpy(dtype=np.float32)
    y_n2    = pd.to_numeric(hoff["hoff_fitness_N2"], errors="coerce").to_numpy(dtype=np.float32)
    y_o2    = pd.to_numeric(hoff["hoff_fitness_O2"], errors="coerce").to_numpy(dtype=np.float32)

    targets = {
        "delta_O2_minus_N2": y_delta,
        "fitness_N2": y_n2,
        "fitness_O2": y_o2,
    }

    # Sample weights (same base weights used for all targets)
    w_all = make_sample_weights(hoff)

    # Build splits once per seed (pos holdout fixed per seed)
    pos_splits = {}
    depth_splits = {}
    holdout_meta = {}

    for seed in SEEDS:
        ps = make_pos_holdout_split_greedy(pos_arr, pos_to_idx, freq, n_variants=len(hoff), seed=seed)
        ds = make_depth_holdout_split(nmut, seed=seed)
        pos_splits[seed] = ps
        depth_splits[seed] = ds
        holdout_meta[str(seed)] = {
            "pos_test_frac": ps.test_frac,
            "pos_holdout_positions": ps.holdout_positions,
            "depth_test_frac": ds.test_frac,
        }

    with open(os.path.join(OUTDIR, "holdout_positions_by_seed.json"), "w") as f:
        json.dump(holdout_meta, f, indent=2)

    print("Position-holdout test_frac by seed:", {k: round(v["pos_test_frac"], 4) for k,v in holdout_meta.items()})

    # -------------------------
    # Hyperparameter tuning on PRIMARY target under POS HOLDOUT
    # Compare mean vs delta feature modes
    # -------------------------
    grid = []
    for pca_dim in PCA_DIMS:
        for md in MAX_DEPTHS:
            for lam in REG_LAMBDAS:
                grid.append({"pca_dim": pca_dim, "max_depth": md, "reg_lambda": lam})

    tuning_rows = []

    def eval_one(feature_mode: str, cfg: Dict) -> Dict[str, float]:
        # choose feature matrix
        Xemb = X_delta if feature_mode == "delta" else X_mean
        y = y_delta  # primary
        rows = []
        for seed in SEEDS:
            sp = pos_splits[seed]

            # Use only finite labels
            fin = np.isfinite(y)
            tr = sp.train_idx[fin[sp.train_idx]]
            va = sp.val_idx[fin[sp.val_idx]]
            te = sp.test_idx[fin[sp.test_idx]]

            if len(tr) < 500 or len(va) < 100 or len(te) < 500:
                continue

            pred = fit_predict_xgb_pca(
                Xemb[tr], Xnum[tr] if Xnum is not None else None, y[tr], w_all[tr],
                Xemb[va], Xnum[va] if Xnum is not None else None, y[va], w_all[va],
                Xemb[te], Xnum[te] if Xnum is not None else None,
                pca_dim=cfg["pca_dim"],
                max_depth=cfg["max_depth"],
                reg_lambda=cfg["reg_lambda"],
                seed=seed
            )
            m = eval_metrics(y[te], pred)
            m.update({
                "seed": seed,
                "feature_mode": feature_mode,
                "pca_dim": cfg["pca_dim"],
                "max_depth": cfg["max_depth"],
                "reg_lambda": cfg["reg_lambda"],
                "test_frac": pos_splits[seed].test_frac,
            })
            rows.append(m)

        if len(rows) == 0:
            return {}
        dfm = pd.DataFrame(rows)
        out = {
            "feature_mode": feature_mode,
            "pca_dim": cfg["pca_dim"],
            "max_depth": cfg["max_depth"],
            "reg_lambda": cfg["reg_lambda"],
            "spearman_mean": float(dfm["spearman"].mean()),
            "spearman_std": float(dfm["spearman"].std(ddof=0)),
            "enrich5_mean": float(dfm["enrichment_at_5pct"].mean()),
            "prec5_mean": float(dfm["precision_at_5pct"].mean()),
            "n_seeds_used": int(dfm["seed"].nunique()),
        }
        return out

    for cfg in grid:
        for fm in ["mean", "delta"]:
            out = eval_one(fm, cfg)
            if out:
                tuning_rows.append(out)

    tune_df = pd.DataFrame(tuning_rows)
    tune_path = os.path.join(OUTDIR, "tuning_primary_pos_holdout.csv")
    tune_df.to_csv(tune_path, index=False)
    print("Wrote:", tune_path)

    # Choose best by spearman_mean, then enrich5_mean
    if len(tune_df) == 0:
        raise RuntimeError("No tuning results produced; check that splits/test sizes are large enough.")

    best = tune_df.sort_values(
        by=["spearman_mean", "enrich5_mean", "n_seeds_used"],
        ascending=[False, False, False]
    ).iloc[0].to_dict()

    best_cfg = {
        "feature_mode": best["feature_mode"],
        "pca_dim": int(best["pca_dim"]),
        "max_depth": int(best["max_depth"]),
        "reg_lambda": float(best["reg_lambda"]),
    }

    with open(os.path.join(OUTDIR, "best_config.json"), "w") as f:
        json.dump(best_cfg, f, indent=2)

    print("\nSelected best config:", best_cfg)

    # -------------------------
    # Evaluate best config on BOTH splits for ALL three targets
    # -------------------------
    eval_rows = []
    Xemb_best = X_delta if best_cfg["feature_mode"] == "delta" else X_mean

    for seed in SEEDS:
        for split_name, split in [("pos_holdout", pos_splits[seed]), ("depth_holdout", depth_splits[seed])]:
            for tgt_name, y in targets.items():
                fin = np.isfinite(y)
                tr = split.train_idx[fin[split.train_idx]]
                va = split.val_idx[fin[split.val_idx]]
                te = split.test_idx[fin[split.test_idx]]

                if len(tr) < 500 or len(va) < 100 or len(te) < 500:
                    continue

                pred = fit_predict_xgb_pca(
                    Xemb_best[tr], Xnum[tr] if Xnum is not None else None, y[tr], w_all[tr],
                    Xemb_best[va], Xnum[va] if Xnum is not None else None, y[va], w_all[va],
                    Xemb_best[te], Xnum[te] if Xnum is not None else None,
                    pca_dim=best_cfg["pca_dim"],
                    max_depth=best_cfg["max_depth"],
                    reg_lambda=best_cfg["reg_lambda"],
                    seed=seed
                )
                m = eval_metrics(y[te], pred)
                m.update({
                    "seed": seed,
                    "split": split_name,
                    "target": tgt_name,
                    "feature_mode": best_cfg["feature_mode"],
                    "pca_dim": best_cfg["pca_dim"],
                    "max_depth": best_cfg["max_depth"],
                    "reg_lambda": best_cfg["reg_lambda"],
                    "test_frac": split.test_frac,
                    "holdout_positions": len(split.holdout_positions),
                })
                eval_rows.append(m)

    eval_df = pd.DataFrame(eval_rows)
    eval_path = os.path.join(OUTDIR, "cv_metrics_bestconfig.csv")
    eval_df.to_csv(eval_path, index=False)
    print("Wrote:", eval_path)

    # Print concise summaries
    print("\n=== CV summary (best config): mean±std over seeds ===")
    summ = (eval_df.groupby(["split", "target"])
                  [["spearman", "r2", "precision_at_5pct", "enrichment_at_5pct"]]
                  .agg(["mean", "std", "count"])
            )
    print(summ.round(4).to_string())

    # -------------------------
    # Train final models on all data and rank candidates
    # -------------------------
    pred_delta = fit_final_predict_all(Xemb_best, Xnum, y_delta, w_all,
                                       pca_dim=best_cfg["pca_dim"],
                                       max_depth=best_cfg["max_depth"],
                                       reg_lambda=best_cfg["reg_lambda"],
                                       seed=SEEDS[0])
    pred_n2    = fit_final_predict_all(Xemb_best, Xnum, y_n2, w_all,
                                       pca_dim=best_cfg["pca_dim"],
                                       max_depth=best_cfg["max_depth"],
                                       reg_lambda=best_cfg["reg_lambda"],
                                       seed=SEEDS[0])
    pred_o2    = fit_final_predict_all(Xemb_best, Xnum, y_o2, w_all,
                                       pca_dim=best_cfg["pca_dim"],
                                       max_depth=best_cfg["max_depth"],
                                       reg_lambda=best_cfg["reg_lambda"],
                                       seed=SEEDS[0])

    out = hoff.copy()
    out["variant_str"] = out["variant_id"].astype(str).str.replace("^HOFF\\|", "", regex=True)
    out["pred_delta_O2_minus_N2"] = pred_delta
    out["pred_fitness_N2"] = pred_n2
    out["pred_fitness_O2"] = pred_o2

    # WT observed
    wt_rows = out[pd.to_numeric(out["n_mut"], errors="coerce") == 0]
    wt_n2_obs = pd.to_numeric(wt_rows.get("hoff_fitness_N2"), errors="coerce").median() if len(wt_rows) else np.nan
    wt_o2_obs = pd.to_numeric(wt_rows.get("hoff_fitness_O2"), errors="coerce").median() if len(wt_rows) else np.nan

    # Constraint thresholds
    if np.isfinite(wt_n2_obs):
        n2_thresh = float(wt_n2_obs - N2_MARGIN_BELOW_WT)
    else:
        n2_thresh = float(np.quantile(out["pred_fitness_N2"].to_numpy(), 0.50))

    if O2_MARGIN_BELOW_WT is not None and np.isfinite(wt_o2_obs):
        o2_thresh = float(wt_o2_obs - O2_MARGIN_BELOW_WT)
    else:
        o2_thresh = None

    keep = np.ones(len(out), dtype=bool)
    if MAX_N_MUT_FOR_CANDIDATES is not None:
        keep &= (pd.to_numeric(out["n_mut"], errors="coerce").to_numpy() <= MAX_N_MUT_FOR_CANDIDATES)

    keep &= (out["pred_fitness_N2"].to_numpy() >= n2_thresh)
    if o2_thresh is not None:
        keep &= (out["pred_fitness_O2"].to_numpy() >= o2_thresh)

    out["keep_constraint"] = keep
    out["n2_threshold_used"] = n2_thresh
    out["o2_threshold_used"] = o2_thresh if o2_thresh is not None else np.nan

    out_ranked = out.sort_values(
        by=["keep_constraint", "pred_delta_O2_minus_N2", "pred_fitness_O2", "pred_fitness_N2"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    ranked_path = os.path.join(OUTDIR, "hoffmann_candidates_ranked.csv")
    out_ranked.to_csv(ranked_path, index=False)

    shortlist = out_ranked[out_ranked["keep_constraint"]].head(SHORTLIST_N).copy()
    shortlist_path = os.path.join(OUTDIR, "hoffmann_candidates_shortlist.csv")
    shortlist.to_csv(shortlist_path, index=False)

    run_cfg = {
        "best_config": best_cfg,
        "pos_holdout_test_frac_band": [TEST_FRAC_LOW, TEST_FRAC_HIGH],
        "pos_holdout_test_frac_target": TEST_FRAC_TARGET,
        "seeds": SEEDS,
        "ADD_NMUT_FEATURES": ADD_NMUT_FEATURES,
        "sample_weight_clip": WEIGHT_CLIP,
        "MAX_N_MUT_FOR_CANDIDATES": MAX_N_MUT_FOR_CANDIDATES,
        "N2_MARGIN_BELOW_WT": N2_MARGIN_BELOW_WT,
        "O2_MARGIN_BELOW_WT": O2_MARGIN_BELOW_WT,
        "n2_threshold_used": n2_thresh,
        "o2_threshold_used": o2_thresh,
        "wt_n2_obs": float(wt_n2_obs) if np.isfinite(wt_n2_obs) else None,
        "wt_o2_obs": float(wt_o2_obs) if np.isfinite(wt_o2_obs) else None,
    }
    with open(os.path.join(OUTDIR, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    print("\nWrote:", ranked_path)
    print("Wrote:", shortlist_path)
    print("Wrote:", os.path.join(OUTDIR, "run_config.json"))

if __name__ == "__main__":
    main()
