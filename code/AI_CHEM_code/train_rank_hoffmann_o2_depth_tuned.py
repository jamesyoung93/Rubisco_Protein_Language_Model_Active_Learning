import os, json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

import xgboost as xgb

# -----------------------------
# FILES
# -----------------------------
EMB_FILE = "esm2_t33_650m_full.npy"
WT_EMB_FILE = "esm2_wt_embeddings.npy"
LABELS_FILE = "rubisco_datasets_merged.csv"
OUTDIR = "results_full_eval/hoffmann_depth_tuned"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# SPLITS / SEEDS
# -----------------------------
SEEDS = [0, 1, 2, 3, 4]
VAL_FRAC = 0.10

# Depth split definition
TRAIN_MAX_NMUT = 4
VAL_NMUT = 5
TEST_MIN_NMUT = 6

# -----------------------------
# MODEL / GRID
# -----------------------------
PCA_DIMS = [64, 128, 256]
MAX_DEPTHS = [4, 6]
REG_LAMBDAS = [1.0, 10.0]

ETA = 0.03
SUBSAMPLE = 0.85
COLSAMPLE = 0.85
MIN_CHILD_WEIGHT = 1.0
NUM_BOOST_ROUND = 8000
EARLY_STOPPING = 200
NTHREAD = 16

ADD_NMUT_FEATURES = True

# Sample weights
WEIGHT_CLIP = (0.25, 4.0)

# Candidate selection
MAX_N_MUT_FOR_CANDIDATES = 8
N2_MARGIN_BELOW_WT = 0.10
O2_MARGIN_BELOW_WT = None
SHORTLIST_N = 300

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

def make_sample_weights(hoff: pd.DataFrame) -> np.ndarray:
    n = len(hoff)

    def num(col):
        if col in hoff.columns:
            return pd.to_numeric(hoff[col], errors="coerce").to_numpy()
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
    w = np.clip(w, WEIGHT_CLIP[0], WEIGHT_CLIP[1])
    return w.astype(np.float32)

@dataclass
class DepthSplit:
    tr_idx: np.ndarray
    va_idx: np.ndarray
    te_idx: np.ndarray

def make_depth_split(nmut: np.ndarray, seed: int) -> DepthSplit:
    tr_mask = (nmut <= TRAIN_MAX_NMUT)
    va_mask = (nmut == VAL_NMUT)
    te_mask = (nmut >= TEST_MIN_NMUT)

    # fallback val if sparse
    if va_mask.sum() < 50:
        tr_all = np.where(tr_mask)[0]
        tr_idx, va_idx = train_test_split(tr_all, test_size=VAL_FRAC, random_state=seed)
    else:
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]

    # fallback test if sparse
    if te_mask.sum() < 50:
        te_mask = (nmut >= VAL_NMUT)
    te_idx = np.where(te_mask)[0]

    return DepthSplit(tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx)

def fit_predict_xgb_pca(
    Xemb_tr, Xnum_tr, ytr, wtr,
    Xemb_va, Xnum_va, yva, wva,
    Xemb_te, Xnum_te,
    pca_dim: int,
    max_depth: int,
    reg_lambda: float,
    seed: int,
) -> np.ndarray:
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
    bi = bst.best_iteration
    if bi is None:
        pred = bst.predict(dte)
    else:
        pred = bst.predict(dte, iteration_range=(0, bi + 1))
    return pred.astype(np.float32)

def fit_final_predict_all(
    Xemb, Xnum, y, w,
    pca_dim: int, max_depth: int, reg_lambda: float,
    seed: int
) -> np.ndarray:
    idx = np.where(np.isfinite(y))[0]
    tr_idx, va_idx = train_test_split(idx, test_size=VAL_FRAC, random_state=seed)

    pred_all = fit_predict_xgb_pca(
        Xemb[tr_idx], Xnum[tr_idx] if Xnum is not None else None, y[tr_idx], w[tr_idx],
        Xemb[va_idx], Xnum[va_idx] if Xnum is not None else None, y[va_idx], w[va_idx],
        Xemb, Xnum,
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

    # Load labels aligned to embeddings
    df = pd.read_csv(LABELS_FILE, low_memory=False).set_index("variant_id").loc[ids_all].reset_index()

    hoff_mask = (df["dataset_id"].values == "HOFF")
    hoff = df.loc[hoff_mask].copy()
    X = X_all[hoff_mask]

    # Filter stop variants
    if "has_stop" in hoff.columns:
        good = ~hoff["has_stop"].fillna(False).astype(bool).to_numpy()
        hoff = hoff.loc[good].copy()
        X = X[good]

    hoff["n_mut"] = pd.to_numeric(hoff["n_mut"], errors="coerce")
    nmut = hoff["n_mut"].astype(int).to_numpy()

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

    # Numeric features
    Xnum = None
    if ADD_NMUT_FEATURES:
        nm = nmut.astype(np.float32)
        Xnum = np.vstack([nm, nm**2]).T.astype(np.float32)

    # Targets
    y_delta = pd.to_numeric(hoff["hoff_delta_O2_minus_N2"], errors="coerce").to_numpy(dtype=np.float32)
    y_n2    = pd.to_numeric(hoff["hoff_fitness_N2"], errors="coerce").to_numpy(dtype=np.float32)
    y_o2    = pd.to_numeric(hoff["hoff_fitness_O2"], errors="coerce").to_numpy(dtype=np.float32)

    # Weights
    w_all = make_sample_weights(hoff)

    # Splits
    splits = {seed: make_depth_split(nmut, seed) for seed in SEEDS}
    print("Depth split sizes (seed -> n_train/n_val/n_test):")
    for seed, sp in splits.items():
        print(seed, len(sp.tr_idx), len(sp.va_idx), len(sp.te_idx))

    # Grid search comparing:
    # - feature_mode in {mean, delta}
    # - delta_method in {direct, derived}
    grid = [{"pca_dim": p, "max_depth": md, "reg_lambda": lam}
            for p in PCA_DIMS for md in MAX_DEPTHS for lam in REG_LAMBDAS]

    rows = []

    def run_one(feature_mode: str, cfg: Dict, delta_method: str) -> Dict:
        Xemb = X_delta if feature_mode == "delta" else X_mean
        per_seed = []

        for seed in SEEDS:
            sp = splits[seed]

            # indices restricted to finite labels for whichever we train
            def filt(idx, y):
                idx = np.asarray(idx)
                return idx[np.isfinite(y[idx])]

            tr = sp.tr_idx
            va = sp.va_idx
            te = sp.te_idx

            if delta_method == "direct":
                tr2 = filt(tr, y_delta); va2 = filt(va, y_delta); te2 = filt(te, y_delta)
                if min(len(tr2), len(va2), len(te2)) < 200:
                    continue
                pred = fit_predict_xgb_pca(
                    Xemb[tr2], Xnum[tr2] if Xnum is not None else None, y_delta[tr2], w_all[tr2],
                    Xemb[va2], Xnum[va2] if Xnum is not None else None, y_delta[va2], w_all[va2],
                    Xemb[te2], Xnum[te2] if Xnum is not None else None,
                    pca_dim=cfg["pca_dim"], max_depth=cfg["max_depth"], reg_lambda=cfg["reg_lambda"], seed=seed
                )
                m = eval_metrics(y_delta[te2], pred)
                per_seed.append(m)

            elif delta_method == "derived":
                # fit O2 and N2 then subtract
                tr_o2 = filt(tr, y_o2); va_o2 = filt(va, y_o2); te_o2 = filt(te, y_o2)
                tr_n2 = filt(tr, y_n2); va_n2 = filt(va, y_n2); te_n2 = filt(te, y_n2)

                # evaluation requires delta label on test too
                te_delta = filt(te, y_delta)
                if min(len(tr_o2), len(va_o2), len(te_delta), len(tr_n2), len(va_n2)) < 200:
                    continue

                pred_o2 = fit_predict_xgb_pca(
                    Xemb[tr_o2], Xnum[tr_o2] if Xnum is not None else None, y_o2[tr_o2], w_all[tr_o2],
                    Xemb[va_o2], Xnum[va_o2] if Xnum is not None else None, y_o2[va_o2], w_all[va_o2],
                    Xemb[te_delta], Xnum[te_delta] if Xnum is not None else None,
                    pca_dim=cfg["pca_dim"], max_depth=cfg["max_depth"], reg_lambda=cfg["reg_lambda"], seed=seed
                )
                pred_n2 = fit_predict_xgb_pca(
                    Xemb[tr_n2], Xnum[tr_n2] if Xnum is not None else None, y_n2[tr_n2], w_all[tr_n2],
                    Xemb[va_n2], Xnum[va_n2] if Xnum is not None else None, y_n2[va_n2], w_all[va_n2],
                    Xemb[te_delta], Xnum[te_delta] if Xnum is not None else None,
                    pca_dim=cfg["pca_dim"], max_depth=cfg["max_depth"], reg_lambda=cfg["reg_lambda"], seed=seed
                )
                pred = pred_o2 - pred_n2
                m = eval_metrics(y_delta[te_delta], pred)
                per_seed.append(m)
            else:
                raise ValueError(delta_method)

        if not per_seed:
            return {}

        dfm = pd.DataFrame(per_seed)
        return {
            "feature_mode": feature_mode,
            "delta_method": delta_method,
            "pca_dim": cfg["pca_dim"],
            "max_depth": cfg["max_depth"],
            "reg_lambda": cfg["reg_lambda"],
            "spearman_mean": float(dfm["spearman"].mean()),
            "spearman_std": float(dfm["spearman"].std(ddof=0)),
            "enrich5_mean": float(dfm["enrichment_at_5pct"].mean()),
            "prec5_mean": float(dfm["precision_at_5pct"].mean()),
            "n_seeds_used": int(len(dfm)),
        }

    for cfg in grid:
        for fm in ["mean", "delta"]:
            for dm in ["direct", "derived"]:
                out = run_one(fm, cfg, dm)
                if out:
                    rows.append(out)

    tune = pd.DataFrame(rows)
    tune_path = os.path.join(OUTDIR, "tuning_depth_holdout_delta.csv")
    tune.to_csv(tune_path, index=False)
    print("Wrote:", tune_path)

    if len(tune) == 0:
        raise RuntimeError("No tuning rows produced.")

    best = tune.sort_values(
        by=["spearman_mean", "enrich5_mean", "n_seeds_used"],
        ascending=[False, False, False]
    ).iloc[0].to_dict()

    best_cfg = {
        "feature_mode": best["feature_mode"],
        "delta_method": best["delta_method"],
        "pca_dim": int(best["pca_dim"]),
        "max_depth": int(best["max_depth"]),
        "reg_lambda": float(best["reg_lambda"]),
    }
    with open(os.path.join(OUTDIR, "best_config.json"), "w") as f:
        json.dump(best_cfg, f, indent=2)

    print("\nSelected best config:", best_cfg)

    # Train final models on all data and rank candidates
    Xemb_best = X_delta if best_cfg["feature_mode"] == "delta" else X_mean

    # Always train O2 and N2 for constraints; delta may be derived or direct
    pred_o2_all = fit_final_predict_all(Xemb_best, Xnum, y_o2, w_all,
                                        pca_dim=best_cfg["pca_dim"], max_depth=best_cfg["max_depth"],
                                        reg_lambda=best_cfg["reg_lambda"], seed=SEEDS[0])
    pred_n2_all = fit_final_predict_all(Xemb_best, Xnum, y_n2, w_all,
                                        pca_dim=best_cfg["pca_dim"], max_depth=best_cfg["max_depth"],
                                        reg_lambda=best_cfg["reg_lambda"], seed=SEEDS[0])

    if best_cfg["delta_method"] == "derived":
        pred_delta_all = pred_o2_all - pred_n2_all
    else:
        pred_delta_all = fit_final_predict_all(Xemb_best, Xnum, y_delta, w_all,
                                               pca_dim=best_cfg["pca_dim"], max_depth=best_cfg["max_depth"],
                                               reg_lambda=best_cfg["reg_lambda"], seed=SEEDS[0])

    out = hoff.copy()
    out["variant_str"] = out["variant_id"].astype(str).str.replace("^HOFF\\|", "", regex=True)
    out["pred_delta_O2_minus_N2"] = pred_delta_all
    out["pred_fitness_O2"] = pred_o2_all
    out["pred_fitness_N2"] = pred_n2_all

    # WT observed thresholds
    wt_rows = out[pd.to_numeric(out["n_mut"], errors="coerce") == 0]
    wt_n2_obs = pd.to_numeric(wt_rows.get("hoff_fitness_N2"), errors="coerce").median() if len(wt_rows) else np.nan
    wt_o2_obs = pd.to_numeric(wt_rows.get("hoff_fitness_O2"), errors="coerce").median() if len(wt_rows) else np.nan

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
        "seeds": SEEDS,
        "TRAIN_MAX_NMUT": TRAIN_MAX_NMUT,
        "VAL_NMUT": VAL_NMUT,
        "TEST_MIN_NMUT": TEST_MIN_NMUT,
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
