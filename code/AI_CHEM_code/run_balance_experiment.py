#!/usr/bin/env python3
"""Task A: Balance experiment — XGBoost vs TabPFN under quantile-balanced training.

Target: dms_enrichment_mean, within-position split, seeds 0/1/2.
Two conditions: original (full train set) and quantile_balanced (4 quantile bins).
PCA=128, canonical v1 XGB params throughout.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

try:
    from tabpfn import TabPFNRegressor
except Exception:
    TabPFNRegressor = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = "/mmfs1/scratch/jacks.local/jyoung67391/Rubisco_Protein_Language_Model_Active_Learning"
EMB_NPY = "/mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed/esm2_t33_650m_full.npy"
LABELS_CSV = os.path.join(REPO, "code/AI_CHEM_code/rubisco_datasets_merged.csv")
OUT_CSV = os.path.join(REPO, "results/balance_experiment.csv")
OUT_TXT = os.path.join(REPO, "results/balance_experiment_summary.txt")

PCA_DIM = 128
SEEDS = [0, 1, 2]
TARGET = "dms_enrichment_mean"
N_BINS = 4  # quantile bins for balancing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def quantile_balance_indices(y, n_bins=4, seed=42):
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


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def fit_predict_xgb(Xtr, ytr, Xte, seed=0):
    """Canonical v1 XGBoost."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ytr))
    tr_idx, va_idx = train_test_split(
        idx, test_size=0.10, random_state=int(rng.integers(1e9))
    )
    dtr = xgb.DMatrix(Xtr[tr_idx], label=ytr[tr_idx])
    dva = xgb.DMatrix(Xtr[va_idx], label=ytr[va_idx])
    dte = xgb.DMatrix(Xte)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 10.0,
        "tree_method": "hist",
        "seed": seed,
        "nthread": 16,
    }
    bst = xgb.train(
        params, dtr,
        num_boost_round=8000,
        evals=[(dva, "val")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    bi = bst.best_iteration
    pred = bst.predict(dte) if bi is None else bst.predict(dte, iteration_range=(0, bi + 1))
    return pred.astype(np.float32)


def fit_predict_tabpfn(Xtr, ytr, Xte):
    """TabPFN regressor (CPU)."""
    if TabPFNRegressor is None:
        raise RuntimeError("tabpfn not installed")
    reg = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
    reg.fit(Xtr, ytr)
    return reg.predict(Xte)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading embeddings …", flush=True)
    emb = np.load(EMB_NPY, allow_pickle=True).item()
    ids = emb["ids"].astype(str)
    Xall = emb["emb"].astype(np.float32)

    print("Loading labels …", flush=True)
    df = pd.read_csv(LABELS_CSV).set_index("variant_id").loc[ids].reset_index()

    # Filter to DMS with valid enrichment_mean
    dms_mask = df["dataset_id"].eq("DMS").to_numpy()
    dms = df.loc[dms_mask].copy()
    Xd = Xall[dms_mask]
    yd = pd.to_numeric(dms[TARGET], errors="coerce").to_numpy(dtype=np.float32)
    pos = pd.to_numeric(dms["position_external"], errors="coerce").to_numpy()
    keep = np.isfinite(yd) & np.isfinite(pos)
    Xd, yd, pos = Xd[keep], yd[keep], pos[keep].astype(int)
    print(f"DMS subset: {len(yd)} variants, {len(np.unique(pos))} positions", flush=True)

    rows = []

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===", flush=True)

        # Stratified 80/20 split by position
        tr, te = train_test_split(
            np.arange(len(yd)), test_size=0.2, random_state=seed, stratify=pos
        )

        # PCA transform (canonical: random_state=0, svd_solver=randomized)
        scaler = StandardScaler()
        pca = PCA(n_components=PCA_DIM, random_state=0, svd_solver="randomized")
        Xtr_full = pca.fit_transform(scaler.fit_transform(Xd[tr])).astype(np.float32)
        Xte_pca = pca.transform(scaler.transform(Xd[te])).astype(np.float32)
        ytr_full = yd[tr]
        yte = yd[te]

        # Quantile-balanced subset
        bal_idx = quantile_balance_indices(ytr_full, n_bins=N_BINS, seed=seed)
        Xtr_bal = Xtr_full[bal_idx]
        ytr_bal = ytr_full[bal_idx]
        print(f"  Train original: {len(ytr_full)}, balanced: {len(ytr_bal)}, test: {len(yte)}", flush=True)

        for condition, Xtr, ytr in [
            ("original", Xtr_full, ytr_full),
            ("quantile_balanced", Xtr_bal, ytr_bal),
        ]:
            # --- XGBoost ---
            print(f"  [{condition}] XGBoost …", flush=True)
            pred = fit_predict_xgb(Xtr, ytr, Xte_pca, seed=seed)
            rows.append({
                "condition": condition,
                "model": "xgb",
                "seed": seed,
                "spearman": safe_spearman(yte, pred),
                "top5_precision": topk_precision(yte, pred, 0.05),
                "top5_enrichment_diff": topk_enrich_diff(yte, pred, 0.05),
                "n_train": len(ytr),
                "n_test": len(yte),
            })
            print(f"    spearman={rows[-1]['spearman']:.4f}", flush=True)

            # --- TabPFN ---
            if TabPFNRegressor is not None:
                print(f"  [{condition}] TabPFN …", flush=True)
                try:
                    pred = fit_predict_tabpfn(Xtr, ytr, Xte_pca)
                    rows.append({
                        "condition": condition,
                        "model": "tabpfn",
                        "seed": seed,
                        "spearman": safe_spearman(yte, pred),
                        "top5_precision": topk_precision(yte, pred, 0.05),
                        "top5_enrichment_diff": topk_enrich_diff(yte, pred, 0.05),
                        "n_train": len(ytr),
                        "n_test": len(yte),
                    })
                    print(f"    spearman={rows[-1]['spearman']:.4f}", flush=True)
                except Exception as e:
                    print(f"    TabPFN FAILED: {e}", flush=True)
            else:
                print(f"  [{condition}] TabPFN skipped (not installed)", flush=True)

    # Save raw results
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}", flush=True)

    # Summary table
    lines = []
    lines.append("Balance Experiment Summary (Task A)")
    lines.append("=" * 60)
    lines.append(f"Target: {TARGET}")
    lines.append(f"Split: within-position (stratified 80/20)")
    lines.append(f"Seeds: {SEEDS}")
    lines.append(f"PCA: {PCA_DIM}")
    lines.append(f"Quantile bins: {N_BINS}")
    lines.append("")

    lines.append(f"{'Condition':<22} {'Model':<8} {'Spearman':>18} {'Top5% Prec':>18} {'Top5% Enrich':>18}")
    lines.append("-" * 88)

    for (cond, model), grp in results_df.groupby(["condition", "model"]):
        sp = grp["spearman"]
        tp = grp["top5_precision"]
        te = grp["top5_enrichment_diff"]
        lines.append(
            f"{cond:<22} {model:<8} "
            f"{sp.mean():.4f} ± {sp.std():.4f}  "
            f"{tp.mean():.4f} ± {tp.std():.4f}  "
            f"{te.mean():.4f} ± {te.std():.4f}"
        )

    lines.append("")
    lines.append("Gap Analysis: TabPFN - XGBoost Spearman")
    lines.append("-" * 60)

    for cond in ["original", "quantile_balanced"]:
        sub = results_df[results_df["condition"] == cond]
        xgb_sp = sub[sub["model"] == "xgb"]["spearman"].values
        tab_sp = sub[sub["model"] == "tabpfn"]["spearman"].values
        if len(xgb_sp) > 0 and len(tab_sp) > 0:
            gap = tab_sp - xgb_sp
            lines.append(f"  {cond}: gap = {gap.mean():.4f} ± {gap.std():.4f}")

    # Narrowing analysis
    orig_sub = results_df[results_df["condition"] == "original"]
    bal_sub = results_df[results_df["condition"] == "quantile_balanced"]

    orig_xgb = orig_sub[orig_sub["model"] == "xgb"]["spearman"].values
    orig_tab = orig_sub[orig_sub["model"] == "tabpfn"]["spearman"].values
    bal_xgb = bal_sub[bal_sub["model"] == "xgb"]["spearman"].values
    bal_tab = bal_sub[bal_sub["model"] == "tabpfn"]["spearman"].values

    if len(orig_xgb) > 0 and len(orig_tab) > 0 and len(bal_xgb) > 0 and len(bal_tab) > 0:
        orig_gap = (orig_tab - orig_xgb).mean()
        bal_gap = (bal_tab - bal_xgb).mean()
        lines.append("")
        lines.append(f"  Original gap (TabPFN - XGB):  {orig_gap:.4f}")
        lines.append(f"  Balanced gap (TabPFN - XGB):  {bal_gap:.4f}")
        delta = bal_gap - orig_gap
        direction = "NARROWED" if delta < 0 else "WIDENED" if delta > 0 else "UNCHANGED"
        lines.append(f"  Delta: {delta:+.4f} ({direction})")
        lines.append("")
        lines.append(f"Does the Spearman gap narrow under quantile_balanced? {direction}")
        lines.append(f"By how much? {abs(delta):.4f}")

    summary_text = "\n".join(lines)
    print("\n" + summary_text, flush=True)

    with open(OUT_TXT, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nWrote {OUT_TXT}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
