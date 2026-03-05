#!/usr/bin/env python3
"""Task B: ProtT5 embedding comparison.

Run the same XGB + TabPFN benchmark pipeline with ProtT5-XL embeddings,
then compare against the published ESM2 v1 results.

Structure matches v1 exactly:
  - DMS (3 targets): within-position (3 split_seeds × 5 model_seeds) + pos-holdout (5 folds × 5 model_seeds)
  - HOFF (delta_O2_minus_N2): depth-holdout (5 model_seeds)
  - TabPFN: all 3 DMS targets (deterministic, 1 run per split/fold) + HOFF
  - PCA=128, canonical v1 XGB params
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
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
PROTT5_NPY = os.path.join(REPO, "results/prott5_embeddings.npy")
LABELS_CSV = os.path.join(REPO, "code/AI_CHEM_code/rubisco_datasets_merged.csv")
V1_SUMMARY = os.path.join(REPO, "_orginal_v1/results/results_pubready_xgb_tabpfn/summary_pub.csv")
OUT_DIR = os.path.join(REPO, "results/results_prott5_comparison")

PCA_DIM = 128
SPLIT_SEEDS = [0, 1, 2]
MODEL_SEEDS = [0, 1, 2, 3, 4]
DMS_TARGETS = ["dms_enrichment_mean", "dms_KmCO2_logfit", "dms_VmaxRatio_logfit"]
TABPFN_TARGETS = DMS_TARGETS  # TabPFN on all 3 DMS targets (matches CSV data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_spearman(y, p):
    c = spearmanr(y, p).correlation
    return float(c) if np.isfinite(c) else np.nan


def topk_precision(y_true, y_pred, frac=0.05):
    k = max(1, int(np.ceil(frac * len(y_true))))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    return len(true_top & pred_top) / k


def topk_enrich_diff(y_true, y_pred, frac=0.05):
    k = max(1, int(np.ceil(frac * len(y_true))))
    pred_top = np.argsort(y_pred)[-k:]
    return float(np.mean(y_true[pred_top]) - np.mean(y_true))


def transform(X, tr, te):
    scaler = StandardScaler()
    pca = PCA(n_components=PCA_DIM, random_state=0, svd_solver="randomized")
    Xtr = pca.fit_transform(scaler.fit_transform(X[tr]))
    Xte = pca.transform(scaler.transform(X[te]))
    return Xtr.astype(np.float32), Xte.astype(np.float32)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def fit_predict_xgb(Xtr, ytr, Xte, seed=0):
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
    if TabPFNRegressor is None:
        raise RuntimeError("tabpfn not installed")
    reg = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
    reg.fit(Xtr, ytr)
    return reg.predict(Xte)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading ProtT5 embeddings …", flush=True)
    emb = np.load(PROTT5_NPY, allow_pickle=True).item()
    ids = emb["ids"].astype(str)
    Xall = emb["emb"].astype(np.float32)
    print(f"  Shape: {Xall.shape}", flush=True)

    print("Loading labels …", flush=True)
    df = pd.read_csv(LABELS_CSV, low_memory=False).set_index("variant_id").loc[ids].reset_index()

    rows = []

    # ===================================================================
    # DMS splits
    # ===================================================================
    dms_mask = df["dataset_id"].eq("DMS").to_numpy()
    dms = df.loc[dms_mask].copy()
    Xdms = Xall[dms_mask]
    pos = pd.to_numeric(dms["position_external"], errors="coerce").to_numpy()

    for target in DMS_TARGETS:
        yd = pd.to_numeric(dms[target], errors="coerce").to_numpy(dtype=np.float32)
        keep = np.isfinite(yd) & np.isfinite(pos)
        Xd, yd_clean, pos_clean = Xdms[keep], yd[keep], pos[keep].astype(int)
        print(f"\n--- Target: {target} (N={len(yd_clean)}) ---", flush=True)

        run_tabpfn = (target in TABPFN_TARGETS)

        # Within-position splits
        for split_seed in SPLIT_SEEDS:
            tr, te = train_test_split(
                np.arange(len(yd_clean)), test_size=0.2,
                random_state=split_seed, stratify=pos_clean,
            )
            Xtr, Xte = transform(Xd, tr, te)
            ytr, yte = yd_clean[tr], yd_clean[te]
            split_id = f"within_seed{split_seed}"

            for ms in MODEL_SEEDS:
                pred = fit_predict_xgb(Xtr, ytr, Xte, seed=ms)
                rows.append({
                    "embedding": "prott5",
                    "target": target,
                    "split": "within_position",
                    "split_id": split_id,
                    "model": "xgb",
                    "model_seed": ms,
                    "n_train": len(ytr),
                    "n_test": len(yte),
                    "spearman": safe_spearman(yte, pred),
                    "top5_precision": topk_precision(yte, pred),
                    "top5_enrich_diff": topk_enrich_diff(yte, pred),
                })

            if run_tabpfn and TabPFNRegressor is not None:
                print(f"  TabPFN {split_id} …", flush=True)
                try:
                    pred = fit_predict_tabpfn(Xtr, ytr, Xte)
                    rows.append({
                        "embedding": "prott5",
                        "target": target,
                        "split": "within_position",
                        "split_id": split_id,
                        "model": "tabpfn",
                        "model_seed": 0,
                        "n_train": len(ytr),
                        "n_test": len(yte),
                        "spearman": safe_spearman(yte, pred),
                        "top5_precision": topk_precision(yte, pred),
                        "top5_enrich_diff": topk_enrich_diff(yte, pred),
                    })
                    print(f"    spearman={rows[-1]['spearman']:.4f}", flush=True)
                except Exception as e:
                    print(f"    TabPFN FAILED: {e}", flush=True)

            print(f"  XGB {split_id}: spearman(mean 5 seeds)="
                  f"{np.mean([r['spearman'] for r in rows if r['split_id']==split_id and r['model']=='xgb']):.4f}",
                  flush=True)

        # Position-holdout splits
        gkf = GroupKFold(n_splits=min(5, len(np.unique(pos_clean))))
        for fold, (tr, te) in enumerate(gkf.split(Xd, yd_clean, groups=pos_clean), start=1):
            Xtr, Xte = transform(Xd, tr, te)
            ytr, yte = yd_clean[tr], yd_clean[te]
            split_id = f"pos_holdout_fold{fold}"

            for ms in MODEL_SEEDS:
                pred = fit_predict_xgb(Xtr, ytr, Xte, seed=ms)
                rows.append({
                    "embedding": "prott5",
                    "target": target,
                    "split": "pos_holdout",
                    "split_id": split_id,
                    "model": "xgb",
                    "model_seed": ms,
                    "n_train": len(ytr),
                    "n_test": len(yte),
                    "spearman": safe_spearman(yte, pred),
                    "top5_precision": topk_precision(yte, pred),
                    "top5_enrich_diff": topk_enrich_diff(yte, pred),
                })

            if run_tabpfn and TabPFNRegressor is not None:
                print(f"  TabPFN {split_id} …", flush=True)
                try:
                    pred = fit_predict_tabpfn(Xtr, ytr, Xte)
                    rows.append({
                        "embedding": "prott5",
                        "target": target,
                        "split": "pos_holdout",
                        "split_id": split_id,
                        "model": "tabpfn",
                        "model_seed": 0,
                        "n_train": len(ytr),
                        "n_test": len(yte),
                        "spearman": safe_spearman(yte, pred),
                        "top5_precision": topk_precision(yte, pred),
                        "top5_enrich_diff": topk_enrich_diff(yte, pred),
                    })
                    print(f"    spearman={rows[-1]['spearman']:.4f}", flush=True)
                except Exception as e:
                    print(f"    TabPFN FAILED: {e}", flush=True)

            print(f"  XGB {split_id}: spearman(mean 5 seeds)="
                  f"{np.mean([r['spearman'] for r in rows if r['split_id']==split_id and r['model']=='xgb']):.4f}",
                  flush=True)

    # ===================================================================
    # HOFF depth-holdout
    # ===================================================================
    print("\n--- HOFF depth-holdout ---", flush=True)
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
    print(f"  N_train={len(tr)} (n_mut<=4), N_test={len(te)} (n_mut>=6)", flush=True)

    Xtr, Xte = transform(Xh, tr, te)
    ytr, yte = yh[tr], yh[te]

    for ms in MODEL_SEEDS:
        pred = fit_predict_xgb(Xtr, ytr, Xte, seed=ms)
        rows.append({
            "embedding": "prott5",
            "target": "hoff_delta_O2_minus_N2",
            "split": "depth_holdout",
            "split_id": "depth_holdout",
            "model": "xgb",
            "model_seed": ms,
            "n_train": len(ytr),
            "n_test": len(yte),
            "spearman": safe_spearman(yte, pred),
            "top5_precision": topk_precision(yte, pred),
            "top5_enrich_diff": topk_enrich_diff(yte, pred),
        })
    xgb_sp = np.mean([r["spearman"] for r in rows if r["target"]=="hoff_delta_O2_minus_N2" and r["model"]=="xgb"])
    print(f"  XGB spearman(mean 5 seeds): {xgb_sp:.4f}", flush=True)

    if TabPFNRegressor is not None:
        print("  TabPFN HOFF …", flush=True)
        try:
            pred = fit_predict_tabpfn(Xtr, ytr, Xte)
            rows.append({
                "embedding": "prott5",
                "target": "hoff_delta_O2_minus_N2",
                "split": "depth_holdout",
                "split_id": "depth_holdout",
                "model": "tabpfn",
                "model_seed": 0,
                "n_train": len(ytr),
                "n_test": len(yte),
                "spearman": safe_spearman(yte, pred),
                "top5_precision": topk_precision(yte, pred),
                "top5_enrich_diff": topk_enrich_diff(yte, pred),
            })
            print(f"    spearman={rows[-1]['spearman']:.4f}", flush=True)
        except Exception as e:
            print(f"    TabPFN FAILED: {e}", flush=True)

    # ===================================================================
    # Save raw runs
    # ===================================================================
    runs_df = pd.DataFrame(rows)
    runs_path = os.path.join(OUT_DIR, "prott5_runs_raw.csv")
    runs_df.to_csv(runs_path, index=False)
    print(f"\nWrote {runs_path} ({len(runs_df)} rows)", flush=True)

    # ===================================================================
    # Summary (aggregate like v1: average model_seeds within split_id, then mean/std over split_ids)
    # ===================================================================
    summary_rows = []
    for (target, split, model), grp in runs_df.groupby(["target", "split", "model"]):
        # Average over model_seeds within each split_id
        unit_means = grp.groupby("split_id")["spearman"].mean()
        unit_top5 = grp.groupby("split_id")["top5_precision"].mean()
        unit_enrich = grp.groupby("split_id")["top5_enrich_diff"].mean()

        summary_rows.append({
            "embedding": "prott5",
            "target": target,
            "split": split,
            "model": model,
            "n_units": len(unit_means),
            "spearman_mean": unit_means.mean(),
            "spearman_std": unit_means.std() if len(unit_means) > 1 else 0.0,
            "top5_precision_mean": unit_top5.mean(),
            "top5_enrich_diff_mean": unit_enrich.mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "prott5_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}", flush=True)

    # ===================================================================
    # Comparison with ESM2 v1
    # ===================================================================
    lines = []
    lines.append("Task B: ProtT5 vs ESM2 Embedding Comparison")
    lines.append("=" * 72)
    lines.append(f"ProtT5 dim: 1024, ESM2 dim: 1280, PCA: {PCA_DIM}")
    lines.append(f"Model seeds: {MODEL_SEEDS}")
    lines.append("")

    # Load v1 ESM2 summary (fixed config only)
    v1 = pd.read_csv(V1_SUMMARY)
    v1_fixed = v1[v1["config_label"] == "fixed"].copy()

    # Map v1 columns to our schema
    # v1 has: dataset, task_name, target, split, model, spearman_mean, etc.
    lines.append(f"{'Target':<28} {'Split':<18} {'Model':<8} {'ESM2 ρ':>10} {'ProtT5 ρ':>10} {'Δ (P-E)':>10}")
    lines.append("-" * 90)

    for _, row in summary_df.iterrows():
        target = row["target"]
        split = row["split"]
        model = row["model"]

        # Find matching v1 row
        if "hoff" in target:
            v1_match = v1_fixed[
                (v1_fixed["target"] == target) &
                (v1_fixed["split"] == split) &
                (v1_fixed["model"] == model) &
                (v1_fixed["task_name"] == "HOFF_delta_direct")
            ]
        else:
            v1_match = v1_fixed[
                (v1_fixed["target"] == target) &
                (v1_fixed["split"] == split) &
                (v1_fixed["model"] == model)
            ]

        esm2_sp = v1_match["spearman_mean"].values[0] if len(v1_match) > 0 else np.nan
        prott5_sp = row["spearman_mean"]
        delta = prott5_sp - esm2_sp if np.isfinite(esm2_sp) else np.nan

        delta_str = f"{delta:+.4f}" if np.isfinite(delta) else "N/A"
        esm2_str = f"{esm2_sp:.4f}" if np.isfinite(esm2_sp) else "N/A"
        lines.append(f"{target:<28} {split:<18} {model:<8} {esm2_str:>10} {prott5_sp:>10.4f} {delta_str:>10}")

    lines.append("")
    lines.append("Summary:")
    lines.append("-" * 72)

    # Overall delta for XGB
    xgb_deltas = []
    for _, row in summary_df[summary_df["model"] == "xgb"].iterrows():
        target = row["target"]
        split = row["split"]
        if "hoff" in target:
            v1_match = v1_fixed[
                (v1_fixed["target"] == target) &
                (v1_fixed["split"] == split) &
                (v1_fixed["model"] == "xgb") &
                (v1_fixed["task_name"] == "HOFF_delta_direct")
            ]
        else:
            v1_match = v1_fixed[
                (v1_fixed["target"] == target) &
                (v1_fixed["split"] == split) &
                (v1_fixed["model"] == "xgb")
            ]
        if len(v1_match) > 0:
            xgb_deltas.append(row["spearman_mean"] - v1_match["spearman_mean"].values[0])

    tab_deltas = []
    for _, row in summary_df[summary_df["model"] == "tabpfn"].iterrows():
        target = row["target"]
        split = row["split"]
        if "hoff" in target:
            v1_match = v1_fixed[
                (v1_fixed["target"] == target) &
                (v1_fixed["split"] == split) &
                (v1_fixed["model"] == "tabpfn") &
                (v1_fixed["task_name"] == "HOFF_delta_direct")
            ]
        else:
            v1_match = v1_fixed[
                (v1_fixed["target"] == target) &
                (v1_fixed["split"] == split) &
                (v1_fixed["model"] == "tabpfn")
            ]
        if len(v1_match) > 0:
            tab_deltas.append(row["spearman_mean"] - v1_match["spearman_mean"].values[0])

    if xgb_deltas:
        lines.append(f"  XGB mean Δ(ProtT5 - ESM2) across all tasks: {np.mean(xgb_deltas):+.4f}")
    if tab_deltas:
        lines.append(f"  TabPFN mean Δ(ProtT5 - ESM2) across tasks:  {np.mean(tab_deltas):+.4f}")

    overall = xgb_deltas + tab_deltas
    if overall:
        direction = "ProtT5 > ESM2" if np.mean(overall) > 0 else "ESM2 > ProtT5" if np.mean(overall) < 0 else "tied"
        lines.append(f"  Overall direction: {direction} (mean Δ = {np.mean(overall):+.4f})")

    elapsed = time.time() - t0
    lines.append(f"\nTotal runtime: {elapsed/60:.1f} min")

    comparison_text = "\n".join(lines)
    print("\n" + comparison_text, flush=True)

    comp_path = os.path.join(OUT_DIR, "prott5_vs_esm2_comparison.txt")
    with open(comp_path, "w") as f:
        f.write(comparison_text + "\n")
    print(f"\nWrote {comp_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
