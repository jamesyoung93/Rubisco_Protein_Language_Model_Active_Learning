#!/usr/bin/env python3
"""Fill missing ProtT5+TabPFN results for KmCO2 and VmaxRatio.

Runs TabPFN-2.5 on ProtT5 embeddings for the two DMS targets that were
skipped in the original Task B run due to CPU cost.  Uses identical PCA=128,
preprocessing, and split definitions.  Appends to prott5_runs_raw.csv.
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

REPO = "/mmfs1/scratch/jacks.local/jyoung67391/Rubisco_Protein_Language_Model_Active_Learning"
PROTT5_NPY = os.path.join(REPO, "results/prott5_embeddings.npy")
LABELS_CSV = os.path.join(REPO, "code/AI_CHEM_code/rubisco_datasets_merged.csv")
RAW_CSV = os.path.join(REPO, "results/results_prott5_comparison/prott5_runs_raw.csv")

PCA_DIM = 128
SPLIT_SEEDS = [0, 1, 2]
TARGETS = ["dms_KmCO2_logfit", "dms_VmaxRatio_logfit"]


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


def fit_predict_tabpfn(Xtr, ytr, Xte):
    reg = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
    reg.fit(Xtr, ytr)
    return reg.predict(Xte)


def main():
    t0 = time.time()

    # Count existing rows
    old_df = pd.read_csv(RAW_CSV)
    old_count = len(old_df)
    print(f"Existing prott5_runs_raw.csv: {old_count} rows", flush=True)

    # Load embeddings
    print("Loading ProtT5 embeddings ...", flush=True)
    emb = np.load(PROTT5_NPY, allow_pickle=True).item()
    ids = emb["ids"].astype(str)
    Xall = emb["emb"].astype(np.float32)
    print(f"  Shape: {Xall.shape}", flush=True)

    # Load labels
    df = pd.read_csv(LABELS_CSV, low_memory=False).set_index("variant_id").loc[ids].reset_index()

    # DMS subset
    dms_mask = df["dataset_id"].eq("DMS").to_numpy()
    dms = df.loc[dms_mask].copy()
    Xdms = Xall[dms_mask]
    pos = pd.to_numeric(dms["position_external"], errors="coerce").to_numpy()

    rows = []

    for target in TARGETS:
        yd = pd.to_numeric(dms[target], errors="coerce").to_numpy(dtype=np.float32)
        keep = np.isfinite(yd) & np.isfinite(pos)
        Xd, yd_clean, pos_clean = Xdms[keep], yd[keep], pos[keep].astype(int)
        print(f"\n--- Target: {target} (N={len(yd_clean)}) ---", flush=True)

        # Within-position splits
        for split_seed in SPLIT_SEEDS:
            tr, te = train_test_split(
                np.arange(len(yd_clean)), test_size=0.2,
                random_state=split_seed, stratify=pos_clean,
            )
            Xtr, Xte = transform(Xd, tr, te)
            ytr, yte = yd_clean[tr], yd_clean[te]
            split_id = f"within_seed{split_seed}"

            print(f"  TabPFN {split_id} ...", flush=True)
            pred = fit_predict_tabpfn(Xtr, ytr, Xte)
            sp = safe_spearman(yte, pred)
            rows.append({
                "embedding": "prott5",
                "target": target,
                "split": "within_position",
                "split_id": split_id,
                "model": "tabpfn",
                "model_seed": 0,
                "n_train": len(ytr),
                "n_test": len(yte),
                "spearman": sp,
                "top5_precision": topk_precision(yte, pred),
                "top5_enrich_diff": topk_enrich_diff(yte, pred),
            })
            print(f"    spearman={sp:.4f}", flush=True)

        # Position-holdout splits
        gkf = GroupKFold(n_splits=min(5, len(np.unique(pos_clean))))
        for fold, (tr, te) in enumerate(gkf.split(Xd, yd_clean, groups=pos_clean), start=1):
            Xtr, Xte = transform(Xd, tr, te)
            ytr, yte = yd_clean[tr], yd_clean[te]
            split_id = f"pos_holdout_fold{fold}"

            print(f"  TabPFN {split_id} ...", flush=True)
            pred = fit_predict_tabpfn(Xtr, ytr, Xte)
            sp = safe_spearman(yte, pred)
            rows.append({
                "embedding": "prott5",
                "target": target,
                "split": "pos_holdout",
                "split_id": split_id,
                "model": "tabpfn",
                "model_seed": 0,
                "n_train": len(ytr),
                "n_test": len(yte),
                "spearman": sp,
                "top5_precision": topk_precision(yte, pred),
                "top5_enrich_diff": topk_enrich_diff(yte, pred),
            })
            print(f"    spearman={sp:.4f}", flush=True)

    # Append to CSV
    new_df = pd.DataFrame(rows)
    new_df.to_csv(RAW_CSV, mode="a", header=False, index=False)
    new_count = old_count + len(new_df)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Appended {len(new_df)} new rows to prott5_runs_raw.csv")
    print(f"Old row count: {old_count}")
    print(f"New row count: {new_count}")
    print(f"Runtime: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
