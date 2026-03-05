#!/usr/bin/env python3
"""Run SVR + MLP baselines for the two DMS targets missing from svr_mlp_baselines.csv.

Targets: dms_KmCO2_logfit, dms_VmaxRatio_logfit
Splits:  within_position (seeds 0/1/2), pos_holdout (5-fold GroupKFold)
Embedding: ESM2, PCA=128

Uses identical preprocessing, split construction, and model logic as
run_svr_mlp_baselines.py.  Appends new rows to the existing CSV.
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

REPO = "/mmfs1/scratch/jacks.local/jyoung67391/Rubisco_Protein_Language_Model_Active_Learning"
EMB_NPY = "/mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed/esm2_t33_650m_full.npy"
LABELS_CSV = os.path.join(REPO, "code/AI_CHEM_code/rubisco_datasets_merged.csv")
OUT_CSV = os.path.join(REPO, "results/svr_mlp_baselines.csv")

PCA_DIM = 128
MISSING_TARGETS = ["dms_KmCO2_logfit", "dms_VmaxRatio_logfit"]


def safe_spearman(y, p):
    c = spearmanr(y, p).correlation
    return float(c) if np.isfinite(c) else np.nan


def topk_precision(y_true, y_pred, frac=0.05):
    k = max(1, int(np.ceil(frac * len(y_true))))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    return len(true_top & pred_top) / k


def transform(X, tr, te):
    scaler = StandardScaler()
    pca = PCA(n_components=PCA_DIM, random_state=0, svd_solver="randomized")
    Xtr = pca.fit_transform(scaler.fit_transform(X[tr]))
    Xte = pca.transform(scaler.transform(X[te]))
    return Xtr.astype(np.float32), Xte.astype(np.float32)


def fit_predict_svr(Xtr, ytr, Xte, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ytr))
    tr_idx, va_idx = train_test_split(
        idx, test_size=0.10, random_state=int(rng.integers(1e9))
    )
    best_score, best_params = -np.inf, None
    for C in [0.1, 1.0, 10.0, 100.0]:
        for gamma in ["scale", "auto"]:
            m = SVR(kernel="rbf", C=C, gamma=gamma)
            m.fit(Xtr[tr_idx], ytr[tr_idx])
            p = m.predict(Xtr[va_idx])
            s = safe_spearman(ytr[va_idx], p)
            if s > best_score:
                best_score = s
                best_params = {"C": C, "gamma": gamma}
    final = SVR(kernel="rbf", **best_params)
    final.fit(Xtr, ytr)
    return final.predict(Xte), best_params


def fit_predict_mlp(Xtr, ytr, Xte, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ytr))
    tr_idx, va_idx = train_test_split(
        idx, test_size=0.10, random_state=int(rng.integers(1e9))
    )
    best_score, best_alpha = -np.inf, None
    for alpha in [0.0001, 0.001, 0.01]:
        m = MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu", solver="adam",
            alpha=alpha, max_iter=400,
            early_stopping=True, random_state=seed,
        )
        m.fit(Xtr[tr_idx], ytr[tr_idx])
        p = m.predict(Xtr[va_idx])
        s = safe_spearman(ytr[va_idx], p)
        if s > best_score:
            best_score = s
            best_alpha = alpha
    final = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu", solver="adam",
        alpha=best_alpha, max_iter=600,
        early_stopping=True, random_state=seed,
    )
    final.fit(Xtr, ytr)
    return final.predict(Xte), {"alpha": best_alpha}


def main():
    t0 = time.time()

    print("Loading ESM2 embeddings …", flush=True)
    emb = np.load(EMB_NPY, allow_pickle=True).item()
    ids = emb["ids"].astype(str)
    Xall = emb["emb"].astype(np.float32)
    print(f"  Shape: {Xall.shape}", flush=True)

    print("Loading labels …", flush=True)
    df = pd.read_csv(LABELS_CSV, low_memory=False).set_index("variant_id").loc[ids].reset_index()

    dms_mask = df["dataset_id"].eq("DMS").to_numpy()
    dms = df.loc[dms_mask].copy()
    Xdms = Xall[dms_mask]
    pos_all = pd.to_numeric(dms["position_external"], errors="coerce").to_numpy()

    rows = []

    for target in MISSING_TARGETS:
        yd = pd.to_numeric(dms[target], errors="coerce").to_numpy(dtype=np.float32)
        keep = np.isfinite(yd) & np.isfinite(pos_all)
        Xd, yd_clean, pos = Xdms[keep], yd[keep], pos_all[keep].astype(int)
        print(f"\n=== Target: {target} (N={len(yd_clean)}) ===", flush=True)

        # --- Within-position, seeds 0/1/2 ---
        for seed in [0, 1, 2]:
            tr, te = train_test_split(
                np.arange(len(yd_clean)), test_size=0.2,
                random_state=seed, stratify=pos,
            )
            Xtr, Xte = transform(Xd, tr, te)
            ytr, yte = yd_clean[tr], yd_clean[te]

            print(f"  within-position seed={seed} (train={len(ytr)}, test={len(yte)})", flush=True)

            print(f"    SVR …", flush=True)
            pred, hp = fit_predict_svr(Xtr, ytr, Xte, seed=seed)
            sp = safe_spearman(yte, pred)
            tp = topk_precision(yte, pred)
            rows.append({
                "model": "svr", "target": target, "split": "within_position",
                "seed_or_fold": seed, "spearman": sp, "top5_precision": tp,
                "best_C": hp["C"], "best_gamma": hp["gamma"],
                "n_train": len(ytr), "n_test": len(yte),
            })
            print(f"      spearman={sp:.4f}", flush=True)

            print(f"    MLP …", flush=True)
            pred, hp = fit_predict_mlp(Xtr, ytr, Xte, seed=seed)
            sp = safe_spearman(yte, pred)
            tp = topk_precision(yte, pred)
            rows.append({
                "model": "mlp", "target": target, "split": "within_position",
                "seed_or_fold": seed, "spearman": sp, "top5_precision": tp,
                "best_alpha": hp["alpha"],
                "n_train": len(ytr), "n_test": len(yte),
            })
            print(f"      spearman={sp:.4f}", flush=True)

        # --- Position-holdout, 5 folds ---
        gkf = GroupKFold(n_splits=min(5, len(np.unique(pos))))
        for fold, (tr, te) in enumerate(gkf.split(Xd, yd_clean, groups=pos), start=1):
            Xtr, Xte = transform(Xd, tr, te)
            ytr, yte = yd_clean[tr], yd_clean[te]
            fold_seed = 42 + fold

            print(f"  pos-holdout fold={fold} (train={len(ytr)}, test={len(yte)})", flush=True)

            print(f"    SVR …", flush=True)
            pred, hp = fit_predict_svr(Xtr, ytr, Xte, seed=fold_seed)
            sp = safe_spearman(yte, pred)
            tp = topk_precision(yte, pred)
            rows.append({
                "model": "svr", "target": target, "split": "pos_holdout",
                "seed_or_fold": fold, "spearman": sp, "top5_precision": tp,
                "best_C": hp["C"], "best_gamma": hp["gamma"],
                "n_train": len(ytr), "n_test": len(yte),
            })
            print(f"      spearman={sp:.4f}", flush=True)

            print(f"    MLP …", flush=True)
            pred, hp = fit_predict_mlp(Xtr, ytr, Xte, seed=fold_seed)
            sp = safe_spearman(yte, pred)
            tp = topk_precision(yte, pred)
            rows.append({
                "model": "mlp", "target": target, "split": "pos_holdout",
                "seed_or_fold": fold, "spearman": sp, "top5_precision": tp,
                "best_alpha": hp["alpha"],
                "n_train": len(ytr), "n_test": len(yte),
            })
            print(f"      spearman={sp:.4f}", flush=True)

    # Append to existing CSV
    new_df = pd.DataFrame(rows)
    existing = pd.read_csv(OUT_CSV)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)

    elapsed = time.time() - t0
    print(f"\nAppended {len(new_df)} new rows to {OUT_CSV} "
          f"(total now: {len(combined)})", flush=True)
    print(f"Runtime: {elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
