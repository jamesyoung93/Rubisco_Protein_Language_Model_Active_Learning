#!/usr/bin/env python3
"""Task C: SVR and MLP baselines.

SVR: RBF kernel, tune C in [0.1, 1, 10, 100], gamma in ['scale', 'auto']
     using same 10% inner validation split as XGB.
MLP: hidden_layers=(256,128), relu, adam,
     tune alpha in [0.0001, 0.001, 0.01] using same 10% inner val.

Splits:
  - DMS targets (configurable), within-position (seeds 0/1/2)
  - DMS targets (configurable), pos_holdout (5 folds)
  - hoff_delta_O2_minus_N2, depth-holdout

ESM2 embeddings, PCA=128, canonical preprocessing.
"""
import argparse
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

# ---------------------------------------------------------------------------
REPO = "/mmfs1/scratch/jacks.local/jyoung67391/Rubisco_Protein_Language_Model_Active_Learning"
EMB_NPY = "/mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed/esm2_t33_650m_full.npy"
LABELS_CSV = os.path.join(REPO, "code/AI_CHEM_code/rubisco_datasets_merged.csv")
V1_SUMMARY = os.path.join(REPO, "_orginal_v1/results/results_pubready_xgb_tabpfn/summary_pub.csv")
OUT_CSV = os.path.join(REPO, "results/svr_mlp_baselines.csv")
OUT_TXT = os.path.join(REPO, "results/svr_mlp_summary.txt")

PCA_DIM = 128
ALL_DMS_TARGETS = ["dms_enrichment_mean", "dms_KmCO2_logfit", "dms_VmaxRatio_logfit"]


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


def transform(X, tr, te):
    scaler = StandardScaler()
    pca = PCA(n_components=PCA_DIM, random_state=0, svd_solver="randomized")
    Xtr = pca.fit_transform(scaler.fit_transform(X[tr]))
    Xte = pca.transform(scaler.transform(X[te]))
    return Xtr.astype(np.float32), Xte.astype(np.float32)


# ---------------------------------------------------------------------------
# SVR: RBF kernel, tune C × gamma using 10% inner val (same as XGB)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# MLP: (256,128), relu, adam, tune alpha using 10% inner val
# ---------------------------------------------------------------------------
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
            activation="relu",
            solver="adam",
            alpha=alpha,
            max_iter=400,
            early_stopping=True,
            random_state=seed,
        )
        m.fit(Xtr[tr_idx], ytr[tr_idx])
        p = m.predict(Xtr[va_idx])
        s = safe_spearman(ytr[va_idx], p)
        if s > best_score:
            best_score = s
            best_alpha = alpha

    final = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=best_alpha,
        max_iter=600,
        early_stopping=True,
        random_state=seed,
    )
    final.fit(Xtr, ytr)
    return final.predict(Xte), {"alpha": best_alpha}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SVR/MLP baselines")
    parser.add_argument("--targets", nargs="*", default=ALL_DMS_TARGETS,
                        help="DMS targets to evaluate")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing CSV instead of overwriting")
    parser.add_argument("--skip-hoff", action="store_true",
                        help="Skip HOFF depth-holdout")
    parser.add_argument("--skip-summary", action="store_true",
                        help="Skip summary text generation")
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    print("Loading ESM2 embeddings …", flush=True)
    emb = np.load(EMB_NPY, allow_pickle=True).item()
    ids = emb["ids"].astype(str)
    Xall = emb["emb"].astype(np.float32)
    print(f"  Shape: {Xall.shape}", flush=True)

    print("Loading labels …", flush=True)
    df = pd.read_csv(LABELS_CSV, low_memory=False).set_index("variant_id").loc[ids].reset_index()

    rows = []

    # =================================================================
    # DMS targets
    # =================================================================
    dms_mask = df["dataset_id"].eq("DMS").to_numpy()

    for target in args.targets:
        dms = df.loc[dms_mask].copy()
        Xdms = Xall[dms_mask]
        yd = pd.to_numeric(dms[target], errors="coerce").to_numpy(dtype=np.float32)
        pos = pd.to_numeric(dms["position_external"], errors="coerce").to_numpy()
        keep = np.isfinite(yd) & np.isfinite(pos)
        Xd, yd, pos = Xdms[keep], yd[keep], pos[keep].astype(int)
        print(f"DMS {target}: {len(yd)} variants, {len(np.unique(pos))} positions\n", flush=True)

        # --- Within-position, seeds 0/1/2 ---
        for seed in [0, 1, 2]:
            tr, te = train_test_split(
                np.arange(len(yd)), test_size=0.2,
                random_state=seed, stratify=pos,
            )
            Xtr, Xte = transform(Xd, tr, te)
            ytr, yte = yd[tr], yd[te]

            print(f"DMS {target} within-position seed={seed} (train={len(ytr)}, test={len(yte)})", flush=True)

            # SVR
            print(f"  SVR tuning …", flush=True)
            pred, hp = fit_predict_svr(Xtr, ytr, Xte, seed=seed)
            sp = safe_spearman(yte, pred)
            tp = topk_precision(yte, pred)
            rows.append({
                "model": "svr",
                "target": target,
                "split": "within_position",
                "seed_or_fold": seed,
                "spearman": sp,
                "top5_precision": tp,
                "best_C": hp["C"],
                "best_gamma": hp["gamma"],
                "n_train": len(ytr),
                "n_test": len(yte),
            })
            print(f"    spearman={sp:.4f}  C={hp['C']}  gamma={hp['gamma']}", flush=True)

            # MLP
            print(f"  MLP tuning …", flush=True)
            pred, hp = fit_predict_mlp(Xtr, ytr, Xte, seed=seed)
            sp = safe_spearman(yte, pred)
            tp = topk_precision(yte, pred)
            rows.append({
                "model": "mlp",
                "target": target,
                "split": "within_position",
                "seed_or_fold": seed,
                "spearman": sp,
                "top5_precision": tp,
                "best_alpha": hp["alpha"],
                "n_train": len(ytr),
                "n_test": len(yte),
            })
            print(f"    spearman={sp:.4f}  alpha={hp['alpha']}", flush=True)

        # --- Position-holdout, 5 folds ---
        gkf = GroupKFold(n_splits=min(5, len(np.unique(pos))))
        for fold, (tr, te) in enumerate(gkf.split(Xd, yd, groups=pos), start=1):
            Xtr, Xte = transform(Xd, tr, te)
            ytr, yte = yd[tr], yd[te]
            fold_seed = 42 + fold

            print(f"\nDMS {target} pos-holdout fold={fold} (train={len(ytr)}, test={len(yte)})", flush=True)

            # SVR
            print(f"  SVR tuning …", flush=True)
            pred, hp = fit_predict_svr(Xtr, ytr, Xte, seed=fold_seed)
            sp = safe_spearman(yte, pred)
            tp = topk_precision(yte, pred)
            rows.append({
                "model": "svr",
                "target": target,
                "split": "pos_holdout",
                "seed_or_fold": fold,
                "spearman": sp,
                "top5_precision": tp,
                "best_C": hp["C"],
                "best_gamma": hp["gamma"],
                "n_train": len(ytr),
                "n_test": len(yte),
            })
            print(f"    spearman={sp:.4f}  C={hp['C']}  gamma={hp['gamma']}", flush=True)

            # MLP
            print(f"  MLP tuning …", flush=True)
            pred, hp = fit_predict_mlp(Xtr, ytr, Xte, seed=fold_seed)
            sp = safe_spearman(yte, pred)
            tp = topk_precision(yte, pred)
            rows.append({
                "model": "mlp",
                "target": target,
                "split": "pos_holdout",
                "seed_or_fold": fold,
                "spearman": sp,
                "top5_precision": tp,
                "best_alpha": hp["alpha"],
                "n_train": len(ytr),
                "n_test": len(yte),
            })
            print(f"    spearman={sp:.4f}  alpha={hp['alpha']}", flush=True)

    # =================================================================
    # HOFF depth-holdout
    # =================================================================
    if not args.skip_hoff:
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
        Xtr, Xte = transform(Xh, tr, te)
        ytr, yte = yh[tr], yh[te]
        print(f"  N_train={len(ytr)} (n_mut<=4), N_test={len(yte)} (n_mut>=6)", flush=True)

        # SVR
        print(f"  SVR tuning …", flush=True)
        pred, hp = fit_predict_svr(Xtr, ytr, Xte, seed=42)
        sp = safe_spearman(yte, pred)
        tp = topk_precision(yte, pred)
        rows.append({
            "model": "svr",
            "target": "hoff_delta_O2_minus_N2",
            "split": "depth_holdout",
            "seed_or_fold": 0,
            "spearman": sp,
            "top5_precision": tp,
            "best_C": hp["C"],
            "best_gamma": hp["gamma"],
            "n_train": len(ytr),
            "n_test": len(yte),
        })
        print(f"    spearman={sp:.4f}  C={hp['C']}  gamma={hp['gamma']}", flush=True)

        # MLP
        print(f"  MLP tuning …", flush=True)
        pred, hp = fit_predict_mlp(Xtr, ytr, Xte, seed=42)
        sp = safe_spearman(yte, pred)
        tp = topk_precision(yte, pred)
        rows.append({
            "model": "mlp",
            "target": "hoff_delta_O2_minus_N2",
            "split": "depth_holdout",
            "seed_or_fold": 0,
            "spearman": sp,
            "top5_precision": tp,
            "best_alpha": hp["alpha"],
            "n_train": len(ytr),
            "n_test": len(yte),
        })
        print(f"    spearman={sp:.4f}  alpha={hp['alpha']}", flush=True)

    # =================================================================
    # Save raw results
    # =================================================================
    results_df = pd.DataFrame(rows)
    if args.append and os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(results_df)} rows)", flush=True)

    # =================================================================
    # Summary + comparison with v1 XGB/TabPFN
    # =================================================================
    if args.skip_summary:
        elapsed = time.time() - t0
        print(f"\nTotal runtime: {elapsed / 60:.1f} min")
        print("Done.", flush=True)
        return

    v1 = pd.read_csv(V1_SUMMARY)
    v1f = v1[v1["config_label"] == "fixed"].copy()

    lines = []
    lines.append("Task C: SVR and MLP Baselines — Comparison with XGB and TabPFN")
    lines.append("=" * 80)
    lines.append(f"SVR: RBF kernel, C in [0.1, 1, 10, 100], gamma in [scale, auto]")
    lines.append(f"MLP: (256,128) relu adam, alpha in [0.0001, 0.001, 0.01]")
    lines.append(f"Inner CV: 10% val split (same as XGB)")
    lines.append(f"PCA: {PCA_DIM}, ESM2 embeddings")
    lines.append("")

    # Build comparison table dynamically from results
    comparisons = []
    for t in results_df["target"].unique():
        if t.startswith("dms_"):
            comparisons.append((t, "within_position", f"DMS within-pos ({t})"))
            comparisons.append((t, "pos_holdout", f"DMS pos-holdout ({t})"))
        elif t == "hoff_delta_O2_minus_N2":
            comparisons.append((t, "depth_holdout", "HOFF depth-holdout"))

    for target, split, label in comparisons:
        lines.append(f"\n{label}")
        lines.append("-" * 80)
        lines.append(f"{'Model':<10} {'n':>4} {'Spearman':>22} {'Top-5% Precision':>22}")
        lines.append("-" * 80)

        # SVR/MLP from our results
        for model in ["svr", "mlp"]:
            sub = results_df[(results_df["model"] == model) &
                             (results_df["target"] == target) &
                             (results_df["split"] == split)]
            n = len(sub)
            if n > 1:
                sp_str = f"{sub['spearman'].mean():.4f} ± {sub['spearman'].std():.4f}"
                tp_str = f"{sub['top5_precision'].mean():.4f} ± {sub['top5_precision'].std():.4f}"
            elif n == 1:
                sp_str = f"{sub['spearman'].values[0]:.4f}"
                tp_str = f"{sub['top5_precision'].values[0]:.4f}"
            else:
                sp_str = "N/A"
                tp_str = "N/A"
            lines.append(f"{model.upper():<10} {n:>4} {sp_str:>22} {tp_str:>22}")

        # XGB and TabPFN from v1 summary
        for model in ["xgb", "tabpfn"]:
            if "hoff" in target:
                v1_match = v1f[
                    (v1f["target"] == target) &
                    (v1f["split"] == split) &
                    (v1f["model"] == model) &
                    (v1f["task_name"] == "HOFF_delta_direct")
                ]
            else:
                v1_match = v1f[
                    (v1f["target"] == target) &
                    (v1f["split"] == split) &
                    (v1f["model"] == model)
                ]
            if len(v1_match) > 0:
                r = v1_match.iloc[0]
                n = int(r["n_units"])
                sp_str = f"{r['spearman_mean']:.4f} ± {r['spearman_std']:.4f}"
                tp_str = f"{r['top5_precision_mean']:.4f}"
                lines.append(f"{model.upper():<10} {n:>4} {sp_str:>22} {tp_str:>22}")

    # Ranking summary
    lines.append("\n")
    lines.append("Model Ranking by Spearman (mean across splits)")
    lines.append("=" * 80)

    for target, split, label in comparisons:
        rankings = []
        # SVR/MLP
        for model in ["svr", "mlp"]:
            sub = results_df[(results_df["model"] == model) &
                             (results_df["target"] == target) &
                             (results_df["split"] == split)]
            if len(sub) > 0:
                rankings.append((model.upper(), sub["spearman"].mean()))
        # XGB/TabPFN from v1
        for model in ["xgb", "tabpfn"]:
            if "hoff" in target:
                v1_match = v1f[
                    (v1f["target"] == target) &
                    (v1f["split"] == split) &
                    (v1f["model"] == model) &
                    (v1f["task_name"] == "HOFF_delta_direct")
                ]
            else:
                v1_match = v1f[
                    (v1f["target"] == target) &
                    (v1f["split"] == split) &
                    (v1f["model"] == model)
                ]
            if len(v1_match) > 0:
                rankings.append((model.upper(), v1_match.iloc[0]["spearman_mean"]))

        rankings.sort(key=lambda x: -x[1])
        rank_str = " > ".join(f"{m} ({s:.4f})" for m, s in rankings)
        lines.append(f"  {label}: {rank_str}")

    elapsed = time.time() - t0
    lines.append(f"\nTotal runtime: {elapsed / 60:.1f} min")

    summary_text = "\n".join(lines)
    print("\n" + summary_text, flush=True)

    with open(OUT_TXT, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nWrote {OUT_TXT}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
