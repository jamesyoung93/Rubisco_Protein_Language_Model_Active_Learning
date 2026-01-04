import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from xgboost import XGBRegressor

SEED = 0
np.random.seed(SEED)

def safe_spearman(a, b):
    c = spearmanr(a, b).correlation
    return float(c) if np.isfinite(c) else np.nan

def per_group_spearman(y_true, y_pred, groups):
    corrs = []
    for g in np.unique(groups):
        idx = (groups == g)
        if idx.sum() >= 3:
            c = safe_spearman(y_true[idx], y_pred[idx])
            if np.isfinite(c):
                corrs.append(c)
    return float(np.mean(corrs)) if corrs else np.nan

def best_sub_acc(y_true, y_pred, groups, topk=1):
    acc = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        true_best = idx[np.argmax(y_true[idx])]
        pred_order = idx[np.argsort(y_pred[idx])]
        pred_top = pred_order[-topk:]
        acc.append(1.0 if true_best in pred_top else 0.0)
    return float(np.mean(acc)) if acc else np.nan

def topk_precision_and_enrichment(y_true, y_pred, frac):
    n = len(y_true)
    k = max(1, int(np.ceil(frac * n)))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    precision = len(true_top & pred_top) / k
    enrich = float(np.mean(y_true[list(pred_top)]) / (np.mean(y_true) + 1e-12))
    return float(precision), float(enrich)

def fit_predict_ridge(Xtr, ytr, Xte):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    model = Ridge(alpha=1.0)
    model.fit(Xtr_s, ytr)
    return model.predict(Xte_s)

def fit_predict_pca_xgb(Xtr, ytr, Xte, pca_dim=128, inner_val_frac=0.1, n_jobs=16):
    Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=inner_val_frac, random_state=SEED)

    scaler = StandardScaler()
    pca = PCA(n_components=pca_dim, random_state=SEED)

    Xtr2_t = pca.fit_transform(scaler.fit_transform(Xtr2))
    Xval_t = pca.transform(scaler.transform(Xval))
    Xte_t  = pca.transform(scaler.transform(Xte))

    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1,
        gamma=0.0,
        n_jobs=n_jobs,
        random_state=SEED,
        tree_method="hist",
        early_stopping_rounds=100,   # <-- move here (NOT in fit)
    )

    xgb.fit(
        Xtr2_t, ytr2,
        eval_set=[(Xval_t, yval)],
        verbose=False
    )

    return xgb.predict(Xte_t)

def evaluate(y_true, y_pred, groups):
    out = {}
    out["n"] = int(len(y_true))
    out["r2"] = float(r2_score(y_true, y_pred))
    out["spearman_global"] = safe_spearman(y_true, y_pred)
    out["spearman_per_pos_mean"] = per_group_spearman(y_true, y_pred, groups)
    out["best_sub_top1_acc"] = best_sub_acc(y_true, y_pred, groups, topk=1)
    out["best_sub_top3_acc"] = best_sub_acc(y_true, y_pred, groups, topk=3)
    p1, e1 = topk_precision_and_enrichment(y_true, y_pred, 0.01)
    p5, e5 = topk_precision_and_enrichment(y_true, y_pred, 0.05)
    out["precision_at_1pct"] = p1
    out["enrichment_at_1pct"] = e1
    out["precision_at_5pct"] = p5
    out["enrichment_at_5pct"] = e5
    return out

def main():
    emb_path = "esm2_t33_650m_full.npy"
    labels_path = "rubisco_datasets_merged.csv"
    out_csv = "results_full_eval/dms_enrichment_eval.csv"
    os.makedirs("results_full_eval", exist_ok=True)

    d = np.load(emb_path, allow_pickle=True).item()
    X_all = d["emb"]
    ids_all = d["ids"].astype(str)

    df = pd.read_csv(labels_path, low_memory=False).set_index("variant_id").loc[ids_all].reset_index()

    # DMS only
    mask = (df["dataset_id"].values == "DMS")
    X = X_all[mask]
    dms = df.loc[mask].copy()

    y = pd.to_numeric(dms["dms_enrichment_mean"], errors="coerce").to_numpy()
    groups = pd.to_numeric(dms["position_external"], errors="coerce").to_numpy()

    keep = np.isfinite(y) & np.isfinite(groups)
    X, y, groups = X[keep], y[keep], groups[keep].astype(int)

    # drop sparse positions
    vc = pd.Series(groups).value_counts()
    ok_groups = set(vc[vc >= 2].index.tolist())
    ok = np.array([g in ok_groups for g in groups], dtype=bool)
    X, y, groups = X[ok], y[ok], groups[ok]

    results = []

    # Within-position split
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        X, y, groups, test_size=0.2, random_state=SEED, stratify=groups
    )

    t0 = time.time()
    pr = fit_predict_ridge(Xtr, ytr, Xte)
    r = evaluate(yte, pr, gte)
    r.update({"split":"within_position", "model":"ridge", "seconds": time.time()-t0})
    results.append(r)

    t0 = time.time()
    px = fit_predict_pca_xgb(Xtr, ytr, Xte, pca_dim=128, inner_val_frac=0.1, n_jobs=16)
    r = evaluate(yte, px, gte)
    r.update({"split":"within_position", "model":"pca128_xgb", "seconds": time.time()-t0})
    results.append(r)

    # Position-holdout CV
    uniq = np.unique(groups)
    n_splits = min(5, len(uniq))
    gkf = GroupKFold(n_splits=n_splits)

    fold = 0
    for tr, te in gkf.split(X, y, groups=groups):
        fold += 1
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        gte = groups[te]

        t0 = time.time()
        pr = fit_predict_ridge(Xtr, ytr, Xte)
        r = evaluate(yte, pr, gte)
        r.update({"split":f"pos_holdout_cv_fold{fold}", "model":"ridge", "fold":fold, "seconds": time.time()-t0})
        results.append(r)

        t0 = time.time()
        px = fit_predict_pca_xgb(Xtr, ytr, Xte, pca_dim=128, inner_val_frac=0.1, n_jobs=16)
        r = evaluate(yte, px, gte)
        r.update({"split":f"pos_holdout_cv_fold{fold}", "model":"pca128_xgb", "fold":fold, "seconds": time.time()-t0})
        results.append(r)

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False)

    print("Saved:", out_csv)

    def show(split_prefix):
        sub = res_df[res_df["split"].str.startswith(split_prefix)]
        for model in ["ridge", "pca128_xgb"]:
            m = sub[sub["model"] == model]
            if len(m) == 0:
                continue
            cols = ["r2","spearman_global","spearman_per_pos_mean","best_sub_top1_acc","best_sub_top3_acc","precision_at_5pct","enrichment_at_5pct"]
            if "fold" in m.columns and m["fold"].notna().any():
                s = m[cols].mean(numeric_only=True)
                print(model, "\n", s.to_string(), "\n")
            else:
                s = m[cols].iloc[0]
                print(model, "\n", s.to_string(), "\n")

    print("\n=== within-position ===")
    show("within_position")

    print("\n=== position-holdout CV (mean across folds) ===")
    show("pos_holdout_cv_fold")

if __name__ == "__main__":
    main()
