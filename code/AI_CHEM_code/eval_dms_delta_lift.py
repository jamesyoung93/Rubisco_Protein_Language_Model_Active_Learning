import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from xgboost import XGBRegressor

SEED = 0
np.random.seed(SEED)

def safe_spear(y, p):
    c = spearmanr(y, p).correlation
    return float(c) if np.isfinite(c) else np.nan

def per_pos_spear(y, p, g):
    vals=[]
    for pos in np.unique(g):
        idx = (g==pos)
        if idx.sum() >= 3:
            c = safe_spear(y[idx], p[idx])
            if np.isfinite(c): vals.append(c)
    return float(np.mean(vals)) if vals else np.nan

def best_sub_acc(y, p, g, topk=1):
    acc=[]
    for pos in np.unique(g):
        idx = np.where(g==pos)[0]
        if len(idx) < 2: continue
        true_best = idx[np.argmax(y[idx])]
        pred_order = idx[np.argsort(p[idx])]
        pred_top = pred_order[-topk:]
        acc.append(1.0 if true_best in pred_top else 0.0)
    return float(np.mean(acc)) if acc else np.nan

def topk_precision_and_enrich(y, p, frac):
    n=len(y)
    k=max(1, int(np.ceil(frac*n)))
    true_top=set(np.argsort(y)[-k:])
    pred_top=set(np.argsort(p)[-k:])
    precision=len(true_top & pred_top)/k
    enrich=float(np.mean(y[list(pred_top)])/(np.mean(y)+1e-12))
    return float(precision), float(enrich)

def fit_predict_pca_xgb(Xtr, ytr, Xte, pca_dim=128, inner_val_frac=0.1, n_jobs=16):
    Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=inner_val_frac, random_state=SEED)
    scaler = StandardScaler()
    pca = PCA(n_components=pca_dim, random_state=SEED, svd_solver="randomized")

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
        n_jobs=n_jobs,
        random_state=SEED,
        tree_method="hist",
        early_stopping_rounds=100,
    )
    xgb.fit(Xtr2_t, ytr2, eval_set=[(Xval_t, yval)], verbose=False)
    return xgb.predict(Xte_t)

def eval_once(split_name, feature_name, y, pred, groups):
    out = {
        "split": split_name,
        "features": feature_name,
        "r2": float(r2_score(y, pred)),
        "spearman_global": safe_spear(y, pred),
        "spearman_per_pos_mean": per_pos_spear(y, pred, groups),
        "best_sub_top1_acc": best_sub_acc(y, pred, groups, topk=1),
        "best_sub_top3_acc": best_sub_acc(y, pred, groups, topk=3),
    }
    p5, e5 = topk_precision_and_enrich(y, pred, 0.05)
    out["precision_at_5pct"] = p5
    out["enrichment_at_5pct"] = e5
    return out

def main():
    full = np.load("esm2_t33_650m_full.npy", allow_pickle=True).item()
    X_all = full["emb"]
    ids_all = full["ids"].astype(str)

    wt = np.load("esm2_wt_embeddings.npy", allow_pickle=True).item()
    wt_map = {str(i): wt["emb"][k] for k, i in enumerate(wt["ids"].astype(str))}
    wt_dms = wt_map["DMS|WT"].astype(np.float32)

    df = pd.read_csv("rubisco_datasets_merged.csv", low_memory=False).set_index("variant_id").loc[ids_all].reset_index()

    mask = (df["dataset_id"].values == "DMS")
    X = X_all[mask].astype(np.float32)
    dms = df.loc[mask].copy()

    y = pd.to_numeric(dms["dms_enrichment_mean"], errors="coerce").to_numpy()
    groups = pd.to_numeric(dms["position_external"], errors="coerce").to_numpy()

    keep = np.isfinite(y) & np.isfinite(groups)
    X, y, groups = X[keep], y[keep], groups[keep].astype(int)

    feats = {
        "mean": X,
        "delta": (X - wt_dms[None, :]),
        "concat": np.hstack([X, (X - wt_dms[None, :])]),
    }
    pca_dims = {"mean":128, "delta":128, "concat":256}

    results = []

    # within-position split
    Xtr_idx, Xte_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=groups)
    for fname, F in feats.items():
        pred = fit_predict_pca_xgb(F[Xtr_idx], y[Xtr_idx], F[Xte_idx], pca_dim=pca_dims[fname], n_jobs=16)
        results.append(eval_once("within_position", fname, y[Xte_idx], pred, groups[Xte_idx]))

    # position-holdout CV
    uniq = np.unique(groups)
    n_splits = min(5, len(uniq))
    gkf = GroupKFold(n_splits=n_splits)
    fold = 0
    for tr, te in gkf.split(np.zeros(len(y)), y, groups=groups):
        fold += 1
        for fname, F in feats.items():
            pred = fit_predict_pca_xgb(F[tr], y[tr], F[te], pca_dim=pca_dims[fname], n_jobs=16)
            results.append(eval_once(f"pos_holdout_cv_fold{fold}", fname, y[te], pred, groups[te]))

    res = pd.DataFrame(results)
    res.to_csv("results_full_eval/dms_delta_lift.csv", index=False)
    print("Saved results_full_eval/dms_delta_lift.csv")

    def summarize(prefix):
        sub = res[res["split"].str.startswith(prefix)]
        for fname in ["mean","delta","concat"]:
            m = sub[sub["features"] == fname]
            cols = ["r2","spearman_global","spearman_per_pos_mean","best_sub_top1_acc","best_sub_top3_acc","precision_at_5pct","enrichment_at_5pct"]
            s = m[cols].mean(numeric_only=True) if len(m) else None
            if s is not None:
                print(fname, "\n", s.to_string(), "\n")

    print("\n=== within-position (single split) ===")
    summarize("within_position")
    print("\n=== position-holdout CV (mean across folds) ===")
    summarize("pos_holdout_cv_fold")

if __name__ == "__main__":
    main()
