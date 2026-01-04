import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from xgboost import XGBRegressor

SEED = 0
rng = np.random.default_rng(SEED)

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

def fit_xgb_with_val(Xtr, ytr, Xval, yval, Xte, pca_dim=128, n_jobs=16):
    scaler = StandardScaler()
    pca = PCA(n_components=pca_dim, random_state=SEED, svd_solver="randomized")

    Xtr_t = pca.fit_transform(scaler.fit_transform(Xtr))
    Xval_t = pca.transform(scaler.transform(Xval))
    Xte_t  = pca.transform(scaler.transform(Xte))

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=8000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=n_jobs,
        random_state=SEED,
        tree_method="hist",
        early_stopping_rounds=200,
    )
    model.fit(Xtr_t, ytr, eval_set=[(Xval_t, yval)], verbose=False)
    return model.predict(Xte_t)

def parse_pos_list(s):
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

def eval_block(name, yte, pred):
    p5, e5 = topk_precision_and_enrich(yte, pred, 0.05)
    return {
        "split": name,
        "n_test": int(len(yte)),
        "r2": float(r2_score(yte, pred)),
        "spearman": safe_spear(yte, pred),
        "precision_at_5pct": p5,
        "enrichment_at_5pct": e5,
    }

def main():
    os.makedirs("results_full_eval", exist_ok=True)

    # Embeddings
    full = np.load("esm2_t33_650m_full.npy", allow_pickle=True).item()
    X_all = full["emb"].astype(np.float32)
    ids_all = full["ids"].astype(str)

    # WT embeddings
    wt = np.load("esm2_wt_embeddings.npy", allow_pickle=True).item()
    wt_map = {str(i): wt["emb"][k].astype(np.float32) for k, i in enumerate(wt["ids"].astype(str))}
    wt_hoff = wt_map["HOFF|WT"]

    # Labels
    df = pd.read_csv("rubisco_datasets_merged.csv", low_memory=False)
    df = df.set_index("variant_id").loc[ids_all].reset_index()

    hoff_mask = (df["dataset_id"].values == "HOFF")
    hoff = df.loc[hoff_mask].copy()
    X = X_all[hoff_mask]

    # Basic filters
    if "has_stop" in hoff.columns:
        good = ~hoff["has_stop"].fillna(False).to_numpy()
        hoff = hoff.loc[good].copy()
        X = X[good]

    hoff["n_mut"] = pd.to_numeric(hoff["n_mut"], errors="coerce")
    y = pd.to_numeric(hoff["hoff_delta_O2_minus_N2"], errors="coerce").to_numpy()

    keep = np.isfinite(y) & np.isfinite(hoff["n_mut"].to_numpy())
    hoff = hoff.loc[keep].copy()
    X = X[keep]
    y = y[keep]
    nmut = hoff["n_mut"].astype(int).to_numpy()

    feats = {
        "mean": X,
        "delta": (X - wt_hoff[None, :]),
        "concat": np.hstack([X, (X - wt_hoff[None, :])]),
    }
    pca_dims = {"mean": 128, "delta": 128, "concat": 256}

    results = []

    # -------------------------
    # (1) Depth holdout split
    # -------------------------
    tr = (nmut <= 4)
    va = (nmut == 5)
    te = (nmut >= 6)

    # Fallback validation if n_mut==5 is sparse
    if va.sum() < 50:
        tr_idx = np.where(tr)[0]
        tr2, va2 = train_test_split(tr_idx, test_size=0.1, random_state=SEED)
        tr_mask = np.zeros(len(y), dtype=bool); tr_mask[tr2] = True
        va_mask = np.zeros(len(y), dtype=bool); va_mask[va2] = True
        tr, va = tr_mask, va_mask

    # Ensure test is non-empty; if not, loosen threshold
    if te.sum() < 50:
        te = (nmut >= 5)

    for fname, F in feats.items():
        pred = fit_xgb_with_val(F[tr], y[tr], F[va], y[va], F[te], pca_dim=pca_dims[fname], n_jobs=16)
        out = eval_block("depth_holdout_test", y[te], pred)
        out["features"] = fname
        out["test_nmut_min"] = int(np.min(nmut[te]))
        results.append(out)

    # -------------------------
    # (2) Position holdout split
    # -------------------------
    pos_lists = hoff["mut_positions"].apply(parse_pos_list).tolist()
    all_pos = sorted({p for lst in pos_lists for p in lst})
    if len(all_pos) == 0:
        raise RuntimeError("No mutation positions parsed from hoff.mut_positions")

    k = max(1, int(round(0.10 * len(all_pos))))
    hold = set(rng.choice(all_pos, size=k, replace=False).tolist())

    is_test = np.array([any(p in hold for p in lst) for lst in pos_lists], dtype=bool)
    is_train = ~is_test

    tr_idx = np.where(is_train)[0]
    tr2, va2 = train_test_split(tr_idx, test_size=0.1, random_state=SEED)
    tr_mask = np.zeros(len(y), dtype=bool); tr_mask[tr2] = True
    va_mask = np.zeros(len(y), dtype=bool); va_mask[va2] = True

    for fname, F in feats.items():
        pred = fit_xgb_with_val(F[tr_mask], y[tr_mask], F[va_mask], y[va_mask], F[is_test], pca_dim=pca_dims[fname], n_jobs=16)
        out = eval_block("pos_holdout_test", y[is_test], pred)
        out["features"] = fname
        out["holdout_positions"] = int(len(hold))
        out["test_frac"] = float(is_test.mean())
        results.append(out)

    res = pd.DataFrame(results)
    out_csv = "results_full_eval/hoffmann_oxygen_eval.csv"
    res.to_csv(out_csv, index=False)

    print("Saved", out_csv)
    print(res.to_string(index=False))

if __name__ == "__main__":
    main()
