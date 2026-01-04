#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

import xgboost as xgb
from tabpfn import TabPFNRegressor


def safe_spearman(y, p):
    c = spearmanr(y, p).correlation
    return float(c) if np.isfinite(c) else np.nan

def fit_predict_xgb(
    Xtr, ytr, Xte,
    nthread=16, seed=0,
    max_depth=6, reg_lambda=10.0,
    eta=0.03, subsample=0.85, colsample=0.85,
    num_round=8000, early_stop=200, val_frac=0.10
):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ytr))
    tr_idx, va_idx = train_test_split(idx, test_size=val_frac, random_state=int(rng.integers(1e9)))

    dtr = xgb.DMatrix(Xtr[tr_idx], label=ytr[tr_idx])
    dva = xgb.DMatrix(Xtr[va_idx], label=ytr[va_idx])
    dte = xgb.DMatrix(Xte)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": float(eta),
        "max_depth": int(max_depth),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample),
        "lambda": float(reg_lambda),
        "tree_method": "hist",
        "seed": int(seed),
        "nthread": int(nthread),
    }

    bst = xgb.train(
        params,
        dtr,
        num_boost_round=int(num_round),
        evals=[(dva, "val")],
        early_stopping_rounds=int(early_stop),
        verbose_eval=False,
    )
    bi = bst.best_iteration
    if bi is None:
        pred = bst.predict(dte)
    else:
        pred = bst.predict(dte, iteration_range=(0, bi + 1))
    return pred.astype(np.float32)

def make_tabpfn(device: str, ignore_limits: bool):
    try:
        return TabPFNRegressor(device=device, ignore_pretraining_limits=ignore_limits)
    except TypeError:
        return TabPFNRegressor(device=device)

def fit_predict_tabpfn(
    Xtr, ytr, Xte,
    device="cuda",
    ignore_limits=True,
    train_cap=5000,
    seed=0
):
    n = len(ytr)
    idx = np.arange(n)
    if train_cap and train_cap > 0 and n > train_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=train_cap, replace=False)
    reg = make_tabpfn(device, ignore_limits)
    reg.fit(Xtr[idx], ytr[idx])
    pred = reg.predict(Xte).astype(np.float32)
    return pred

def transform_train_test(X, tr, te, pca_dim=128):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X[tr])
    Xte_s = scaler.transform(X[te])

    pca = PCA(n_components=int(pca_dim), random_state=0, svd_solver="randomized")
    Xtr_p = pca.fit_transform(Xtr_s).astype(np.float32)
    Xte_p = pca.transform(Xte_s).astype(np.float32)
    return Xtr_p, Xte_p

def eval_metrics(y_true, pred):
    return {
        "spearman": safe_spearman(y_true, pred),
        "r2": float(r2_score(y_true, pred)) if len(y_true) >= 2 else np.nan,
        "mse": float(mean_squared_error(y_true, pred)),
    }

def run_dms(df, X, target, pca_dim, xgb_params, tabpfn_device, tabpfn_cap, tabpfn_ignore):
    out_rows = []

    dms = df[df["dataset_id"] == "DMS"].copy()
    y = pd.to_numeric(dms[target], errors="coerce").to_numpy(dtype=np.float32)
    pos = pd.to_numeric(dms["position_external"], errors="coerce").to_numpy()

    keep = np.isfinite(y) & np.isfinite(pos)
    dms = dms.loc[keep].copy()
    y = y[keep]
    pos = pos[keep].astype(int)
    Xd = X[df["dataset_id"].values == "DMS"][keep]

    # Within-position split (stratify by pos)
    tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0, stratify=pos)
    Xtr_p, Xte_p = transform_train_test(Xd, tr_idx, te_idx, pca_dim=pca_dim)

    pred_xgb = fit_predict_xgb(Xtr_p, y[tr_idx], Xte_p, **xgb_params)
    pred_tab = fit_predict_tabpfn(Xtr_p, y[tr_idx], Xte_p, device=tabpfn_device,
                                  ignore_limits=tabpfn_ignore, train_cap=tabpfn_cap, seed=0)

    mx = eval_metrics(y[te_idx], pred_xgb)
    mt = eval_metrics(y[te_idx], pred_tab)
    out_rows.append({"task":"DMS", "target":target, "split":"within_position", "model":"xgb", **mx,
                     "n_train":len(tr_idx), "n_test":len(te_idx)})
    out_rows.append({"task":"DMS", "target":target, "split":"within_position", "model":"tabpfn", **mt,
                     "n_train":len(tr_idx), "n_test":len(te_idx), "tabpfn_train_cap": tabpfn_cap})

    # Position-holdout CV (GroupKFold by pos)
    uniq_pos = np.unique(pos)
    n_splits = min(5, len(uniq_pos))
    gkf = GroupKFold(n_splits=n_splits)

    fold = 0
    for tr, te in gkf.split(np.zeros(len(y)), y, groups=pos):
        fold += 1
        Xtr_p, Xte_p = transform_train_test(Xd, tr, te, pca_dim=pca_dim)

        pred_xgb = fit_predict_xgb(Xtr_p, y[tr], Xte_p, **xgb_params)
        pred_tab = fit_predict_tabpfn(Xtr_p, y[tr], Xte_p, device=tabpfn_device,
                                      ignore_limits=tabpfn_ignore, train_cap=tabpfn_cap, seed=fold)

        mx = eval_metrics(y[te], pred_xgb)
        mt = eval_metrics(y[te], pred_tab)
        out_rows.append({"task":"DMS", "target":target, "split":f"pos_holdout_fold{fold}", "model":"xgb", **mx,
                         "n_train":len(tr), "n_test":len(te)})
        out_rows.append({"task":"DMS", "target":target, "split":f"pos_holdout_fold{fold}", "model":"tabpfn", **mt,
                         "n_train":len(tr), "n_test":len(te), "tabpfn_train_cap": tabpfn_cap})

    return pd.DataFrame(out_rows)


def run_hoff(df, X, target, pca_dim, xgb_params, tabpfn_device, tabpfn_cap, tabpfn_ignore):
    out_rows = []

    hoff_mask = (df["dataset_id"].values == "HOFF")
    hoff = df.loc[hoff_mask].copy()
    Xh = X[hoff_mask]

    # Apply has_stop filter to BOTH hoff and Xh (keeps them aligned)
    if "has_stop" in hoff.columns:
        good = ~hoff["has_stop"].fillna(False).astype(bool).to_numpy()
        hoff = hoff.loc[good].copy()
        Xh = Xh[good]

    y = pd.to_numeric(hoff[target], errors="coerce").to_numpy(dtype=np.float32)
    nmut = pd.to_numeric(hoff["n_mut"], errors="coerce").to_numpy()

    keep = np.isfinite(y) & np.isfinite(nmut)
    hoff = hoff.loc[keep].copy()
    Xh = Xh[keep]
    y = y[keep]
    nmut = nmut[keep].astype(int)

    # Depth holdout: train <=4, val ==5 (fallback), test >=6 (fallback >=5)
    tr_idx = np.where(nmut <= 4)[0]
    va_idx = np.where(nmut == 5)[0]
    te_idx = np.where(nmut >= 6)[0]

    if len(va_idx) < 50:
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.1, random_state=0)

    if len(te_idx) < 50:
        te_idx = np.where(nmut >= 5)[0]

    # Train-only transform (fit scaler+PCA on train, apply to test)
    Xtr_p, Xte_p = transform_train_test(Xh, tr_idx, te_idx, pca_dim=pca_dim)

    pred_xgb = fit_predict_xgb(Xtr_p, y[tr_idx], Xte_p, **xgb_params)
    pred_tab = fit_predict_tabpfn(
        Xtr_p, y[tr_idx], Xte_p,
        device=tabpfn_device,
        ignore_limits=tabpfn_ignore,
        train_cap=tabpfn_cap,
        seed=0
    )

    mx = eval_metrics(y[te_idx], pred_xgb)
    mt = eval_metrics(y[te_idx], pred_tab)

    out_rows.append({
        "task": "HOFF",
        "target": target,
        "split": "depth_holdout_test",
        "model": "xgb",
        **mx,
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
    })
    out_rows.append({
        "task": "HOFF",
        "target": target,
        "split": "depth_holdout_test",
        "model": "tabpfn",
        **mt,
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "tabpfn_train_cap": tabpfn_cap,
    })

    return pd.DataFrame(out_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npy", default="esm2_t33_650m_full.npy")
    ap.add_argument("--labels_csv", default="rubisco_datasets_merged.csv")
    ap.add_argument("--out_dir", default="results_rubisco_tabpfn_vs_xgb")

    ap.add_argument("--dms_target", default="dms_enrichment_mean")
    ap.add_argument("--run_dms", action="store_true")
    ap.add_argument("--run_hoff", action="store_true")
    ap.add_argument("--hoff_target", default="hoff_delta_O2_minus_N2")

    ap.add_argument("--pca_dim", type=int, default=128)

    ap.add_argument("--xgb_max_depth", type=int, default=6)
    ap.add_argument("--xgb_reg_lambda", type=float, default=10.0)
    ap.add_argument("--xgb_num_round", type=int, default=8000)
    ap.add_argument("--xgb_early_stop", type=int, default=200)
    ap.add_argument("--xgb_nthread", type=int, default=16)

    ap.add_argument("--tabpfn_device", default="cuda", help="cuda|cpu|auto")
    ap.add_argument("--tabpfn_train_cap", type=int, default=5000)
    ap.add_argument("--tabpfn_ignore_limits", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    embd = np.load(args.emb_npy, allow_pickle=True).item()
    ids = embd["ids"].astype(str)
    X = embd["emb"].astype(np.float32)

    df = pd.read_csv(args.labels_csv, low_memory=False)
    df = df.set_index("variant_id").loc[ids].reset_index()

    xgb_params = dict(
        nthread=args.xgb_nthread,
        seed=0,
        max_depth=args.xgb_max_depth,
        reg_lambda=args.xgb_reg_lambda,
        num_round=args.xgb_num_round,
        early_stop=args.xgb_early_stop,
    )

    results = []

    if not args.run_dms and not args.run_hoff:
        # default both
        args.run_dms = True
        args.run_hoff = True

    if args.run_dms:
        dms_res = run_dms(df, X, args.dms_target, args.pca_dim, xgb_params,
                          args.tabpfn_device, args.tabpfn_train_cap, args.tabpfn_ignore_limits)
        dms_out = os.path.join(args.out_dir, "dms_results.csv")
        dms_res.to_csv(dms_out, index=False)
        print("Wrote:", dms_out)
        results.append(dms_res)

    if args.run_hoff:
        hoff_res = run_hoff(df, X, args.hoff_target, args.pca_dim, xgb_params,
                            args.tabpfn_device, args.tabpfn_train_cap, args.tabpfn_ignore_limits)
        hoff_out = os.path.join(args.out_dir, "hoff_results.csv")
        hoff_res.to_csv(hoff_out, index=False)
        print("Wrote:", hoff_out)
        results.append(hoff_res)

    all_res = pd.concat(results, ignore_index=True)
    # print summaries
    print("\n=== Summary (mean spearman by task/split/model) ===")
    summ = (all_res.groupby(["task","target","split","model"])["spearman"]
                 .mean()
                 .reset_index()
                 .sort_values(["task","split","model"]))
    print(summ.to_string(index=False))

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

if __name__ == "__main__":
    main()
