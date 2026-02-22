#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npy", required=True)
    ap.add_argument("--labels_csv", default="code/AI_CHEM_code/rubisco_datasets_merged.csv")
    ap.add_argument("--dataset", choices=["DMS", "HOFF"], default="DMS")
    ap.add_argument("--target", default="dms_enrichment_mean")
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--out_dir", default="results/results_shap")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    emb = np.load(args.emb_npy, allow_pickle=True).item()
    ids = emb["ids"].astype(str)
    Xall = emb["emb"].astype(np.float32)
    df = pd.read_csv(args.labels_csv).set_index("variant_id").loc[ids].reset_index()

    mask = df["dataset_id"].eq(args.dataset).to_numpy()
    d = df.loc[mask].copy()
    X = Xall[mask]
    y = pd.to_numeric(d[args.target], errors="coerce").to_numpy(dtype=np.float32)
    keep = np.isfinite(y)
    X, y = X[keep], y[keep]

    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    scaler = StandardScaler()
    pca = PCA(n_components=args.pca_dim, random_state=42)
    Xtr = pca.fit_transform(scaler.fit_transform(X[tr]))
    Xte = pca.transform(scaler.transform(X[te]))
    ytr = y[tr]

    dtr = xgb.DMatrix(Xtr, label=ytr)
    dte = xgb.DMatrix(Xte)
    booster = xgb.train({"objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 6, "eta": 0.03}, dtr, num_boost_round=1200)
    preds = booster.predict(dte)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(Xte)
    np.save(os.path.join(args.out_dir, "shap_values.npy"), shap_values)

    feature_names = [f"PC{i+1}" for i in range(Xte.shape[1])]
    shap.summary_plot(shap_values, Xte, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "shap_summary.png"), dpi=200)
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-5:][::-1]
    plt.figure(figsize=(6, 4))
    plt.bar([feature_names[i] for i in top_idx], mean_abs[top_idx])
    plt.ylabel("mean |SHAP|")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "shap_global_importance_top5.png"), dpi=200)
    plt.close()

    for i in top_idx[:3]:
        shap.dependence_plot(i, shap_values, Xte, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"shap_dependence_{feature_names[i]}.png"), dpi=200)
        plt.close()

    loadings = pd.DataFrame(pca.components_[top_idx].T, columns=[feature_names[i] for i in top_idx])
    loadings.to_csv(os.path.join(args.out_dir, "top_pc_loadings.csv"), index=False)

    pd.DataFrame({"y_true": y[te], "y_pred": preds}).to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)
    print("Wrote SHAP outputs to", args.out_dir)


if __name__ == "__main__":
    main()
