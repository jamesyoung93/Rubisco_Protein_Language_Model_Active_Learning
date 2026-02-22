#!/usr/bin/env python3
import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="results/supplementary_hyperparameters.csv")
    args = ap.parse_args()

    rows = [
        ["XGBoost", "max_depth", "6", "{4,6,8}"],
        ["XGBoost", "reg_lambda", "10.0", "{1,10,50}"],
        ["XGBoost", "eta", "0.03", "{0.01,0.03,0.1}"],
        ["XGBoost", "num_boost_round", "8000", "{1200,4000,8000}"],
        ["XGBoost", "early_stopping_rounds", "200", "{50,100,200}"],
        ["TabPFN", "train_cap", "0 (full), 5000, 2000", "{0,2000,5000}"],
        ["TabPFN", "ignore_pretraining_limits", "False", "{False,True}"],
        ["SVR", "kernel", "rbf/linear tuned", "{rbf,linear}"],
        ["SVR", "C", "tuned", "{0.1,1.0,10.0}"],
        ["SVR", "epsilon", "tuned", "{0.01,0.1,0.2}"],
        ["MLP", "hidden_layer_sizes", "tuned", "{(256,128),(512,256),(256,128,64)}"],
        ["MLP", "learning_rate_init", "tuned", "{1e-3,3e-4}"],
        ["MLP", "alpha", "tuned", "{1e-5,1e-4}"],
        ["All", "pca_dim", "64,128,256 (default report 128)", "{64,128,256}"],
        ["ActiveLearning", "bootstrap_ensemble_size", "see active_learning config", "repo-configured"],
    ]

    df = pd.DataFrame(rows, columns=["Model", "Parameter", "Value", "Search Range"])
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Wrote", args.out_csv)


if __name__ == "__main__":
    main()
