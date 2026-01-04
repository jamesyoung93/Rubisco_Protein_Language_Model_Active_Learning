# Package manifest

This file lists the primary analysis entry points and where their precomputed outputs are located.

## Main benchmarks (TabPFN and XGBoost)

- Code: `code/AI_CHEM_code/rubisco_pubready_xgb_tabpfn.py`
- Input data: `code/AI_CHEM_code/rubisco_datasets_merged.csv`
- Outputs (precomputed): `results/results_pubready_xgb_tabpfn/`

## Active learning simulations (XGBoost)

- Code: `code/AI_CHEM_code/active_learning_pubready.py`
- Input data: `code/AI_CHEM_code/rubisco_datasets_merged.csv`
- Outputs (precomputed): `results/results_active_learning_pubready/`

## Supplementary analyses

- Doubling time compilation and figures
  - Code: `supplementary/doubling_time_package_v2/build_cyano_doubling_time_dataset.py`
  - Code: `supplementary/doubling_time_package_v2/analyze_cyano_doubling_times.py`
  - Precomputed outputs: `supplementary/doubling_time_package_v2/doubling_time_outputs/`

- ESM2 embedding vs cyanobacterial growth proxy
  - Code: `supplementary/doubling_time_package_v2/analyze_rbcL_embedding_vs_doubling_time.py`
  - Precomputed outputs: `supplementary/doubling_time_package_v2/manuscript_assets/doubling_time/`

- Flamholz dataset: embedding vs kinetics prediction
  - Code: `supplementary/doubling_time_package_v2/analyze_flamholz_embeddings_vs_kinetics.py`
  - Code: `supplementary/doubling_time_package_v2/tune_flamholz_ridge_pca.py`
  - Precomputed outputs: `supplementary/doubling_time_package_v2/manuscript_assets/flamholz/`
