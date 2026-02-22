# Revision utilities (reviewer requests)

- **ESM2 checkpoint used in this repo:** `facebook/esm2_t33_650M_UR50D` (default in `embed_esm2.py`).
- `extract_prot_t5_embeddings.py`: ProtT5 mean-pooled embedding extraction in `{"ids", "emb"}` `.npy` format.
- `benchmark_with_svr_mlp_balance.py`: Adds SVR + MLP baselines, quantile-bin balancing experiments, and bootstrap CIs.
- `shap_analysis.py`: SHAP TreeExplainer analysis for XGBoost with summary/dependence/importance outputs.
- `export_hyperparameter_table.py`: Writes supplementary hyperparameter table CSV.
- `build_validation_mutant_table.py`: Builds collaborator-ready hit/miss validation table from prediction files.

## Typical commands

```bash
python code/AI_CHEM_code/extract_prot_t5_embeddings.py \
  --in_csv code/AI_CHEM_code/rubisco_datasets_embed_input.csv \
  --id_col variant_id --seq_col sequence --out_npy results/prott5_embeddings.npy

python code/AI_CHEM_code/benchmark_with_svr_mlp_balance.py \
  --emb_npy results/prott5_embeddings.npy \
  --labels_csv code/AI_CHEM_code/rubisco_datasets_merged.csv \
  --out_dir results/results_model_extensions
```

> Active-learning reruns with ProtT5 are not wired into this script; use existing active-learning scripts with the ProtT5 `.npy` embedding path.
