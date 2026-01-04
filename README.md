# Rubisco AI Chemistry public deposition package

This folder consolidates code, processed data, and precomputed outputs that support the
MDPI AI Chemistry manuscript on protein language model embeddings for Rubisco variant
prediction and active learning.

## Licensing

- Code: MIT License. See `LICENSE`.
- Original, code generated figures and tables: CC BY 4.0. See `LICENSE_DATA`.
- Third party resources: see `THIRD_PARTY_NOTICES.md`.

## Folder map

- `code/AI_CHEM_code/`
  - Primary modeling and benchmarking scripts for the main manuscript results.
  - Includes processed input tables and precomputed ESM2 embeddings used for the
    XGBoost and TabPFN comparisons.

- `results/`
  - `results_pubready_xgb_tabpfn/` Precomputed evaluation outputs for the main benchmarks.
  - `results_active_learning_pubready/` Precomputed active learning simulation outputs.

- `supplementary/doubling_time_package_v2/`
  - Supplementary analysis code and outputs.
  - Includes cyanobacterial doubling time compilation outputs and embedding based regressions.
  - Includes Flamholz kinetics benchmark analysis outputs.
  - Full text PDFs are not included. Where needed, scripts can download open access PDFs.

- `manuscript/`
  - Current manuscript draft and figure assets.
  - Excel sheets requested for re rendering supplemental figures and for prediction tables.

- `env/`
  - Environment captures and minimal requirements files.

## Quick start

### 1. Main benchmark reproduction

From the package root:

```bash
cd code/AI_CHEM_code
python rubisco_pubready_xgb_tabpfn.py --mode run
python rubisco_pubready_xgb_tabpfn.py --mode summarize
```

Outputs are written to a new `results_pubready_xgb_tabpfn/` folder in the working directory by default.

Notes
- The full benchmark script imports `tabpfn`. If TabPFN is not installed or is not permitted under your intended use case, reproduce the XGBoost only outputs using the XGBoost only paths in the repository or rely on the precomputed results in `../../results/results_pubready_xgb_tabpfn/`.
- ESM2 embeddings are provided as `.npy` files for reproducibility. If you prefer to regenerate embeddings, use the embedding utilities in `supplementary/doubling_time_package_v2/`.

### 2. Active learning simulation

```bash
cd code/AI_CHEM_code
python active_learning_pubready.py
```

Outputs are written to `results_active_learning_pubready/` in the working directory by default.

### 3. Supplementary analyses

From `supplementary/doubling_time_package_v2`:

```bash
python analyze_cyano_doubling_times.py
```

To rebuild the cyanobacterial doubling time compilation from Yu et al. 2015, use:

```bash
python build_cyano_doubling_time_dataset.py \
  --download_yu2015 \
  --out_dir doubling_time_outputs \
  --skip_lab
```

The `--skip_lab` option omits any local lab specific inputs that are not redistributed in this public package.

To reproduce the Flamholz embedding based kinetics benchmarks:

```bash
python analyze_flamholz_embeddings_vs_kinetics.py \
  --flamholz_csv inputs/flamholz_dataset_S1.csv \
  --meta_csv manuscript_assets/flamholz/flamholz_rubisco_wt_meta.csv \
  --emb_npy manuscript_assets/flamholz/flamholz_rubisco_esm2_t12_mean.npy \
  --out_dir manuscript_assets/flamholz/embedding_vs_kinetics \
  --targets vC KC S KO vO KRuBP \
  --k 5 --pca_dim 16 --ridge_alpha 10.0 --log10 --wt_only --aggregate median
```

## Reproducibility notes

- Random seeds are set inside most benchmark scripts.
- ESM2 models download weights on first use. Ensure you have network access or pre stage weights into your Torch cache.
- Some scripts depend on `torch` and `fair-esm`.

## Suggested citation

Please use `CITATION.cff` and the manuscript DOI when available.

