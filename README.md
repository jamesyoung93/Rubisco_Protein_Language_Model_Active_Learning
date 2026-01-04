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
- Small ESM2 embeddings are included for convenience, but the full `esm2_t33_650m_full.npy` cannot be stored on GitHub because it exceeds the 100 MB limit. Recreate it using the embedding steps below.

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

## Reproducing Results From Scratch (Fresh GitHub Clone)

The commands below reproduce the deposited results starting from a clean clone. They cover both generating the missing full ESM2 embedding file (`esm2_t33_650m_full.npy`, ~122 MB) and rerunning the main scripts. The embedding file is required by `eval_full_models.py`, `train_rank_hoffmann_o2_depth_tuned.py`, and `active_learning_sim.py` and should live in the working directory (`code/AI_CHEM_code/`).

### A. HPC / Slurm (recommended for embedding generation)

1. Clone and enter the repo:

   ```bash
   git clone https://github.com/jamesyoung93/Rubisco_Protein_Language_Model_Active_Learning.git
   cd Rubisco_Protein_Language_Model_Active_Learning
   ```

2. Create or update the Conda environment:

   ```bash
   conda env create -f env/conda_environment_rubisco_embed.yml
   # if the environment already exists
   conda env update -f env/conda_environment_rubisco_embed.yml --prune
   # activate (initialize Conda first if needed)
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate rubisco_embed
   ```

3. Set writable caches (adjust to any scratch path if desired):

   ```bash
   export PROJ="$(pwd)"
   export TORCH_HOME="$PROJ/.cache/torch"
   export HF_HOME="$PROJ/.cache/huggingface"
   mkdir -p "$TORCH_HOME" "$HF_HOME"
   ```

4. Generate the full ESM2 embeddings (Slurm batch):

   ```bash
   cd code/AI_CHEM_code
   mkdir -p logs
   sbatch run_embed_full.sbatch
   ```

   - Edit the `cd` line inside `run_embed_full.sbatch` to point to your cloned `code/AI_CHEM_code/` directory before submission.
   - The batch script runs:
     ```bash
     python embed_esm2.py \
       --in_csv rubisco_datasets_embed_input.csv \
       --id_col variant_id \
       --seq_col aa_sequence \
       --out_npy esm2_t33_650m_full.npy \
       --model facebook/esm2_t33_650M_UR50D \
       --batch_size 16
     ```

5. Sanity check the embedding file:

   ```bash
   ls -lh esm2_t33_650m_full.npy
   python - <<'PY'
   import numpy as np
   d = np.load("esm2_t33_650m_full.npy", allow_pickle=True).item()
   print("keys:", d.keys())
   print("ids:", len(d["ids"]))
   print("emb shape:", d["emb"].shape, "dtype:", d["emb"].dtype)
   PY
   ```

6. Reproduction commands used for deposition (from `code/AI_CHEM_code` after the embedding exists):

   ```bash
   python eval_full_models.py
   python train_rank_hoffmann_o2_depth_tuned.py
   python active_learning_sim.py \
     --emb_file esm2_t33_650m_full.npy \
     --workdir . \
     --dataset BOTH \
     --outdir results_active_learning_pubready_xgb \
     --strategies random,greedy,uncertainty,ucb,thompson \
     --n_reps 20 \
     --init_n 200 \
     --batch_size 48 \
     --max_rounds 25 \
     --ensemble_size 5 \
     --beta 1.0 \
     --dms_target dms_enrichment_mean \
     --dms_pca_dim 128 \
     --hoff_feature_mode mean \
     --hoff_pca_dim 64
   ```

   The quick-start entrypoint (`python rubisco_pubready_xgb_tabpfn.py --mode run` then `--mode summarize`) remains valid for the main benchmarks; the commands above are the exact ones used for the deposited evaluation and active learning results.

### B. Local workstation (CPU possible; GPU faster)

Follow the same steps as above for cloning, environment setup, and cache configuration. To generate embeddings without Slurm, run from `code/AI_CHEM_code`:

```bash
python embed_esm2.py \
  --in_csv rubisco_datasets_embed_input.csv \
  --id_col variant_id \
  --seq_col aa_sequence \
  --out_npy esm2_t33_650m_full.npy \
  --model facebook/esm2_t33_650M_UR50D \
  --batch_size 8
```

Then perform the sanity check and reproduction commands listed above.

### Common pitfalls / troubleshooting

- If you see “Permission denied” creating `/.cache/...`, your `TORCH_HOME`/`HF_HOME` is wrong—set them to a writable path (e.g., the repo-local `.cache` folder shown above).
- If you see `FileNotFoundError: esm2_t33_650m_full.npy`, you have not generated the embeddings or are running from the wrong working directory (`code/AI_CHEM_code`).
- Precomputed outputs are available under `results/` and can be used directly if you wish to skip compute-heavy steps.

## Reproducibility notes

- Random seeds are set inside most benchmark scripts.
- ESM2 models download weights on first use. Ensure you have network access or pre stage weights into your Torch cache.
- Some scripts depend on `torch` and `fair-esm`.

## Suggested citation

Please use `CITATION.cff` and the manuscript DOI when available.

