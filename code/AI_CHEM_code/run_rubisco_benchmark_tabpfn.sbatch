#!/bin/bash
#SBATCH -J rub_bench
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd /mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed

source /mmfs1/cm/shared/apps_local/mamba/24.3/etc/profile.d/conda.sh
conda activate rubisco_embed
export PYTHONNOUSERSITE=1

export HF_HOME=/mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed/hf_cache
mkdir -p "$HF_HOME"

# Start with a manageable sweep (you can widen later)
python rubisco_benchmark_xgb_vs_tabpfn.py \
  --emb_npy esm2_t33_650m_full.npy \
  --labels_csv rubisco_datasets_merged.csv \
  --out_dir results_rubisco_benchmark_xgb_tabpfn_v1 \
  --datasets BOTH \
  --dms_target dms_enrichment_mean \
  --hoff_target hoff_delta_O2_minus_N2 \
  --pca_dims 64,128 \
  --split_seeds 0,1,2 \
  --model_seeds 0,1,2,3,4 \
  --tabpfn_caps 5000,0 \
  --tabpfn_device cuda --tabpfn_ignore_limits \
  --xgb_max_depth 6 --xgb_reg_lambda 10 --xgb_num_round 8000 --xgb_early_stop 200 --xgb_val_frac 0.10 --xgb_nthread 16 \
  --hoff_add_nmut_features
