#!/bin/bash
#SBATCH -J rub_tabpfn
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd /mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed

source /mmfs1/cm/shared/apps_local/mamba/24.3/etc/profile.d/conda.sh
conda activate rubisco_embed
export PYTHONNOUSERSITE=1

export HF_HOME=/mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed/hf_cache
mkdir -p "$HF_HOME" logs

python rubisco_compare_tabpfn_xgb.py \
  --emb_npy esm2_t33_650m_full.npy \
  --labels_csv rubisco_datasets_merged.csv \
  --out_dir results_rubisco_tabpfn_vs_xgb \
  --pca_dim 128 \
  --dms_target dms_enrichment_mean \
  --hoff_target hoff_delta_O2_minus_N2 \
  --xgb_max_depth 6 --xgb_reg_lambda 10 --xgb_num_round 8000 --xgb_early_stop 200 --xgb_nthread 16 \
  --tabpfn_device cuda --tabpfn_train_cap 5000 --tabpfn_ignore_limits
