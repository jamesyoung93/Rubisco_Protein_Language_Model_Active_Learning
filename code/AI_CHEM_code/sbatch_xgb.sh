#!/bin/bash
#SBATCH -J al_xgb
#SBATCH -p compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd /mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed

source /mmfs1/cm/shared/apps_local/mamba/24.3/etc/profile.d/conda.sh
conda activate rubisco_embed
export PYTHONNOUSERSITE=1

# Prevent thread oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs

python active_learning_pubready.py \
  --emb_npy esm2_t33_650m_full.npy \
  --labels_csv rubisco_datasets_merged.csv \
  --out_dir results_active_learning_pubready_v1 \
  --datasets BOTH \
  --surrogates XGB \
  --strategies random,greedy,uncertainty,ucb,thompson \
  --n_reps 20 --init_n 200 --batch_size 48 --max_rounds 25 --beta 1.0 \
  --novelty_cutoff_frac 0.20 --novelty_patience 3 --novelty_min_rounds 5 \
  --dms_target dms_enrichment_mean --dms_pca_dim 128 --dms_pos_coverage_stop 0.95 \
  --hoff_objective delta_direct --hoff_pca_dim 128 --hoff_add_nmut_features --hoff_n2_margin 0.10 \
  --constraint_mode hard --constraint_lambda 5.0 \
  --xgb_ens 5 --xgb_rounds 2000 --xgb_early_stop 100 --xgb_val_frac 0.10 \
  --xgb_max_depth 6 --xgb_reg_lambda 10 --xgb_nthread $SLURM_CPUS_PER_TASK
