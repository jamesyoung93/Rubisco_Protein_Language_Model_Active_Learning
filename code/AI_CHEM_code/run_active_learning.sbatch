#!/bin/bash
#SBATCH -J al_sim
#SBATCH -p compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd /mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed

source /mmfs1/cm/shared/apps_local/mamba/24.3/etc/profile.d/conda.sh
conda activate rubisco_embed
export PYTHONNOUSERSITE=1

python active_learning_sim.py \
  --workdir /mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed \
  --dataset BOTH \
  --outdir results_active_learning_v1 \
  --n_reps 10 --init_n 200 --batch_size 48 --max_rounds 25 \
  --ensemble_size 5 --beta 1.0 \
  --novelty_cutoff_frac 0.20 --novelty_patience 3 --novelty_min_rounds 5 \
  --hoff_constraint_mode hard
