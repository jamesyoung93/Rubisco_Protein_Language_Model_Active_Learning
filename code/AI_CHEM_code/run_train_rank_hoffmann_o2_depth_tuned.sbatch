#!/bin/bash
#SBATCH -J hoff_depth
#SBATCH -p compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd /mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed

source /mmfs1/cm/shared/apps_local/mamba/24.3/etc/profile.d/conda.sh
conda activate rubisco_embed
export PYTHONNOUSERSITE=1

python train_rank_hoffmann_o2_depth_tuned.py
