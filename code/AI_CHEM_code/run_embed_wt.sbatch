#!/bin/bash
#SBATCH -J esm2_wt
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd /mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed

source /mmfs1/cm/shared/apps_local/mamba/24.3/etc/profile.d/conda.sh
conda activate rubisco_embed
export PYTHONNOUSERSITE=1

export HF_HOME="/mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed/hf_cache"
mkdir -p "$HF_HOME"

python embed_esm2.py \
  --in_csv wt_sequences_for_embedding.csv \
  --id_col variant_id \
  --seq_col aa_sequence \
  --out_npy esm2_wt_embeddings.npy \
  --model facebook/esm2_t33_650M_UR50D \
  --batch_size 2
