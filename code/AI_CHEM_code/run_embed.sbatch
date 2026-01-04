#!/bin/bash
#SBATCH -J esm2_embed
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

# Work entirely in scratch
cd /mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed

# Activate conda reliably in a non-interactive batch context
source /mmfs1/cm/shared/apps_local/mamba/24.3/etc/profile.d/conda.sh
conda activate rubisco_embed

# Prevent ~/.local packages from interfering
export PYTHONNOUSERSITE=1

# Put HF cache on scratch
export HF_HOME="/mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Put temp files on scratch
export TMPDIR="/mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed/tmp"
mkdir -p "$TMPDIR"

python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('torch cuda', torch.version.cuda)"

python embed_esm2.py \
  --in_csv rubisco_datasets_embed_input.csv \
  --id_col variant_id \
  --seq_col aa_sequence \
  --out_npy esm2_t33_650m_embeddings.npy \
  --model facebook/esm2_t33_650M_UR50D \
  --batch_size 8
