#!/bin/bash
#SBATCH -J esm2_full
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd /mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed

source /mmfs1/cm/shared/apps_local/mamba/24.3/etc/profile.d/conda.sh
conda activate rubisco_embed

export PYTHONNOUSERSITE=1

# Use HF_HOME only (no TRANSFORMERS_CACHE needed)
export HF_HOME="/mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed/hf_cache"
export TMPDIR="/mmfs1/scratch/jacks.local/jyoung67391/rubisco/esm2_embed/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"

date
python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('torch cuda', torch.version.cuda)"

python embed_esm2.py \
  --in_csv rubisco_datasets_embed_input.csv \
  --id_col variant_id \
  --seq_col aa_sequence \
  --out_npy esm2_t33_650m_full.npy \
  --model facebook/esm2_t33_650M_UR50D \
  --batch_size 16

date
