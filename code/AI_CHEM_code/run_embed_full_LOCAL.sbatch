#!/bin/bash
#SBATCH -J esm2_full
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

# move to the code directory that contains embed_esm2.py and rubisco_datasets_embed_input.csv
cd /mmfs1/scratch/jacks.local/jyoung67391/Rubisco_AI_Chemistry_Public_Deposition_Ready/Rubisco_AI_Chemistry_Public_Deposition_Ready/code/AI_CHEM_code

source /mmfs1/cm/shared/apps_local/mamba/24.3/etc/profile.d/conda.sh
conda activate rubisco_embed

# keep caches on scratch
export HF_HOME="/mmfs1/scratch/jacks.local/jyoung67391/Rubisco_AI_Chemistry_Public_Deposition_Ready/Rubisco_AI_Chemistry_Public_Deposition_Ready/.cache/huggingface"
export TORCH_HOME="/mmfs1/scratch/jacks.local/jyoung67391/Rubisco_AI_Chemistry_Public_Deposition_Ready/Rubisco_AI_Chemistry_Public_Deposition_Ready/.cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available())"

python embed_esm2.py \
  --in_csv rubisco_datasets_embed_input.csv \
  --id_col variant_id \
  --seq_col aa_sequence \
  --out_npy esm2_t33_650m_full.npy \
  --model facebook/esm2_t33_650M_UR50D \
  --batch_size 16
