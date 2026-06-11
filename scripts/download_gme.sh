#!/bin/bash
#SBATCH --job-name=download_gme
#SBATCH --partition=fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/data/divyasaxena_rs/Muskan_internship/MegaRAG/logs/download_%j.out
#SBATCH --error=/scratch/data/divyasaxena_rs/Muskan_internship/MegaRAG/logs/download_%j.err

BASE=/scratch/data/divyasaxena_rs/Muskan_internship

module purge && module load anaconda3/2024
eval "$(conda shell.bash hook)"
conda activate ${BASE}/envs/megarag
export PATH="${BASE}/envs/megarag/bin:$PATH"

export HF_HOME=${BASE}/hf_cache
export TRANSFORMERS_CACHE=${BASE}/hf_cache

echo "Downloading GME model..."
python3 -c "
from transformers import AutoModel
m = AutoModel.from_pretrained(
    'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct',
    trust_remote_code=True
)
print('SUCCESS: GME model downloaded')
"
echo "Done: $(date)"
