#!/bin/bash
#SBATCH --job-name=mmkg_only
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=../logs/mmkg_only_%j.out
#SBATCH --error=../logs/mmkg_only_%j.err

set -e

BASE=/scratch/data/divyasaxena_rs/Muskan_internship

module purge
module load anaconda3/2024

eval "$(conda shell.bash hook)"
conda activate ${BASE}/envs/megarag

export PATH="${BASE}/envs/megarag/bin:$PATH"

# Start Ollama
cd ${BASE}

OLLAMA_MODELS=${BASE}/ollama_local/models \
    ${BASE}/ollama_local/bin/ollama serve \
    > ${BASE}/ollama_local/mmkg_ollama.log 2>&1 &

OLLAMA_PID=$!

echo "Ollama started (PID: $OLLAMA_PID)"

sleep 10

curl http://localhost:11434/api/tags 2>/dev/null | head -c 200 || {
    echo "ERROR: Ollama failed to start"
    exit 1
}

echo

cd ${BASE}/MegaRAG/egs/world_history_full

echo "Starting MMKG build at $(date)"

python ../utils/construct_mmkg.py \
    --config-file "./conf/addon_params.yaml" \
    --working-dir "./exp/World_History_Volume_1" \
    --input-dir "./dumps/World_History_Volume_1/pages_content.json"

echo "Finished MMKG build at $(date)"
kill $OLLAMA_PID 2>/dev/null
