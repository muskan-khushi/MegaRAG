#!/bin/bash
#SBATCH --job-name=megarag_build
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/data/divyasaxena_rs/Muskan_internship/MegaRAG/logs/build_%j.out
#SBATCH --error=/scratch/data/divyasaxena_rs/Muskan_internship/MegaRAG/logs/build_%j.err

echo "============================================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Started: $(date)"
echo "============================================================"

BASE=/scratch/data/divyasaxena_rs/Muskan_internship

# Activate megarag env
module purge && module load anaconda3/2024
eval "$(conda shell.bash hook)"
conda activate ${BASE}/envs/megarag
export PATH="${BASE}/envs/megarag/bin:$PATH"

# Start Ollama
cd ${BASE}
OLLAMA_MODELS=${BASE}/ollama_local/models \
    ${BASE}/ollama_local/bin/ollama serve > ${BASE}/ollama_local/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama started (PID: $OLLAMA_PID)"
sleep 10

# Verify Ollama
curl http://localhost:11434/api/tags 2>/dev/null | head -c 100 || {
    echo "ERROR: Ollama failed to start"
    exit 1
}

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())")
echo "CUDA available: $CUDA_OK"

# Load env vars and run pipeline
cd ${BASE}/MegaRAG
source env.sh
cd egs/world_history_tiny
bash run_build_mmkg.sh

EXIT_CODE=$?
echo "============================================================"
echo "Done: $(date) | Exit code: $EXIT_CODE"
echo "============================================================"
kill $OLLAMA_PID 2>/dev/null
exit $EXIT_CODE
