#!/bin/bash
#SBATCH --job-name=megarag_query
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/data/divyasaxena_rs/Muskan_internship/MegaRAG/logs/query_%j.out
#SBATCH --error=/scratch/data/divyasaxena_rs/Muskan_internship/MegaRAG/logs/query_%j.err

echo "============================================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Started: $(date)"
echo "============================================================"

BASE=/scratch/data/divyasaxena_rs/Muskan_internship

# Activate environment
module purge && module load anaconda3/2024
eval "$(conda shell.bash hook)"
conda activate ${BASE}/envs/megarag
export PATH="${BASE}/envs/megarag/bin:$PATH"

# Start Ollama
cd ${BASE}
OLLAMA_MODELS=${BASE}/ollama_local/models \
    ${BASE}/ollama_local/bin/ollama serve > ${BASE}/ollama_local/query_ollama.log 2>&1 &
OLLAMA_PID=$!

echo "Ollama started (PID: $OLLAMA_PID)"
sleep 10


# Verify Ollama
curl http://localhost:11434/api/tags 2>/dev/null | head -c 200 || {
    echo "ERROR: Ollama failed to start"
    exit 1
}
echo

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())")
echo "CUDA available: $CUDA_OK"


# Load env vars
cd ${BASE}/MegaRAG
source env.sh

echo "===== PYTHON DEBUG ====="

which python
python -V

/scratch/data/divyasaxena_rs/Muskan_internship/envs/megarag/bin/python -V

/scratch/data/divyasaxena_rs/Muskan_internship/envs/megarag/bin/python -c "
import sys
print(sys.executable)
"

echo "========================"

# Run query pipeline
cd egs/world_history_tiny
bash run_quering.sh

EXIT_CODE=$?

echo "============================================================"
echo "Done: $(date) | Exit code: $EXIT_CODE"
echo "============================================================"

kill $OLLAMA_PID 2>/dev/null
exit $EXIT_CODE
