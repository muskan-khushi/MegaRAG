#!/bin/bash
#SBATCH --job-name=d1_125
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=d1_125.out
#SBATCH --error=d1_125.err

BASE=/scratch/data/divyasaxena_rs/Muskan_internship

# Environment
module purge && module load anaconda3/2024
eval "$(conda shell.bash hook)"
conda activate ${BASE}/envs/megarag
export PATH="${BASE}/envs/megarag/bin:$PATH"

cd ${BASE}/MegaRAG/analysis/d1_peg

echo "================================================="
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE:   $SLURM_NODELIST"
echo "START:  $(date)"
echo "================================================="

# Start Ollama
OLLAMA_MODELS=${BASE}/ollama_local/models \
    ${BASE}/ollama_local/bin/ollama serve \
    > ${BASE}/ollama_local/peg_ollama.log 2>&1 &

OLLAMA_PID=$!

echo "Ollama started (PID: $OLLAMA_PID)"
sleep 10

curl http://localhost:11434/api/tags 2>/dev/null | head -c 200 || {
    echo "ERROR: Ollama failed to start"
    exit 1
}
echo

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())")
echo "CUDA available: $CUDA_OK"

# Run D1 experiment
python peg_generate_125.py

echo "Finished: $(date)"

kill $OLLAMA_PID 2>/dev/null
