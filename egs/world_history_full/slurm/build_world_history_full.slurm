#!/bin/bash
#SBATCH --job-name=wh_full_mmkg
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=../logs/wh_full_%j.out
#SBATCH --error=../logs/wh_full_%j.err

set -e

BASE=/scratch/data/divyasaxena_rs/Muskan_internship

module purge
module load anaconda3/2024

eval "$(conda shell.bash hook)"
conda activate ${BASE}/envs/megarag

export PATH="${BASE}/envs/megarag/bin:$PATH"

cd ${BASE}/MegaRAG/egs/world_history_full

echo "Starting build at $(date)"
bash run_build_mmkg_full.sh
echo "Finished at $(date)"
