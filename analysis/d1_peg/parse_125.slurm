#!/bin/bash
#SBATCH --job-name=parse125
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=parse125.out
#SBATCH --error=parse125.err

BASE=/scratch/data/divyasaxena_rs/Muskan_internship

module purge && module load anaconda3/2024
eval "$(conda shell.bash hook)"
conda activate ${BASE}/envs/megarag
export PATH="${BASE}/envs/megarag/bin:$PATH"

cd ${BASE}/MegaRAG/analysis/d1_peg

python parse_llm_judge_125.py
