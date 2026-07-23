#!/bin/bash
#SBATCH --job-name=ollama-server
#SBATCH --output=logs/ollama_%j.log
#SBATCH --gres=gpu:v100:4
#SBATCH --exclusive --mem=0
#SBATCH --time=04:00:00

PORT=$((11000 + (SLURM_JOB_ID % 10000)))
#echo Starting dependency on $SLURM_JOB_ID
#sbatch --job-name="$SLURM_JOB_NAME" --dependency=afterany:$SLURM_JOB_ID sbatch.sh $@
export OLLAMA_RUNNERS_PATH="/mnt/beegfs/home/rresnick/scratch/ollama/runners"
export OLLAMA_MODELS="/mnt/beegfs/home/rresnick/scratch/ollama/models"

# Physically construct the destination paths to grant file system write locks
mkdir -p "$OLLAMA_RUNNERS_PATH"
mkdir -p "$OLLAMA_MODELS"

export OLLAMA_NUM_PARALLEL=20
#PORT=11434
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_HOST="0.0.0.0:$PORT"
# export OLLAMA_MODELS=~/scratch/ollama/models
echo "Starting Ollama instance on port $PORT"
ollama/bin/ollama serve
