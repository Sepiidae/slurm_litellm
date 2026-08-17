#!/bin/bash
#SBATCH --output=logs/ollama_%j.log
#SBATCH --time=4:00:00
PORT=$((11000 + ($SLURM_JOB_ID % 10000)))

export OLLAMA_HOST="0.0.0.0:$PORT"
export OLLAMA_NUM_PARALLEL=20
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_MODELS="/mnt/onefs/scratch/rresnick/ollama/models"
#export OLLAMA_VRAM_ONLY=1

if [ ! -d $OLLAMA_MODELS ] 
then
	mkdir -p $OLLAMA_MODELS
fi

export PATH=$PATH:~/ollama/bin

echo "Starting Ollama instance on port $PORT"

ollama serve &

wait
