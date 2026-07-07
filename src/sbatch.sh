#!/bin/bash
#SBATCH --job-name=ollama_server
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=~/slurm_litellm/logs/ollama-%j.out

# Used to specify the model to follow...
MODEL=$(squeue -j $SLURM_JOB_ID -h -o "%C")
COMMENT=$(squeue -j $SLURM_JOB_ID -h -o "%C")


# Generate a unique key to store the model download based on the model provide

# 0. Start a dependency on this node to keep ollama running when this job ends
sbatch --dependency=afterany:$SLURM_JOB_ID $@ sbatch.sh
echo sbatch --dependency=afterany:$SLURM_JOB_ID $@ sbatch.sh

# 1. Determine this node's exact network IP or hostname, this is a backup incase squeue don't work in proxy.py
NODE_IP=$(hostname -I | awk '{print $1}')
PORT=11434
HEARTBEAT_FILE="~/slurm_litellm/ollama_heartbeats/${SLURM_JOB_ID}"

# 2. Write an active heartbeat entry for the python proxy to pick up
echo "http://${NODE_IP}:${PORT}" > "${HEARTBEAT_FILE}"

# 3. Clean up the heartbeat file when the job terminates or is killed
cleanup() {
    echo "Stopping job and cleaning up entry..."
    rm -f "${HEARTBEAT_FILE}"
}

trap cleanup EXIT INT TERM

# 4. Bind Ollama to all interfaces so the external proxy can talk to it
export OLLAMA_HOST="0.0.0.0:${PORT}"
export OLLAMA_KEEP_ALIVE="30m"


export OLLAMA_MODELS=~/scratch/ollama/models/$(echo $MODEL | sha256)
mkdir -p $OLLAMA_MODELS

# 5. Execute the server
# ~/.local/bin/ollama serve
srun --ntasks=1 --nodes=1 ollama serve &

until curl -s http://${OLLAMA_HOST}/ > /dev/null; do
    sleep 1
done


echo Ollama is ready
ollama pull $MODEL

while curl -s http://${OLLAMA_HOST}/ > /dev/null; do
    sleep 30
done

echo Ollama is dead Jim


