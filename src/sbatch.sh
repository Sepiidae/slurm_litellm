#!/bin/bash
#SBATCH --job-name=ollama_server
#SBATCH --time=02:00:00
#SBATCH --output=logs/ollama-%j.out
#SBATCH --mem=0 --exclusive --gres=gpu:v100:4
cd "$(dirname "$0")" || exit

export PATH=$PATH:/mnt/beegfs/home/rresnick/slurm_litellm/src/ollama/bin

# Used to specify the model to follow...
if [ "$COMMENT" == "" ]
then

  MODEL=$(squeue  --Format comment -h -j $SLURM_JOB_ID)
  COMMENT="$MODEL"
  echo $COMMENT
else
  MODEL=$COMMENT
fi
echo Args: $@
# Generate a unique key to store the model download based on the model provide

# 0. Start a dependency on this node to keep ollama running when this job ends
echo Starting dependency on $SLURM_JOB_ID
sbatch --comment="$COMMENT" --dependency=afterany:$SLURM_JOB_ID sbatch.sh
echo sbatch --comment="$COMMENT" --dependency=afterany:$SLURM_JOB_ID sbatch.sh

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


export OLLAMA_MODELS=~/scratch/ollama/models/$(echo $MODEL | sha256sum |   awk '{print $1}')
mkdir -p $OLLAMA_MODELS

# 5. Execute the server
# ~/.local/bin/ollama serve
ollama serve &

until curl -s http://${OLLAMA_HOST}/ > /dev/null; do
    sleep 1
done

echo $OLLAMA_MODELS
echo Ollama is ready
echo ollama pull $MODEL
for M in $MODEL
do
	echo pulling $M
	ollama pull $M
done


while curl -s http://${OLLAMA_HOST}/ > /dev/null; do
    sleep 5
done

echo Ollama is dead Jim


