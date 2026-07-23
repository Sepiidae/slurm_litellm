#!/bin/bash

# Configuration
JOB_NAME="bulk-llm"
#JOB_NAME="ollama-server"

echo "Checking for running Ollama Slurm jobs..."

# 1. Query squeue for the Job ID and running Compute Node Name
SLURM_INFO=$(squeue --me --name="$JOB_NAME" --states=RUNNING -h -o "%i %N")

if [ -z "$SLURM_INFO" ]; then
    echo "ERROR: No running Slurm job found with name '$JOB_NAME'."
    echo "Please verify your sbatch job is active (not PENDING)."
    exit 1
fi

JOB_ID=$(echo "$SLURM_INFO" | awk '{print $1}')
NODE_NAME=$(echo "$SLURM_INFO" | awk '{print $2}')

echo "Found running Job ID: $JOB_ID on Node: $NODE_NAME"

# 2. Replicate the sbatch port calculation formula
PORT=$((11000 + (JOB_ID % 10000)))
echo "Calculated target port: $PORT"

# 3. Export the target environment address
export OLLAMA_HOST="http://${NODE_NAME}:${PORT}"

echo "--------------------------------------------------------"
echo "Client environment successfully configured!"
echo "Target Host: $OLLAMA_HOST"
echo "--------------------------------------------------------"

# 4. Verify connectivity and spawn an isolated interactive shell
echo "Testing connection to the cluster backend..."
if curl -s --connect-timeout 3 "${OLLAMA_HOST}/api/version" > /dev/null; then
    echo "Success! Ollama backend is responding."
else
    echo "WARNING: Could not ping server yet. It might still be initializing."
fi

echo "Launching sub-shell. Type 'exit' to disconnect."
bash --init-file <(echo "export OLLAMA_HOST='$OLLAMA_HOST'; echo 'Connected to Slurm Ollama node [${NODE_NAME}]. Run your commands now.'")

