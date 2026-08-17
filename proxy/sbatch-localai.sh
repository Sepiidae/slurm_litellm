#!/bin/bash
#SBATCH --mem=0
#SBATCH -p shortq7
#SBATCH --gres=gpu:a100:3
#SBATCH --exclusive
#SBATCH --output=logs/localai_%j.log

if [ ! ~/scratch/image ] 
then
    echo Creating image directory 
    mkdir ~/scratch/image
fi


# Ensure local bind directories exist to prevent Apptainer mount errors
mkdir -p images data backends configuration models logs ~/scratch/image/tmp ~/scratch/image/apptainer

# Cleanup function for background processes and temporary files
cleanup() {
    echo "Cleaning up local processes and temporary files..."
    if [ -n "$AUTOSSH_PID" ]; then
        kill "$AUTOSSH_PID" 2>/dev/null
    fi
    rm -rf "/tmp/localai-$UID"
}

trap cleanup EXIT INT TERM


# Dynamic port selection based on Slurm Job ID
export PORT=$((11000 + (SLURM_JOB_ID % 10000)))

echo "Allocated Port: $PORT"

# Start reverse SSH tunnel in background and save PID
#autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -N xantho@home.scuba2deth.com -R 8156:0.0.0.0:$PORT &
autossh -M 0 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N xantho@home.scuba2deth.com -R 8156:0.0.0.0:$PORT &
AUTOSSH_PID=$!

while sleep 5
do
# Launch LocalAI container
pwd
./start_apptainer.sh $PORT
done


wait
