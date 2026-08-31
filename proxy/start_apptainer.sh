#!/bin/bash

PORT=$1

module load apptainer/1.1.9-gcc-13.2.0-apqpu4x

#export APPTAINERENV_LLAMA_SPLIT_MODE=layer

export APPTAINER_CACHEDIR=~/scratch/image/apptainer
export APPTAINER_TMPDIR=~/scratch/image/tmp

cd ~/scratch/image/


apptainer run --nv \
  --bind images:/generated \
  --bind data:/data \
  --bind backends:/backends \
  --bind configuration:/configuration \
  --bind models:/models \
  docker://localai/localai:latest-gpu-nvidia-cuda-13 \
  --address=0.0.0.0:$PORT
seconds=5
echo "Sleeping $seconds seconds..."
sleep $seconds

