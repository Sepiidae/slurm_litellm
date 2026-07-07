#!/bin/bash

if [ ! -f ollama-linux-amd64.tar.zst ]
then
  curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst -o ollama-linux-amd64.tar.zst
fi

zstd -d ollama-linux-amd64.tar.zst 
mkdir ollama
cd ollama
tar -xf ../ollama-linux-amd64.tar

