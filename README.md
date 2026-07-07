# LiteLLM Proxy + Slurm + Ollama Architecture

This document outlines the setup, deployment, and execution workflow for managing local or clustered LLM workloads using **Ollama** deployed via **Slurm**, paired with a centralized **LiteLLM Proxy**.

---

## 📋 Prerequisites & Infrastructure Requirements

> [!WARNING]  
> **Network Connectivity Requirement:** > These steps must be executed from your login node (or equivalent coordinator node). The master node **must** have uninterrupted connection capabilities to `nodename:11434`. Please ensure all firewalls, security groups, or internal subnet rules are configured to clear this traffic.
> **These instructions assume installation to $HOME
> **These instructions assume ollama models will be in ~/scratch/ollama/models/

Please update the scripts as needed to adjust this. 
---

## 🛠️ Step 1: Initial Environment Setup

Initialize your local Python environment and install the foundational dependency requirements. 

```bash
# Create a isolated Python virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies (such as litellm, requirements, etc.)
pip install -r requirements.txt
```

## Step 2: Install Ollama
### Install Ollama locally in your home directory
```
cd src
./install_ollama.sh


```

### Note: 
Ollama's home directory ~/.ollama uses large amount of storage, hundres of GB, so make sure to place this one a file system with plenty of storage/quota.
You can specify this location using 
```
# Example OLLAMA_MODELS Location
export OLLAMA_MODELS=~/scratch/ollama/models
```

## Step 3: Start your ollama instances
```
sbatch sbatch.sh
```

## Step 4: Start your proxy
```
./start_proxy
```

## Step 5: Start your SSH forward to your PC
```
ssh username@looginnode.example.com -L 0.0.0.0:11434:loginnodeyoustartedproxyon:8000
```
