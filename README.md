# LiteLLM Proxy + Slurm + Ollama Architecture

This document outlines the setup, deployment, and execution workflow for managing local or clustered LLM workloads using **Ollama** deployed via **Slurm**, paired with a centralized **LiteLLM Proxy**.

---

## 📋 Prerequisites & Infrastructure Requirements

> [!WARNING]  
> **Network Connectivity Requirement:** > These steps must be executed from your login node (or equivalent coordinator node). The master node **must** have uninterrupted connection capabilities to `nodename:11434`. Please ensure all firewalls, security groups, or internal subnet rules are configured to clear this traffic.

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

