LiteLLM Proxy + Slurm + Ollama == Yes

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Execute on your login node, or similar, the node must be able to connect to nodename:11434, so firewalls need to be out of the way.

source ./venv/bin/activate
python proxy.py

# Install your ollama instances
mkdir -p ~/.local/bin
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
tar -C ~/.local/bin -xzf ollama-linux-amd64.tgz --strip-components 1

# Start your ollama instances
sbatch sbatch.sh

# Start your proxy
./start_proxy
