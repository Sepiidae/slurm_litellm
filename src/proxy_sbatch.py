import yaml
import subprocess
import time
import re
import litellm
from litellm import Router
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import urllib.request

def load_config(config_path="jobs_config.yaml"):
    """Reads the YAML configuration file defining jobs and models."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_job_info_by_name(job_name):
    """
    Queries squeue for a specific job name.
    Returns a tuple of (job_id, state, node_name) if found, otherwise (None, None, None).
    """
    cmd = ["squeue", "--name", job_name, "-h", "-o", "%i %t %N"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    output = result.stdout.strip()
    if not output:
        return None, None, None
    
    parts = output.split('\n')[0].split()
    if len(parts) >= 2:
        job_id = parts[0]
        state = parts[1]
        node_name = parts[2] if len(parts) > 2 else None
        return job_id, state, node_name
        
    return None, None, None

def launch_slurm_job(job_name):
    """Submits a new job to Slurm with a specific explicit name attribute."""
    cmd = ["sbatch", f"--job-name={job_name}", "sbatch.sh"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to submit Slurm job '{job_name}': {result.stderr.strip()}")
        return None
    
    match = re.search(r"\d+", result.stdout)
    return match.group(0) if match else None

def resolve_cluster_endpoint(job_name, models, timeout=300):
    """
    Runs in a background thread. Resolves an existing cluster or spins up a new one,
    tracking the lifecycle strictly via the Slurm Job Name, and triggers automatic pulls.
    """
    start_time = time.time()
    job_id, state, node_name = get_job_info_by_name(job_name)
    
    if job_id and state == "R" and node_name and "CONFIGURING" not in node_name:
        print(f"♻️  Found existing running cluster '{job_name}' (Job ID: {job_id}) on node {node_name}")
    else:
        if job_id:
            print(f"⏳ Found existing cluster '{job_name}' but status is {state}. Waiting for allocation...")
        else:
            print(f"🔍 Cluster '{job_name}' not found. Submitting fresh Slurm job...")
            job_id = launch_slurm_job(job_name)
            if not job_id:
                return []
                
    while time.time() - start_time < timeout:
        job_id, state, node_name = get_job_info_by_name(job_name)
        
        if state == "R" and node_name and "CONFIGURING" not in node_name:
            port = 11000 + (int(job_id) % 10000)
            endpoint = f"http://{node_name}:{port}"
            print(f"✅ Cluster '{job_name}' is fully active at {endpoint}")
            
            # --- NEW: Automatically trigger pulls for specified models ---
            for model in models:
                try:
                    print(f"📥 Instructing cluster '{job_name}' ({node_name}) to pull model '{model}'...")
                    pull_url = f"{endpoint}/api/pull"
                    payload = json.dumps({"model": model, "stream": False}).encode("utf-8")
                    
                    req = urllib.request.Request(
                        pull_url, 
                        data=payload, 
                        headers={"Content-Type": "application/json"}
                    )
                    # We open the connection to start the download, but use a brief timeout 
                    # so a huge model pull doesn't freeze the whole orchestrator startup.
                    with urllib.request.urlopen(req, timeout=15) as response:
                        if response.status == 200:
                            print(f"📡 Pull initiated/verified for '{model}' on cluster '{job_name}'")
                except Exception as pull_err:
                    # Often throws a timeout error if 'stream': False takes longer than 15s to download,
                    # which is expected and fine since Ollama will continue pulling it in the background.
                    print(f"ℹ️  Pull context dispatched for '{model}' (Backend downloading or already present).")
            # -----------------------------------------------------------------

            job_models = []
            for model in models:
                print(endpoint)
                job_models.append({
                    "model_name": model,
                    "litellm_params": {
                        "model": f"ollama/{model}",
                        "api_base": f"{endpoint}"
                    }
                })
            return job_models
            
        time.sleep(5)
        
    print(f"❌ Timeout: Cluster '{job_name}' failed to resolve within {timeout}s.")
    return []

def main():
    config = load_config()
    model_list = []
    
    print("--- Phase 1: Processing Clusters via Slurm Names ---")
    
    with ThreadPoolExecutor(max_workers=len(config["jobs"])) as executor:
        futures = {
            executor.submit(
                resolve_cluster_endpoint, 
                job_spec["job_name"], 
                job_spec["models"]
            ): job_spec for job_spec in config["jobs"]
        }
        
        for future in as_completed(futures):
            try:
                results = future.result()
                if results:
                    model_list.extend(results)
            except Exception as e:
                print(f"An error occurred resolving a cluster: {e}")

    if not model_list:
        print("\n❌ Failed to connect to or provision any clusters. Router aborted.")
        return

    print("\n--- Phase 2: Generating Dynamic LiteLLM Configuration ---")
    
    # Pack the model list into LiteLLM's native configuration format
    proxy_config = {"model_list": model_list}
    config_filename = "dynamic_litellm_config.yaml"
    
    with open(config_filename, "w") as f:
        yaml.dump(proxy_config, f, default_flow_style=False)
    print(f"💾 Dynamic backend layout saved to {config_filename}")

    print("\n--- Phase 3: Launching LiteLLM Gateway Server ---")
    from litellm.proxy.proxy_cli import run_server
    
    print("🚀 LiteLLM Router Proxy is booting on http://0.0.0.0:8000")
    
    # run_server expects command line array arguments just like the terminal CLI tool
    cli_args = [
        "--config", config_filename,
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    # This invokes the proxy natively and handles the continuous server loop without crashing
    run_server(cli_args, standalone_mode=False)

if __name__ == "__main__":
    main()
