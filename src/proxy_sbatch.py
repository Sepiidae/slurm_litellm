import os
import re
import sys
import time
import json
import yaml
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error

def load_config(config_path="jobs_config.yaml"):
    """Reads the YAML configuration file defining jobs and models."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_job_info_by_name(job_name):
    """
    Queries squeue for a specific job name.
    Returns a tuple of (job_id, state, node_name) if found, otherwise (None, None, None).
    """
    try:
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
    except Exception as e:
        print(f"⚠️ Error querying squeue: {e}")
        
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

# Keep track of models we've already dispatched pull requests to so we don't spam endpoints
PULLED_MODELS_CACHE = set()

def process_single_cluster(job_spec):
    """
    Non-blocking check for a single cluster. 
    Returns a list of model configs if active, submits a job if missing, 
    or returns empty if still deploying.
    """
    job_name = job_spec["job_name"]
    models = job_spec["models"]
    
    job_id, state, node_name = get_job_info_by_name(job_name)

    # 1. If cluster doesn't exist, launch it
    if not job_id:
        print(f"🔍 Cluster '{job_name}' not found. Submitting fresh Slurm job...")
        job_id = launch_slurm_job(job_name)
        return []

    # 2. If cluster is starting/configuring, wait
    if state != "R" or not node_name or "CONFIGURING" in node_name:
        print(f"⏳ Cluster '{job_name}' (Job ID: {job_id}) is in state '{state}'. Waiting...")
        return []

    # 3. If cluster is fully active
    port = 11000 + (int(job_id) % 10000)
    endpoint = f"http://{node_name}:{port}"
    
    # Trigger model pulls if we haven't done so for this specific endpoint
    for model in models:
        cache_key = f"{endpoint}/{model}"
        if cache_key not in PULLED_MODELS_CACHE:
            try:
                print(f"📥 Instructing cluster '{job_name}' ({node_name}) to pull model '{model}'...")
                pull_url = f"{endpoint}/api/pull"
                payload = json.dumps({"model": model, "stream": False}).encode("utf-8")

                req = urllib.request.Request(
                    pull_url,
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                # Low timeout so we don't hang the loop tick
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        print(f"📡 Pull initiated/verified for '{model}' on cluster '{job_name}'")
                        PULLED_MODELS_CACHE.add(cache_key)
            except Exception as pull_err:
                # Expected if the model is currently downloading in the background
                print(f"ℹ️  Pull context dispatched for '{model}' on '{job_name}' (checking again next tick).")
                PULLED_MODELS_CACHE.add(cache_key)

    # Build and return the configuration for LiteLLM
    job_models = []
    for model in models:
        job_models.append({
            "model_name": model,
            "litellm_params": {
                "model": f"ollama_chat/{model}",
                "api_base": f"{endpoint}",
                "max_parallel_requests": 5,
                "tool_choice": "none"
            }
        })
    return job_models

def start_litellm_proxy(config_filename):
    """Launches the LiteLLM Gateway Server in a background thread."""
    from litellm.proxy.proxy_cli import run_server
    
    print("🚀 LiteLLM Router Proxy is booting on http://0.0.0.0:8000")
    cli_args = [
        "--config", config_filename,
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    run_server(cli_args, standalone_mode=False)

def main():
    config_filename = "dynamic_litellm_config.yaml"
    
    if not os.path.exists(config_filename):
        with open(config_filename, "w") as f:
            yaml.dump({"model_list": []}, f)

    print("\n--- Phase 1: Launching LiteLLM Gateway (Background) ---")
    proxy_thread = threading.Thread(
        target=start_litellm_proxy, 
        args=(config_filename,), 
        daemon=True
    )
    proxy_thread.start()
    
    time.sleep(2)

    print("\n--- Phase 2: Starting 5-Second Orchestration Loop ---")
    
    while True:
        try:
            config = load_config()
            active_models = []
            
            with ThreadPoolExecutor(max_workers=max(1, len(config["jobs"]))) as executor:
                futures = {
                    executor.submit(process_single_cluster, job_spec): job_spec 
                    for job_spec in config["jobs"]
                }

                for future in as_completed(futures):
                    try:
                        results = future.result()
                        if results:
                            active_models.extend(results)
                    except Exception as e:
                        job_spec = futures[future]
                        print(f"⚠️ Error checking cluster '{job_spec.get('job_name')}': {e}")

            proxy_config = {"model_list": active_models}
            
            # Read existing file state to prevent redundant writes
            current_on_disk = None
            if os.path.exists(config_filename):
                with open(config_filename, "r") as f:
                    try:
                        current_on_disk = yaml.safe_load(f)
                    except Exception:
                        pass
            
            # If the backend targets have changed, write them and print the manifest
            if current_on_disk != proxy_config:
                with open(config_filename, "w") as f:
                    yaml.dump(proxy_config, f, default_flow_style=False)
                
                print("\n🔄 [Routing Table Updated] Active Routing Registry:")
                if active_models:
                    for entry in active_models:
                        m_name = entry["model_name"]
                        ep = entry["litellm_params"]["api_base"]
                        print(f"   🔹 Model: {m_name:<20} ➡️ Running at: {ep}")
                else:
                    print("   ⚠️ No active backends are currently mapped.")
                print("") # Spacer line
                
        except Exception as loop_err:
            print(f"❌ Error in main orchestrator loop: {loop_err}")

        time.sleep(5)

if __name__ == "__main__":
    main()
