import os
import re
import sys
import time
import json
import yaml
import logging
import subprocess
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("orchestrator")

def load_config(config_path="jobs_config.yaml"):
    """Reads the YAML configuration file defining jobs and models."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_job_info_by_name(job_name):
    """
    Queries squeue for a specific job name.
    Filters and returns the active running ('R') job if multiple exist.
    Returns a tuple of (job_id, state, node_name) if found, otherwise (None, None, None).
    """
    try:
        cmd = ["squeue", "--name", job_name, "-h", "-o", "%i %t %N"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        output = result.stdout.strip()
        if not output:
            return None, None, None

        lines = [line.split() for line in output.split('\n') if line.strip()]
        
        # Look for a running job ('R') first
        running_job = next((parts for parts in lines if len(parts) >= 2 and parts[1] == "R"), None)
        
        # Fallback to the first available job if none are running yet
        selected_parts = running_job if running_job else lines[0]

        if len(selected_parts) >= 2:
            job_id = selected_parts[0]
            state = selected_parts[1]
            node_name = selected_parts[2] if len(selected_parts) > 2 else None
            return job_id, state, node_name
    except Exception as e:
        logger.warning(f"Error querying squeue: {e}")

    return None, None, None

def launch_slurm_job(job_name):
    """Submits a new job to Slurm with a specific explicit name attribute."""
    cmd = ["sbatch", f"--job-name={job_name}", "sbatch.sh"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to submit Slurm job '{job_name}': {result.stderr.strip()}")
        return None

    match = re.search(r"\d+", result.stdout)
    return match.group(0) if match else None

# Cache states
PULLED_MODELS_CACHE = set()
PENDING_PULLS = set()
PENDING_PULLS_LOCK = threading.Lock()

def _async_pull_model(endpoint, model, job_name, node_name, cache_key):
    """Runs in a background thread to prevent blocking the main loop."""
    start_time = time.time()
    try:
        logger.info(f"📥 [Pull Started] Pulling '{model}' on cluster '{job_name}' ({node_name}) -> URL: {endpoint}/api/pull")
        pull_url = f"{endpoint}/api/pull"
        payload = json.dumps({"model": model, "stream": False}).encode("utf-8")

        req = urllib.request.Request(
            pull_url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        # Allow ample time for the model download to complete in the background
        with urllib.request.urlopen(req, timeout=300) as response:
            duration = time.time() - start_time
            if response.status == 200:
                logger.info(f"✅ [Pull Completed] Successfully pulled '{model}' on cluster '{job_name}' in {duration:.2f}s")
                PULLED_MODELS_CACHE.add(cache_key)
            else:
                logger.error(f"❌ [Pull Failed] Server returned status {response.status} for '{model}' on '{job_name}' after {duration:.2f}s")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ [Pull Failed] Failed to pull '{model}' on '{job_name}' after {duration:.2f}s: {e}")
    finally:
        with PENDING_PULLS_LOCK:
            PENDING_PULLS.discard(cache_key)

def process_single_cluster(job_spec):
    """
    Non-blocking check for a single cluster.
    """
    job_name = job_spec["job_name"]
    models = job_spec["models"]

    job_id, state, node_name = get_job_info_by_name(job_name)

    # 1. If cluster doesn't exist, launch it
    if not job_id:
        logger.info(f"🔍 Cluster '{job_name}' not found. Submitting fresh Slurm job...")
        job_id = launch_slurm_job(job_name)
        return []

    # 2. If cluster is starting/configuring, wait
    if state != "R" or not node_name or "CONFIGURING" in node_name:
        logger.info(f"⏳ Cluster '{job_name}' (Job ID: {job_id}) is in state '{state}'. Waiting...")
        return []

    # 3. If cluster is fully active
    port = 11000 + (int(job_id) % 10000)
    endpoint = f"http://{node_name}:{port}"

    # Trigger model pulls asynchronously
    for model in models:
        cache_key = f"{endpoint}/{model}"
        
        # Double-check lock-free check
        if cache_key not in PULLED_MODELS_CACHE:
            with PENDING_PULLS_LOCK:
                if cache_key not in PENDING_PULLS:
                    PENDING_PULLS.add(cache_key)
                    
                    # Spin up pull action in an independent background thread
                    pull_thread = threading.Thread(
                        target=_async_pull_model,
                        args=(endpoint, model, job_name, node_name, cache_key),
                        daemon=True
                    )
                    pull_thread.start()

    # Build and return the configuration for LiteLLM
    job_models = []
    for model in models:
        llm_type = "ollama_chat"
        if "embed" in model:
            llm_type = "ollama"

        job_models.append({
            "model_name": model,
            "litellm_params": {
                "model": f"{llm_type}/{model}",
                "api_base": f"{endpoint}",
                "max_parallel_requests": 5,
                "tool_choice": "none"
            }
        })
    return job_models

def start_litellm_proxy(config_filename):
    """Launches the LiteLLM Gateway Server in a background thread."""
    from litellm.proxy.proxy_cli import run_server

    logger.info("🚀 LiteLLM Router Proxy is booting on http://0.0.0.0:8000")
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

    logger.info("--- Phase 1: Launching LiteLLM Gateway (Background) ---")
    proxy_thread = threading.Thread(
        target=start_litellm_proxy,
        args=(config_filename,),
        daemon=True
    )
    proxy_thread.start()

    time.sleep(2)

    logger.info("--- Phase 2: Starting 5-Second Orchestration Loop ---")

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
                        logger.error(f"Error checking cluster '{job_spec.get('job_name')}': {e}")

            proxy_config = {"model_list": active_models}

            # Read existing file state to prevent redundant writes
            current_on_disk = None
            if os.path.exists(config_filename):
                with open(config_filename, "r") as f:
                    try:
                        current_on_disk = yaml.safe_load(f)
                    except Exception:
                        pass

            # If the backend targets have changed, write them and log the manifest update
            if current_on_disk != proxy_config:
                with open(config_filename, "w") as f:
                    yaml.dump(proxy_config, f, default_flow_style=False)

                logger.info("🔄 [Routing Table Updated] Active Routing Registry changed:")
                if active_models:
                    for entry in active_models:
                        m_name = entry["model_name"]
                        ep = entry["litellm_params"]["api_base"]
                        logger.info(f"    🔹 Model: {m_name:<20} ➡️ Running at: {ep}")
                else:
                    logger.warning("    ⚠️ No active backends are currently mapped.")

        except Exception as loop_err:
            logger.error(f"Error in main orchestrator loop: {loop_err}")

        time.sleep(5)

if __name__ == "__main__":
    main()
