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

def get_jobs_info_by_name(job_name):
    """Queries squeue for all jobs matching a specific job name."""
    try:
        cmd = ["squeue", "--me", "--name", job_name, "-h", "-o", "%i %t %N"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        output = result.stdout.strip()
        if not output:
            return []

        jobs = []
        lines = [line.split() for line in output.split('\n') if line.strip()]
        for parts in lines:
            if len(parts) >= 2:
                job_id = parts[0]
                state = parts[1]
                node_name = parts[2] if len(parts) > 2 else None
                jobs.append((job_id, state, node_name))
        return jobs
    except Exception as e:
        logger.warning(f"Error querying squeue: {e}")
    return []

def launch_slurm_job(job_name, gres=None, mem=None, exclusive=False):
    """Submits a new job to Slurm with optional resource configuration flags."""
    cmd = ["sbatch", f"--job-name={job_name}"]
    
    if gres:
        cmd.append(f"--gres={gres}")
    if mem:
        cmd.append(f"--mem={mem}")
    if exclusive:
        cmd.append("--exclusive")
        
    cmd.append("sbatch.sh")

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
        logger.info(f"📥 [Pull Started] Pulling '{model}' on cluster '{job_name}' ({node_name})")
        pull_url = f"{endpoint}/api/pull"
        payload = json.dumps({"model": model, "stream": False}).encode("utf-8")
        req = urllib.request.Request(pull_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            duration = time.time() - start_time
            if response.status == 200:
                logger.info(f"✅ [Pull Completed] Successfully pulled '{model}' on cluster '{job_name}' in {duration:.2f}s")
                PULLED_MODELS_CACHE.add(cache_key)
    except Exception as e:
        logger.error(f"❌ [Pull Failed] Failed to pull '{model}' on '{job_name}': {e}")
    finally:
        with PENDING_PULLS_LOCK:
            PENDING_PULLS.discard(cache_key)

def process_single_cluster(job_spec):
    """Non-blocking check for cluster jobs matching job_spec."""
    job_name = job_spec["job_name"]
    models = job_spec["models"]
    
    # Extract configurable parameters with sensible defaults
    target_count = job_spec.get("num_jobs", job_spec.get("count", 1))
    gres = job_spec.get("gres", None)
    mem = job_spec.get("mem", job_spec.get("memory", None))
    exclusive = job_spec.get("exclusive", False)

    existing_jobs = get_jobs_info_by_name(job_name)
    current_count = len(existing_jobs)

    # Launch additional job instances if running under target count
    if current_count < target_count:
        needed = target_count - current_count
        logger.info(f"🔍 Cluster '{job_name}' has {current_count}/{target_count} jobs running. Submitting {needed} new job(s)...")
        for _ in range(needed):
            launch_slurm_job(job_name, gres=gres, mem=mem, exclusive=exclusive)

    job_models = []

    # Process all active instances
    for job_id, state, node_name in existing_jobs:
        if state != "R" or not node_name or "CONFIGURING" in node_name:
            logger.info(f"⏳ Cluster '{job_name}' (Job ID: {job_id}) is in state '{state}'. Waiting...")
            continue

        port = 11000 + (int(job_id) % 10000)
        endpoint = f"http://{node_name}:{port}"

        # Trigger model downloads asynchronously for running instances
        for model in models:
            cache_key = f"{endpoint}/{model}"
            if cache_key not in PULLED_MODELS_CACHE:
                with PENDING_PULLS_LOCK:
                    if cache_key not in PENDING_PULLS:
                        PENDING_PULLS.add(cache_key)
                        pull_thread = threading.Thread(
                            target=_async_pull_model,
                            args=(endpoint, model, job_name, node_name, cache_key),
                            daemon=True
                        )
                        pull_thread.start()

        # Build active route endpoints for LiteLLM router
        for model in models:
            llm_type = "ollama_chat" if "embed" not in model else "ollama"
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

    time.sleep(3)  # Allow explicit time for proxy components to bind

    logger.info("--- Phase 2: Starting 5-Second Orchestration Loop ---")

    while True:
        try:
            config = load_config()
            active_models = []

            with ThreadPoolExecutor(max_workers=max(1, len(config.get("jobs", [])))) as executor:
                futures = {
                    executor.submit(process_single_cluster, job_spec): job_spec
                    for job_spec in config.get("jobs", [])
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

            current_on_disk = None
            if os.path.exists(config_filename):
                with open(config_filename, "r") as f:
                    try:
                        current_on_disk = yaml.safe_load(f)
                    except Exception:
                        pass

            if current_on_disk != proxy_config:
                logger.info("Updating physical disk copy")
                with open(config_filename, "w") as f:
                    yaml.dump(proxy_config, f, default_flow_style=False)

                logger.info("Updated physical disk copy")

                try:
                    from litellm.proxy.proxy_server import llm_router
                    if llm_router is not None:
                        llm_router.set_model_list(active_models)
                        logger.info("🔄 [In-Memory Router Reloaded] Live routing table refreshed successfully.")
                    else:
                        logger.warning("⚠️ LiteLLM router instance is not fully initialized yet.")
                except Exception as reload_err:
                    logger.error(f"❌ Failed to dynamically update LiteLLM router memory: {reload_err}")

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
