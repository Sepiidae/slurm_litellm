import argparse
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

# Caches for pulled models
PULLED_MODELS_CACHE = set()
PENDING_PULLS = set()
PENDING_PULLS_LOCK = threading.Lock()

# Persistent tracking of submitted jobs across all cycles: {job_id: {"job_name": str, "submitted_at": float}}
SUBMITTED_JOBS = {}
SUBMITTED_JOBS_LOCK = threading.Lock()

IMAGE_KEYWORDS = ("sdxl", "stable-diffusion", "flux", "dall-e", "midjourney", "imagen", "cascade", "diffusers", "sd3")


def parse_bool(val, default=False):
    """Safely converts string, bool, or int representations to a true boolean."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    return str(val).strip().lower() in ("true", "1", "yes", "on", "t")


def check_should_skip_pull(job_spec, global_no_pull=False):
    """Checks all possible YAML keys and CLI overrides for disabling model pulls."""
    if global_no_pull:
        return True

    no_pull_val = job_spec.get("no_pull", job_spec.get("no-pull", None))
    if no_pull_val is not None:
        if parse_bool(no_pull_val, default=False):
            return True

    pull_val = job_spec.get("pull_models", job_spec.get("pull-models", job_spec.get("pull", None)))
    if pull_val is not None:
        if not parse_bool(pull_val, default=True):
            return True

    return False


def get_api_base_url(node_name, port, job_spec):
    """
    Determines the appropriate api_base for LiteLLM based on backend type or explicit config.
    - Ollama default: http://host:port
    - LocalAI / OpenAI default: http://host:port/v1
    - Custom base_path override supported via YAML.
    """
    backend = str(job_spec.get("backend", "ollama")).strip().lower()
    custom_base_path = job_spec.get("base_path", job_spec.get("api_base_path", None))

    if custom_base_path is not None:
        base_path = str(custom_base_path).strip()
        if base_path and not base_path.startswith("/"):
            base_path = "/" + base_path
        return f"http://{node_name}:{port}{base_path}"

    if backend in ("localai", "openai", "v1"):
        return f"http://{node_name}:{port}/v1"
    else:
        # Default for ollama
        return f"http://{node_name}:{port}"


def parse_cli_args():
    """Parses command-line arguments for custom configuration."""
    parser = argparse.ArgumentParser(
        description="Slurm LiteLLM Local Orchestrator & Proxy."
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Port for LiteLLM Gateway Server"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="jobs_config.yaml",
        help="Path to jobs configuration YAML file"
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Globally disable model pulling across all clusters/jobs"
    )
    return parser.parse_args()


def load_config(config_path):
    """Loads configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file '{config_path}' not found.")
        return {"jobs": []}
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {"jobs": []}
    except Exception as e:
        logger.error(f"Error loading config '{config_path}': {e}")
        return {"jobs": []}


def get_user_slurm_jobs():
    """Queries squeue using pipe delimiters to safely extract Job ID, Name, State, and Node."""
    try:
        cmd = ["squeue", "--me", "-h", "-o", "%i|%j|%t|%N"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"❌ squeue query failed with exit code {result.returncode}: {result.stderr.strip()}")
            return None  # Return None to signal an error rather than empty list

        output = result.stdout.strip()
        if not output:
            return []

        jobs = []
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 4:
                jobs.append({
                    "job_id": parts[0].strip(),
                    "name": parts[1].strip(),
                    "state": parts[2].strip(),
                    "node": parts[3].strip() if parts[3].strip() not in ("(null)", "N/A", "") else None
                })
        return jobs
    except Exception as e:
        logger.error(f"Error querying squeue: {e}")
        return None  # Return None so orchestrator skips reconciliation


def launch_slurm_job(job_name, gres=None, mem=None, exclusive=False, script_path="sbatch.sh"):
    """Submits a new job to Slurm with optional resource configuration flags."""
    cmd = ["sbatch", f"--job-name={job_name}"]
    if gres:
        cmd.append(f"--gres={gres}")
    if mem:
        cmd.append(f"--mem={mem}")
    if exclusive:
        cmd.append("--exclusive")

    cmd.append(script_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to submit Slurm job '{job_name}': {result.stderr.strip()}")
        return None

    match = re.search(r"\d+", result.stdout)
    if match:
        job_id = match.group(0)
        logger.info(f"🚀 Successfully submitted Slurm job '{job_name}' with ID: {job_id} using {script_path}")
        return job_id
    return None


def _async_pull_model(raw_base_endpoint, model, job_name, node_name, cache_key):
    """Runs in a background thread to pull models asynchronously for Ollama endpoints."""
    start_time = time.time()
    try:
        logger.info(f"📥 [Pull Started] Pulling '{model}' on cluster '{job_name}' ({node_name})")
        pull_url = f"{raw_base_endpoint}/api/pull"
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


def process_single_cluster(job_spec, all_slurm_jobs, global_no_pull=False):
    """Non-blocking check for cluster jobs matching job_spec."""
    if all_slurm_jobs is None:
        logger.warning(f"Skipping cluster reconciliation for '{job_spec.get('job_name')}' due to squeue error.")
        return []

    job_name = job_spec["job_name"]
    models = job_spec["models"]
    target_count = job_spec.get("num_jobs", job_spec.get("count", 1))
    gres = job_spec.get("gres", None)
    mem = job_spec.get("mem", job_spec.get("memory", None))
    exclusive = parse_bool(job_spec.get("exclusive", False))
    team_id = job_spec.get("team_id", None)
    backend = str(job_spec.get("backend", "ollama")).strip().lower()

    # Read explicit mode if provided in job_spec
    explicit_mode = job_spec.get("mode", None)

    # Robust Bool/Key Parsing
    should_skip_pull = check_should_skip_pull(job_spec, global_no_pull)

    script_path = job_spec.get("sbatch_file", job_spec.get("script", "sbatch.sh"))

    # Match active Slurm jobs for this cluster specification
    matching_slurm_jobs = [j for j in all_slurm_jobs if j["name"] == job_name]
    squeue_job_ids = {j["job_id"] for j in matching_slurm_jobs}

    # Reconcile submitted registry against active squeue IDs
    with SUBMITTED_JOBS_LOCK:
        pending_in_flight = sum(
            1 for j_id, info in SUBMITTED_JOBS.items()
            if info["job_name"] == job_name and j_id not in squeue_job_ids
        )

    current_active_count = len(matching_slurm_jobs)
    effective_count = current_active_count + pending_in_flight

    if effective_count < target_count:
        needed = target_count - effective_count
        logger.info(
            f"🔍 Cluster '{job_name}': {current_active_count} active in squeue + "
            f"{pending_in_flight} submitted in-flight (Target: {target_count}). Submitting {needed} new job(s)..."
        )
        for _ in range(needed):
            new_job_id = launch_slurm_job(job_name, gres=gres, mem=mem, exclusive=exclusive, script_path=script_path)
            if new_job_id:
                with SUBMITTED_JOBS_LOCK:
                    SUBMITTED_JOBS[new_job_id] = {
                        "job_name": job_name,
                        "submitted_at": time.time()
                    }

    job_models = []
    for job in matching_slurm_jobs:
        job_id = job["job_id"]
        state = job["state"]
        node_name = job["node"]

        if state != "R" or not node_name or "CONFIGURING" in node_name:
            logger.info(f"⏳ Cluster '{job_name}' (Job ID: {job_id}) state is '{state}'. Waiting for execution node...")
            continue

        port = 11000 + (int(job_id) % 10000)

        # Determine specific api_base endpoint for LiteLLM router based on job config
        litellm_api_base = get_api_base_url(node_name, port, job_spec)
        raw_base_endpoint = f"http://{node_name}:{port}"

        # Async model downloads (Ollama API only)
        if backend == "ollama" and not should_skip_pull:
            for model in models:
                cache_key = f"{raw_base_endpoint}/{model}"
                if cache_key not in PULLED_MODELS_CACHE:
                    with PENDING_PULLS_LOCK:
                        if cache_key not in PENDING_PULLS:
                            PENDING_PULLS.add(cache_key)
                            pull_thread = threading.Thread(
                                target=_async_pull_model,
                                args=(raw_base_endpoint, model, job_name, node_name, cache_key),
                                daemon=True
                            )
                            pull_thread.start()
        elif should_skip_pull:
            logger.info(f"⏭️ [No-Pull Active] Skipping model pull for cluster '{job_name}' on {node_name}:{port}")

        # Build active route entries for LiteLLM
        for model in models:
            if backend in ("localai", "openai"):
                llm_type = "openai"
            else:
                llm_type = "ollama_chat" if "embed" not in model else "ollama"

            drop_params_flag = parse_bool(job_spec.get("drop_params", True))

            model_entry = {
                "model_name": model,
                "litellm_params": {
                    "model": f"{llm_type}/{model}",
                    "api_base": litellm_api_base,
                    "drop_params": drop_params_flag,
                    "api_key": "dummy-key",
                    "max_parallel_requests": 5,
                    "tool_choice": "none"
                }
            }

            # Build model_info block
            model_info = {}
            if team_id:
                model_info["team_id"] = team_id

            # Determine mode (image_generation vs chat/embedding)
            if explicit_mode:
                model_info["mode"] = explicit_mode
            elif any(k in model.lower() for k in IMAGE_KEYWORDS):
                model_info["mode"] = "image_generation"

            if model_info:
                model_entry["model_info"] = model_info

            job_models.append(model_entry)

    return job_models


def start_litellm_proxy(config_filename, port=8000):
    """Launches the LiteLLM Gateway Server in a background thread."""
    from litellm.proxy.proxy_cli import run_server

    logger.info(f"🚀 LiteLLM Router Proxy is booting on http://0.0.0.0:{port}")
    cli_args = [
        "--config", config_filename,
        "--host", "0.0.0.0",
        "--port", str(port)
    ]
    run_server(cli_args, standalone_mode=False)


def main():
    args = parse_cli_args()  # Defined safely at entrypoint
    config_filename = "dynamic_litellm_config.yaml"

    if args.no_pull:
        logger.info("🚫 Global --no-pull flag active: Skipping model pulls across all jobs.")

    if not os.path.exists(config_filename):
        with open(config_filename, "w") as f:
            yaml.dump({"model_list": []}, f)

    logger.info("--- Phase 1: Launching LiteLLM Gateway (Background) ---")
    proxy_thread = threading.Thread(
        target=start_litellm_proxy,
        args=(config_filename, args.port),
        daemon=True
    )
    proxy_thread.start()

    time.sleep(3)

    logger.info("--- Phase 2: Starting Orchestration Loop ---")

    while True:
        try:
            config = load_config(args.config)
            active_models = []

            # Single squeue query per loop cycle
            all_slurm_jobs = get_user_slurm_jobs()
            
            # Guard: If squeue failed, skip this cycle to avoid false job creation
            if all_slurm_jobs is None:
                logger.warning("⚠️ Slurm query failed. Skipping orchestration reconciliation for this cycle.")
                time.sleep(5)
                continue

            active_squeue_ids = {j["job_id"] for j in all_slurm_jobs}

            # Clean up tracking table
            now = time.time()
            with SUBMITTED_JOBS_LOCK:
                dead_ids = [
                    j_id for j_id, info in SUBMITTED_JOBS.items()
                    if j_id not in active_squeue_ids and (now - info["submitted_at"]) > 60
                ]
                for d_id in dead_ids:
                    SUBMITTED_JOBS.pop(d_id, None)

            # Process clusters concurrently
            with ThreadPoolExecutor(max_workers=max(1, len(config.get("jobs", [])))) as executor:
                futures = {
                    executor.submit(process_single_cluster, job_spec, all_slurm_jobs, args.no_pull): job_spec
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
                        m_info = entry.get("model_info", {})
                        logger.info(f"    🔹 Model: {m_name:<20} ➡️ Endpoint: {ep} (Info: {m_info})")
                else:
                    logger.warning("    ⚠️ No active backends are currently mapped.")

        except Exception as loop_err:
            logger.error(f"Error in main orchestrator loop: {loop_err}")

        time.sleep(5)


if __name__ == "__main__":
    main()
