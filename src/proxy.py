import os
import subprocess
import time
import asyncio
import logging
import json
import urllib.request
import yaml  # Added for config parsing
from litellm.router import Router
import uvicorn

# Set up visible logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("slurm-proxy")

import litellm.proxy.proxy_server as proxy_server
from litellm.proxy.proxy_server import app

# Initialize LiteLLM Router with an empty list
router = Router(
    model_list=[],
    routing_strategy="least-busy",
    enable_pre_call_checks=True,
    cooldown_time=5
)

proxy_server.llm_router = router

# Keep track of when we last auto-scaled a model to prevent spamming sbatch
last_scale_time = {}

def load_scale_config():
    """Loads model wait time configuration thresholds from config.yaml."""
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f).get("models", {})
    except Exception as e:
        logger.error(f"[Scale Engine] Failed to load config.yaml: {e}")
        return {}

def check_and_scale_models():
    """Inspects router metrics and runs sbatch commands if wait times are too high."""
    config = load_scale_config()
    if not config:
        return

    # Fetch router metrics
    # deployment_metrics holds request/error rates and latency details
    metrics = getattr(router, "deployment_metrics", {})
    if not metrics:
        return

    for model_name, target_config in config.items():
        max_wait = target_config.get("max_wait_time_sec")
        sbatch_cmd = target_config.get("sbatch_command")
        
        if not max_wait or not sbatch_cmd:
            continue

        # Look up metrics matching our model
        # LiteLLM tracks metric per deployment id, so we aggregate if there are multiple
        relevant_latencies = []
        for deployment_id, details in metrics.items():
            # Match if the deployment handles our model variant
            if model_name in deployment_id:
                # 'response_time' tracks moving average latency/wait times
                resp_time = details.get("response_time", 0)
                if resp_time > 0:
                    relevant_latencies.append(resp_time)

        if not relevant_latencies:
            continue

        avg_wait_time = sum(relevant_latencies) / len(relevant_latencies)
        
        if avg_wait_time > max_wait:
            now = time.time()
            # Cooldown check: Only spin up one cluster instance every 5 minutes per model
            if now - last_scale_time.get(model_name, 0) > 300:
                logger.warning(
                    f"⚠️ [Scale Engine] Model '{model_name}' average wait time is {avg_wait_time:.2f}s "
                    f"(Threshold: {max_wait}s). Scaling up via Slurm..."
                )
                try:
                    # Run sbatch configured shell command split safely
                    subprocess.run(sbatch_cmd, shell=True, check=True)
                    logger.info(f"🚀 [Scale Engine] Successfully triggered: {sbatch_cmd}")
                    last_scale_time[model_name] = now
                except Exception as scale_err:
                    logger.error(f"❌ [Scale Engine] Failed executing scaling command for {model_name}: {scale_err}")


def get_active_slurm_nodes():
    """Finds running Ollama cluster endpoints via Slurm or heartbeats."""
    active_endpoints = []
    try:
        cmd = ["squeue", "-t", "RUNNING", "-n", "ollama_server", "-o", "%B"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        hostnames = [line.strip() for line in lines[1:] if line.strip()]
        for host in hostnames:
            active_endpoints.append(f"http://{host}:11434")
    except Exception as e:
        logger.warning(f"[Discovery Engine] squeue lookup failed ({e}), trying fallback heartbeat folder...")
        shared_dir = os.path.expanduser("~/slurm_litellm/ollama_heartbeats")
        if os.path.exists(shared_dir):
            for filename in os.listdir(shared_dir):
                filepath = os.path.join(shared_dir, filename)
                if time.time() - os.path.getmtime(filepath) < 30:
                    try:
                        with open(filepath, 'r') as f:
                            endpoint = f.read().strip()
                            if endpoint:
                                active_endpoints.append(endpoint)
                    except Exception:
                        pass

    return list(set(active_endpoints))

def discover_models_from_endpoints(endpoints):
    discovered_model_list = []

    for url in endpoints:
        tags_url = f"{url}/api/tags"
        try:
            req = urllib.request.Request(tags_url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode())

            for model_info in data.get("models", []):
                full_name = model_info.get("name")
                base_name = full_name.split(":")[0]

                # --- Check explicit capabilities via /api/show ---
                supports_vision = False
                try:
                    show_url = f"{url}/api/show"
                    show_data = json.dumps({"model": full_name}).encode("utf-8")
                    show_req = urllib.request.Request(show_url, data=show_data, method="POST")
                    show_req.add_header("Content-Type", "application/json")

                    with urllib.request.urlopen(show_req, timeout=2) as show_res:
                        info = json.loads(show_res.read().decode())
                        capabilities = info.get("capabilities", [])
                        if "vision" in capabilities:
                            supports_vision = True
                except Exception as show_err:
                    logger.warning(f"Could not fetch capabilities for {full_name}: {show_err}")

                # --- Assign LiteLLM Routing Parameters ---
                for model_identifier in set([full_name, base_name]):
                    litellm_params = {
                        "model": f"ollama/{full_name}",
                        "api_base": url,
                        "request_timeout": 600,
                        "keep_alive": "30m"
                    }

                    if supports_vision:
                        litellm_params["custom_llm_provider"] = "ollama"
                        litellm_params["supports_vision"] = True
                        logger.info(f"👁️ [Vision Enabled] {model_identifier} on {url}")

                    discovered_model_list.append({
                        "model_name": model_identifier,
                        "litellm_params": litellm_params
                    })

        except Exception as e:
            logger.error(f"[Discovery Engine] Failed to fetch models from Ollama node {url}: {e}")

    return discovered_model_list


async def cluster_discovery_loop():
    """Background task tracking both Slurm instances and their active models."""
    logger.info("🚀 SUCCESS: Background Slurm/Ollama discovery loop has successfully started!")

    while True:
        logger.info("[Discovery Engine] Loop wake-up: Scanning cluster topology for active Slurm tasks...")
        try:
            loop = asyncio.get_running_loop()

            # 1. Discover the hardware endpoints
            endpoints = await loop.run_in_executor(None, get_active_slurm_nodes)
            logger.info(f"[Discovery Engine] Slurm check complete. Identified {len(endpoints)} active backend nodes.")

            # 2. Query those endpoints for active models
            updated_model_list = await loop.run_in_executor(None, discover_models_from_endpoints, endpoints)

            # 3. Hot-reload the router configurations
            router.set_model_list(updated_model_list)

            unique_models = list(set(m["model_name"] for m in updated_model_list))
            logger.info(f"[Discovery Engine] Sync complete. Configured {len(endpoints)} backends matching models: {unique_models}")

            # 4. Check internal LiteLLM wait time statistics and perform scaling
            await loop.run_in_executor(None, check_and_scale_models)

        except Exception as e:
            logger.error(f"[Discovery Engine] Error executing background cluster scan: {e}", exc_info=True)

        logger.info("[Discovery Engine] Scan iteration complete. Sleeping for 30 seconds...")
        await asyncio.sleep(30)

async def main():
    """Main execution orchestrator."""
    # 1. Fire up discovery task directly into the active async loop context
    asyncio.create_task(cluster_discovery_loop())

    # 2. Configure and run uvicorn programmatically inside the loop context
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)

    # This keeps the script running and serving requests asynchronously
    await server.serve()

if __name__ == "__main__":
    logger.info("Starting Auto-Discovering LiteLLM Gateway Server...")
    # Bootstrap the execution entry point cleanly
    asyncio.run(main())
