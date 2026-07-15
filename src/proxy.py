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
import getpass

# Set up visible logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("slurm-proxy")

import litellm.proxy.proxy_server as proxy_server
from litellm.proxy.proxy_server import app


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


# Initialize the LiteLLM Router instance safely
router = Router(
    model_list=[],  # Starts empty, populated dynamically by the loop
    routing_strategy="least-busy",
    enable_pre_call_checks=True,
    cooldown_time=5
)

proxy_server.llm_router = router

# Keep track of when we last auto-scaled a model to prevent spamming sbatch
last_scale_time = {}

def check_and_scale_models():
    """Inspects router metrics and runs sbatch commands if wait times are too high."""
    config = load_scale_config()
    if not config:
        return

    metrics = getattr(router, "deployment_metrics", {})
    if not metrics:
        return

    for model_name, target_config in config.items():
        max_wait = target_config.get("max_wait_time_sec")
        sbatch_cmd = target_config.get("sbatch_command")

        if not max_wait or not sbatch_cmd:
            continue

        relevant_latencies = []
        for deployment_id, details in metrics.items():
            if model_name in deployment_id:
                resp_time = details.get("response_time", 0)
                if resp_time > 0:
                    relevant_latencies.append(resp_time)

        if not relevant_latencies:
            continue

        avg_wait_time = sum(relevant_latencies) / len(relevant_latencies)

        if avg_wait_time > max_wait:
            now = time.time()
            if now - last_scale_time.get(model_name, 0) > 300:
                logger.warning(
                    f"⚠️ [Scale Engine] Model '{model_name}' average wait time is {avg_wait_time:.2f}s "
                    f"(Threshold: {max_wait}s). Scaling up via Slurm..."
                )
                try:
                    subprocess.run(sbatch_cmd, shell=True, check=True)
                    logger.info(f"🚀 [Scale Engine] Successfully triggered: {sbatch_cmd}")
                    last_scale_time[model_name] = now
                except Exception as scale_err:
                    logger.error(f"❌ [Scale Engine] Failed executing scaling command for {model_name}: {scale_err}")


def get_active_slurm_nodes():
    """Finds running Ollama cluster endpoints via Slurm queue inspection."""
    active_endpoints = []
    try:
        current_user = getpass.getuser()
        logger.info(f"[Discovery Engine] Scanning Slurm jobs for user: {current_user}")

        cmd = ["squeue", "-u", current_user, "-t", "RUNNING", "-n", "ollama_server", "-o", "%B"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        
        # Parse hostnames ignoring table headers
        hostnames = [line.strip() for line in lines[1:] if line.strip() and not line.startswith("EXEC_HOST")]
        for host in hostnames:
            if host:
                active_endpoints.append(f"http://{host}:11434")
    except Exception as e:
        logger.warning(f"[Discovery Engine] Could not fetch endpoints via Slurm queue status: {e}")

    return list(set(active_endpoints))


def discover_models_from_endpoints(endpoints):
    """Queries endpoints to find active models and dynamically detects all backend capabilities."""
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

                # Initialize broad capability maps
                supports_vision = False
                supports_tools = False
                is_embedding = False

                # --- Query /api/show to analyze all capabilities ---
                try:
                    show_url = f"{url}/api/show"
                    show_data = json.dumps({"model": full_name}).encode("utf-8")
                    show_req = urllib.request.Request(show_url, data=show_data, method="POST")
                    show_req.add_header("Content-Type", "application/json")

                    with urllib.request.urlopen(show_req, timeout=2) as show_res:
                        info = json.loads(show_res.read().decode())
                        
                        # Extract capabilities and format parameters
                        details = info.get("details", {})
                        families = details.get("families", []) or [details.get("family")] if details.get("family") else []
                        capabilities = info.get("capabilities", []) or details.get("capabilities", []) or []

                        # 1. Vision Detection (clip architectures or vision tag arrays)
                        if "vision" in capabilities or "clip" in families or any("clip" in str(f).lower() for f in families):
                            supports_vision = True

                        # 2. Tool / Function Calling Detection (Ollama exposes tool-capable architectures)
                        # Native architectures supporting tools: llama3.1, mistral, command-r, qwen2, etc.
                        tool_friendly_families = ["llama", "mistral", "command-r", "qwen", "gemini", "gemma"]
                        if any(any(tf in str(f).lower() for tf in tool_friendly_families) for f in families):
                            supports_tools = True

                        # 3. Embedding Capability Detection
                        if "embedding" in capabilities or any("bert" in str(f).lower() for f in families):
                            is_embedding = True

                except Exception as show_err:
                    logger.warning(f"Could not fetch capability analysis matrix for {full_name}: {show_err}")

                # --- Assign LiteLLM Routing Parameters Map ---
                for model_identifier in set([full_name, base_name]):
                    litellm_params = {
                        "model": f"ollama/{full_name}",
                        "api_base": url,
                        "request_timeout": 600,
                        "keep_alive": "30m"
                    }

                    # Inject discovered flags cleanly into LiteLLM configurations
                    if supports_vision:
                        litellm_params["custom_llm_provider"] = "ollama"
                        litellm_params["supports_vision"] = True
                        logger.info(f"👁️ [Capability Flag] Vision enabled for {model_identifier} on {url}")

                    if supports_tools:
                        # Ensures LiteLLM knows this routing node can unpack openAI tool calls natively
                        litellm_params["supports_function_calling"] = True
                        logger.info(f"🛠️ [Capability Flag] Function Calling enabled for {model_identifier} on {url}")

                    if is_embedding:
                        # Flags it for alternative embedding router endpoints
                        litellm_params["model_type"] = "embedding"

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

    await server.serve()


if __name__ == "__main__":
    logger.info("Starting Auto-Discovering LiteLLM Gateway Server...")
    asyncio.run(main())
