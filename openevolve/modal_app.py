"""
Modal app configuration with import-time setup.

This module loads configuration and applies Modal decorators at import time,
which is required for Modal to properly register functions and classes.
"""

import logging
import os
from pathlib import Path

import modal

# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Loading
# ============================================================================


def load_modal_config():
    """Load Modal configuration from environment or default values."""

    # Try to load from environment variable pointing to config file
    config_path = os.getenv("OE_MODAL_CONFIG_PATH")
    if config_path and Path(config_path).exists():
        import yaml

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
            modal_config = config_data.get("modal", {})
            logger.info(f"Loaded Modal config from {config_path}")
            return modal_config

    # Fallback to environment variables with defaults
    return {
        "enabled": os.getenv("OE_MODAL_ENABLED", "true").lower() == "true",
        "app_name": os.getenv("OE_MODAL_APP_NAME", "openevolve"),
        "hub_timeout": int(os.getenv("OE_HUB_TIMEOUT", "3600")),
        "hub_max_containers": int(os.getenv("OE_HUB_MAX_CONTAINERS", "1")),
        "hub_max_concurrent_requests": int(os.getenv("OE_HUB_MAX_CONCURRENT", "999")),
        "worker_min_containers": int(os.getenv("OE_WORKER_MIN_CONTAINERS", "0")),
        "worker_max_containers": int(os.getenv("OE_WORKER_MAX_CONTAINERS", "400")),
        "worker_timeout": int(os.getenv("OE_WORKER_TIMEOUT", "900")),
        "llm_min_containers": int(os.getenv("OE_LLM_MIN_CONTAINERS", "0")),
        "llm_max_containers": int(os.getenv("OE_LLM_MAX_CONTAINERS", "50")),
        "llm_timeout": int(os.getenv("OE_LLM_TIMEOUT", "600")),
        "eval_min_containers": int(os.getenv("OE_EVAL_MIN_CONTAINERS", "0")),
        "eval_max_containers": int(os.getenv("OE_EVAL_MAX_CONTAINERS", "256")),
        "eval_timeout": int(os.getenv("OE_EVAL_TIMEOUT", "600")),
        "eval_retries": int(os.getenv("OE_EVAL_RETRIES", "3")),
        "eval_max_concurrent_inputs": int(os.getenv("OE_EVAL_MAX_CONCURRENT", "64")),
        "eval_target_concurrent_inputs": int(
            os.getenv("OE_EVAL_TARGET_CONCURRENT", "32")
        ),
        "database_volume_path": os.getenv("OE_DATABASE_VOLUME_PATH", "/db"),
        "evaluation_volume_path": os.getenv("OE_EVALUATION_VOLUME_PATH", "/eval"),
    }


# Load configuration at import time
modal_config = load_modal_config()

# ============================================================================
# Modal Resources
# ============================================================================

# Create the app
app = modal.App(modal_config["app_name"])

# Define the base image
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "apt-get update && apt-get install -y git",
        "pip install uv",
        "uv pip install --system openai anthropic httpx pyyaml numpy scipy",
    )
    .add_local_python_source("openevolve", copy=True)
)

# Define volumes
database_volume = modal.Volume.from_name(
    f"{modal_config['app_name']}-database", create_if_missing=True
)
evaluation_volume = modal.Volume.from_name(
    f"{modal_config['app_name']}-evaluation", create_if_missing=True
)

# Define secrets
inference_secret = modal.Secret.from_name("openevolve-vllm-secret")

# ============================================================================
# Import-Time Decorated Functions and Classes
# ============================================================================

# Import the implementations
from openevolve.modal_impl import (
    ControllerHub as ControllerHubImpl,
    evolve_worker as evolve_worker_impl,
    llm_generate as llm_generate_impl,
    modal_evaluate as modal_evaluate_impl,
)


# Apply decorators with configuration at import time
@app.cls(
    image=cpu_image,
    volumes={modal_config["database_volume_path"]: database_volume},
    timeout=modal_config["hub_timeout"],
    max_containers=modal_config["hub_max_containers"],
)
@modal.concurrent(max_inputs=modal_config["hub_max_concurrent_requests"])
class ControllerHub(ControllerHubImpl):
    pass


# For functions, apply decorators directly to the imported functions
evolve_worker = app.function(
    image=cpu_image,
    min_containers=modal_config["worker_min_containers"],
    max_containers=modal_config["worker_max_containers"],
    timeout=modal_config["worker_timeout"],
    volumes={modal_config["evaluation_volume_path"]: evaluation_volume},
)(evolve_worker_impl)

llm_generate = app.function(
    image=cpu_image,
    min_containers=modal_config["llm_min_containers"],
    max_containers=modal_config["llm_max_containers"],
    timeout=modal_config["llm_timeout"],
    secrets=[inference_secret],
)(llm_generate_impl)

modal_evaluate = app.function(
    image=cpu_image,
    min_containers=modal_config["eval_min_containers"],
    max_containers=modal_config["eval_max_containers"],
    timeout=modal_config["eval_timeout"],
    retries=modal_config["eval_retries"],
    volumes={modal_config["evaluation_volume_path"]: evaluation_volume},
)(
    modal.concurrent(
        max_inputs=modal_config["eval_max_concurrent_inputs"],
        target_inputs=modal_config["eval_target_concurrent_inputs"],
    )(modal_evaluate_impl)
)

# Export for lookup compatibility
__all__ = [
    "app",
    "ControllerHub",
    "evolve_worker",
    "llm_generate",
    "modal_evaluate",
    "cpu_image",
    "database_volume",
    "evaluation_volume",
    "inference_secret",
]
