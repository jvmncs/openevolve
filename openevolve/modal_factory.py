"""
Modal component factory for dynamic configuration.

This module provides a factory function that takes a Config object and
applies Modal decorators with the appropriate configuration values.
"""

import logging
import os
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from openevolve.config import Config

logger = logging.getLogger(__name__)

# Global flag to prevent double registration
_registered = False


def register_modal_components(config: "Config") -> None:
    """
    Register Modal components with configuration-driven decorators.
    
    This function must be called before app.deploy() to ensure all
    Modal functions and classes are properly configured.
    
    Args:
        config: OpenEvolve configuration object
    """
    global _registered
    
    if _registered:
        logger.debug("Modal components already registered, skipping")
        return
    
    _registered = True
    
    # Import the app and resources
    from openevolve.modal_app import (
        app,
        cpu_image,
        database_volume,
        evaluation_volume,
        inference_secret,
    )
    
    # Import the undecorated implementations
    from openevolve.modal_impl import (
        ControllerHub,
        evolve_worker,
        llm_generate,
        modal_evaluate,
    )
    
    logger.info("Registering Modal components with configuration")
    
    # Allow environment variable overrides for ops flexibility
    def get_config_value(env_var: str, config_value: any, value_type=int):
        """Get a config value with optional environment variable override."""
        env_value = os.environ.get(env_var)
        if env_value is not None:
            try:
                return value_type(env_value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid {env_var}={env_value}, using config value")
        return config_value
    
    # ----- ControllerHub -----
    hub_timeout = get_config_value("OE_HUB_TIMEOUT", config.modal.hub_timeout, int)
    hub_max_containers = get_config_value("OE_HUB_MAX_CONTAINERS", config.modal.hub_max_containers, int)
    hub_max_concurrent = get_config_value("OE_HUB_MAX_CONCURRENT", config.modal.hub_max_concurrent_requests, int)
    
    # Apply modal.concurrent first, then app.cls (which must be outermost)
    concurrent_hub = modal.concurrent(max_inputs=hub_max_concurrent)(ControllerHub)
    decorated_controller_hub = app.cls(
        image=cpu_image,
        volumes={config.modal.database_volume_path: database_volume},
        timeout=hub_timeout,
        max_containers=hub_max_containers,
    )(concurrent_hub)
    
    # ----- Evolution Worker -----
    worker_min_containers = get_config_value("OE_WORKER_MIN_CONTAINERS", config.modal.worker_min_containers, int)
    worker_max_containers = get_config_value("OE_WORKER_MAX_CONTAINERS", config.modal.worker_max_containers, int)
    worker_timeout = get_config_value("OE_WORKER_TIMEOUT", config.modal.worker_timeout, int)
    
    decorated_evolve_worker = app.function(
        image=cpu_image,
        min_containers=worker_min_containers,
        max_containers=worker_max_containers,
        timeout=worker_timeout,
        volumes={config.modal.evaluation_volume_path: evaluation_volume},
    )(evolve_worker)
    
    # ----- LLM Generation -----
    llm_min_containers = get_config_value("OE_LLM_MIN_CONTAINERS", config.modal.llm_min_containers, int)
    llm_max_containers = get_config_value("OE_LLM_MAX_CONTAINERS", config.modal.llm_max_containers, int)
    llm_timeout = get_config_value("OE_LLM_TIMEOUT", config.modal.llm_timeout, int)
    
    decorated_llm_generate = app.function(
        image=cpu_image,
        min_containers=llm_min_containers,
        max_containers=llm_max_containers,
        timeout=llm_timeout,
        secrets=[inference_secret],
    )(llm_generate)
    
    # ----- Evaluation -----
    eval_min_containers = get_config_value("OE_EVAL_MIN_CONTAINERS", config.modal.eval_min_containers, int)
    eval_max_containers = get_config_value("OE_EVAL_MAX_CONTAINERS", config.modal.eval_max_containers, int)
    eval_timeout = get_config_value("OE_EVAL_TIMEOUT", config.modal.eval_timeout, int)
    eval_retries = get_config_value("OE_EVAL_RETRIES", config.modal.eval_retries, int)
    eval_max_concurrent = get_config_value("OE_EVAL_MAX_CONCURRENT", config.modal.eval_max_concurrent_inputs, int)
    eval_target_concurrent = get_config_value("OE_EVAL_TARGET_CONCURRENT", config.modal.eval_target_concurrent_inputs, int)
    
    # Apply modal.concurrent first, then app.function (which must be outermost)
    concurrent_evaluate = modal.concurrent(
        max_inputs=eval_max_concurrent,
        target_inputs=eval_target_concurrent,
    )(modal_evaluate)
    
    decorated_modal_evaluate = app.function(
        image=cpu_image,
        min_containers=eval_min_containers,
        max_containers=eval_max_containers,
        timeout=eval_timeout,
        retries=eval_retries,
        volumes={config.modal.evaluation_volume_path: evaluation_volume},
    )(concurrent_evaluate)
    
    # Export names to module namespace so Modal can find them by name
    # Modal will use the original function/class names for lookup
    current_module = __import__(__name__)
    setattr(current_module, "ControllerHub", decorated_controller_hub)
    setattr(current_module, "evolve_worker", decorated_evolve_worker)
    setattr(current_module, "llm_generate", decorated_llm_generate)
    setattr(current_module, "modal_evaluate", decorated_modal_evaluate)
    
    # Also export to globals for good measure
    globals().update({
        "ControllerHub": decorated_controller_hub,
        "evolve_worker": decorated_evolve_worker,
        "llm_generate": decorated_llm_generate,
        "modal_evaluate": decorated_modal_evaluate,
    })
    
    logger.info(f"Successfully registered Modal components:")
    logger.info(f"  ControllerHub: timeout={hub_timeout}, max_containers={hub_max_containers}, max_concurrent={hub_max_concurrent}")
    logger.info(f"  evolve_worker: min={worker_min_containers}, max={worker_max_containers}, timeout={worker_timeout}")
    logger.info(f"  llm_generate: min={llm_min_containers}, max={llm_max_containers}, timeout={llm_timeout}")
    logger.info(f"  modal_evaluate: min={eval_min_containers}, max={eval_max_containers}, timeout={eval_timeout}, retries={eval_retries}")
    logger.info(f"  modal_evaluate concurrency: max={eval_max_concurrent}, target={eval_target_concurrent}")


def reset_registration():
    """Reset the registration flag (useful for testing)."""
    global _registered
    _registered = False
