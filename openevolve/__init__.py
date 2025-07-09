"""
OpenEvolve: An open-source implementation of AlphaEvolve
"""

__version__ = "0.1.0"

# Configure logging for Modal containers
import logging
import os
import sys

# Only configure logging if not already configured and we're in a Modal container
if not logging.getLogger().handlers and os.environ.get("MODAL_TASK_ID"):
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s  %(process)d  %(name)s  %(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # overwrite default root handler
    )
    
    # Silence noisy third-party logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Log that the logging setup is complete
    logger = logging.getLogger(__name__)
    logger.debug("Modal container logging configured via openevolve module")

from openevolve.controller import OpenEvolve

__all__ = ["OpenEvolve"]
