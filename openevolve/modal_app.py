"""
Shared Modal app configuration for OpenEvolve distributed execution.

This module provides a centralized Modal app instance that can be used
by all distributed components to avoid circular dependencies and ensure
resource sharing.
"""

import modal

# Main Modal app for all OpenEvolve distributed functions
app = modal.App("openevolve")

# Shared CPU image for lightweight operations
cpu_image = (
    modal.Image.debian_slim()
    .run_commands("uv pip install --system asyncio uuid pathlib requests httpx pyyaml numpy openai")
    .add_local_python_source("openevolve", copy=True)
)

# Shared volume for evaluation scripts and database storage
evaluation_volume = modal.Volume.from_name(
    "openevolve-evaluation", create_if_missing=True
)
database_volume = modal.Volume.from_name("openevolve-db", create_if_missing=True)
