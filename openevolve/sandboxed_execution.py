"""
Modal-based sandboxed execution for OpenEvolve.

This module provides infrastructure for running code evaluations in isolated
Modal sandboxes to ensure security and reproducibility.
"""

import asyncio
import io
import json
import logging
import os
import sys
import importlib.util
import tempfile
import functools
from pathlib import Path
from typing import Dict, Optional, Any

import modal

from .modal_app import app, evaluation_volume, database_volume

logger = logging.getLogger(__name__)


def detect_sandbox_config(example_dir: str) -> bool:
    """
    Check if an example directory has a sandbox configuration.

    Args:
        example_dir: Path to the example directory

    Returns:
        True if sandbox.py exists and exports sandbox_image
    """
    sandbox_path = Path(example_dir) / "sandbox.py"
    if not sandbox_path.exists():
        return False

    try:
        spec = importlib.util.spec_from_file_location("sandbox_config", sandbox_path)
        if spec is None or spec.loader is None:
            return False

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return hasattr(module, "sandbox_image")
    except Exception as e:
        logger.warning(f"Failed to load sandbox config from {sandbox_path}: {e}")
        return False


def load_sandbox_image(example_dir: str) -> "modal.Image":
    """
    Load the sandbox image from an example's sandbox.py file.

    Args:
        example_dir: Path to the example directory

    Returns:
        The Modal Image object defined in sandbox.py

    Raises:
        ImportError: If sandbox.py cannot be loaded or doesn't export sandbox_image
    """
    sandbox_path = Path(example_dir) / "sandbox.py"

    spec = importlib.util.spec_from_file_location("sandbox_config", sandbox_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load {sandbox_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "sandbox_image"):
        raise ImportError(f"{sandbox_path} does not export 'sandbox_image'")

    return module.sandbox_image  # type: ignore


@app.cls(
    volumes={"/eval": evaluation_volume},
    timeout=60 * 60,  # Keep alive for 1 hour idle
)
class SandboxExecutor:
    """
    Manages Modal sandbox creation and code execution for evaluations.
    """

    evaluation_file: str = modal.parameter()

    @modal.enter()
    def setup(self):
        """Initialize the sandbox executor with evaluation file."""
        self.volume = evaluation_volume
        self.sandbox_image = build_sandbox_image(self.evaluation_file)
        logger.info(
            f"Initialized SandboxExecutor with evaluation file: {self.evaluation_file}"
        )

    @modal.method()
    async def create_sandbox(self) -> "modal.Sandbox":
        """
        Create a new Modal sandbox with the configured image and security settings.

        Returns:
            A new Modal Sandbox instance
        """
        try:
            sandbox = await asyncio.to_thread(
                modal.Sandbox.create,
                app=app,
                image=self.sandbox_image,
                volumes={"/eval": self.volume},
                block_network=True,  # Block all network access for security
                cpu=0.25,  # Limit CPU to prevent resource exhaustion
                memory=512,  # Memory limit in MiB
                timeout=600,  # 10 minute timeout for long evaluations
                workdir="/workspace",  # Set working directory
            )
            logger.debug(f"Created sandbox {sandbox.object_id}")
            return sandbox
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            raise

    @modal.method()
    async def evaluate_in_sandbox(
        self, sandbox: "modal.Sandbox", program_code: str, program_id: str
    ) -> Dict[str, float]:
        """
        Run evaluation of a program in a sandbox.

        Args:
            sandbox: The Modal sandbox to use
            program_code: The program code to evaluate
            program_id: Unique identifier for the program

        Returns:
            Dictionary of metric names to scores
        """
        try:
            # The evaluation script is already in the volume at /eval/evaluator.py

            # Write the program to evaluate
            program_path = f"/tmp/program_{program_id}.py"
            with sandbox.open(program_path, "w") as f:
                f.write(program_code)

            # Create a runner script that imports and calls the evaluation
            runner_script = f"""
import sys
import json
import traceback

# Add eval directory to path
sys.path.insert(0, '/eval')

try:
    # Import the evaluator module
    from evaluator import evaluate

    # Run the evaluation
    result = evaluate('{program_path}')

    # Output the result as JSON
    print(json.dumps({{"success": True, "result": result}}))
except ImportError as e:
    print(json.dumps({{
        "success": False,
        "error": f"Failed to import evaluator: {{e}}",
        "traceback": traceback.format_exc()
    }}))
except Exception as e:
    print(json.dumps({{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}))
"""

            # Execute the runner script
            process = sandbox.exec("python", "-c", runner_script)

            # Collect output
            stdout = process.stdout.read()
            stderr = process.stderr.read()

            if stderr:
                logger.warning(f"Sandbox stderr for {program_id}: {stderr}")

            # Parse the result
            try:
                output = json.loads(stdout)
                if output["success"]:
                    result = output["result"]
                    # Ensure all values are floats
                    return {k: float(v) for k, v in result.items()}
                else:
                    logger.error(
                        f"Evaluation failed for {program_id}: {output['error']}"
                    )
                    logger.debug(f"Traceback: {output.get('traceback', 'N/A')}")
                    return {"error": 0.0}
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse evaluation output for {program_id}: {stdout}"
                )
                logger.error(f"Stderr: {stderr}")
                return {"error": 0.0}

        except Exception as e:
            logger.error(f"Sandbox evaluation failed for {program_id}: {e}")
            return {"error": 0.0}

    @modal.method()
    async def evaluate_program(
        self, program_code: str, program_id: str
    ) -> Dict[str, float]:
        """
        Create a sandbox, evaluate a program, and clean up.

        Args:
            program_code: The program code to evaluate
            program_id: Unique identifier for the program

        Returns:
            Dictionary of metric names to scores
        """
        sandbox = None
        try:
            sandbox = await self.create_sandbox.remote.aio()
            result = await self.evaluate_in_sandbox.remote.aio(
                sandbox, program_code, program_id
            )
            return result
        finally:
            if sandbox:
                try:
                    await asyncio.to_thread(sandbox.terminate)
                    logger.debug(f"Terminated sandbox {sandbox.object_id}")
                except Exception as e:
                    logger.warning(f"Failed to terminate sandbox: {e}")


def build_sandbox_image(evaluation_file: str) -> "modal.Image":
    """
    Build the appropriate sandbox image for the evaluation script.

    Args:
        evaluation_file: Path to the evaluation script

    Returns:
        A Modal Image configured for the evaluation
    """
    example_dir = Path(evaluation_file).parent

    # Check if example has custom sandbox configuration
    if detect_sandbox_config(example_dir):
        sandbox_image = load_sandbox_image(example_dir)
        logger.info(f"Using custom sandbox image from {example_dir}/sandbox.py")
    else:
        # Default image with common dependencies
        sandbox_image = (
            modal.Image.debian_slim()
            .run_commands("uv pip install --system pytest numpy")  # Add common evaluation dependencies
        )
        logger.info(f"Using default sandbox image for {evaluation_file}")

    return sandbox_image


def upload_evaluation_script(evaluation_file: str) -> "modal.Volume":
    """
    Upload evaluation script to a Modal volume once at startup.

    Args:
        evaluation_file: Path to the evaluation script

    Returns:
        Modal Volume containing the evaluation script
    """
    # Use the shared evaluation volume
    volume = evaluation_volume

    evaluation_script_content = Path(evaluation_file).read_text()

    try:
        with volume.batch_upload(force=True) as batch:
            # Upload the evaluation script as bytes
            batch.put_file(
                io.BytesIO(evaluation_script_content.encode()),
                "/evaluator.py",
            )

        logger.info(f"Uploaded evaluation script to volume from {evaluation_file}")
        return volume
    except Exception as e:
        logger.error(f"Failed to upload evaluation script to volume: {e}")
        raise
