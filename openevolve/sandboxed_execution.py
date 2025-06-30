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
from pathlib import Path
from typing import Dict, Optional, Any

try:
    import modal

    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Modal is not installed. Sandbox execution will not be available.")

if MODAL_AVAILABLE:
    logger = logging.getLogger(__name__)


def detect_sandbox_config(example_dir: str) -> bool:
    """
    Check if an example directory has a sandbox configuration.

    Args:
        example_dir: Path to the example directory

    Returns:
        True if sandbox.py exists and exports sandbox_image
    """
    if not MODAL_AVAILABLE:
        return False
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
    if not MODAL_AVAILABLE:
        raise ImportError("Modal is not installed")
    sandbox_path = Path(example_dir) / "sandbox.py"

    spec = importlib.util.spec_from_file_location("sandbox_config", sandbox_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load {sandbox_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "sandbox_image"):
        raise ImportError(f"{sandbox_path} does not export 'sandbox_image'")

    return module.sandbox_image


class SandboxExecutor:
    """
    Manages Modal sandbox creation and code execution for evaluations.
    """

    def __init__(self, sandbox_image: "modal.Image", evaluation_file: str):
        """
        Initialize the sandbox executor.

        Args:
            sandbox_image: Modal Image to use for sandboxes
            evaluation_file: Path to the evaluation script
        """
        if not MODAL_AVAILABLE:
            raise ImportError("Modal is not installed")
        self.sandbox_image = sandbox_image
        self.evaluation_file = Path(evaluation_file)
        self.evaluation_script_content = self.evaluation_file.read_text()

        # Create a Modal app for this executor
        app_name = f"openevolve-sandbox-{self.evaluation_file.stem}"
        self.app = modal.App.lookup(app_name, create_if_missing=True)

        # Create or get the shared volume for evaluation scripts
        self.volume = modal.Volume.from_name(
            "openevolve-evaluation-scripts", create_if_missing=True
        )

        # Upload the evaluation script to the volume
        self._upload_evaluation_script()

        logger.info(
            f"Initialized SandboxExecutor with image and evaluation script {self.evaluation_file}"
        )

    def _upload_evaluation_script(self):
        """Upload the evaluation script to the shared volume."""
        try:
            with self.volume.batch_upload(force=True) as batch:
                # Upload the evaluation script as bytes
                batch.put_file(
                    io.BytesIO(self.evaluation_script_content.encode()),
                    "/evaluator.py",
                )

            logger.debug(
                f"Uploaded evaluation script to volume from {self.evaluation_file}"
            )
        except Exception as e:
            logger.error(f"Failed to upload evaluation script to volume: {e}")
            raise

    async def create_sandbox(self) -> "modal.Sandbox":
        """
        Create a new Modal sandbox with the configured image.

        Returns:
            A new Modal Sandbox instance
        """
        try:
            sandbox = await asyncio.to_thread(
                modal.Sandbox.create,
                app=self.app,
                image=self.sandbox_image,
                volumes={"/eval": self.volume},
                timeout=600,  # 10 minute timeout for long evaluations
            )
            logger.debug(f"Created sandbox {sandbox.object_id}")
            return sandbox
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            raise

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
            sandbox = await self.create_sandbox()
            result = await self.evaluate_in_sandbox(sandbox, program_code, program_id)
            return result
        finally:
            if sandbox:
                try:
                    await asyncio.to_thread(sandbox.terminate)
                    logger.debug(f"Terminated sandbox {sandbox.object_id}")
                except Exception as e:
                    logger.warning(f"Failed to terminate sandbox: {e}")
