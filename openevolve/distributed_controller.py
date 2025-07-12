"""
Distributed controller for OpenEvolve using Modal for auto-scaling

This implementation splits the evolution loop into distributed Modal Functions
while maintaining full compatibility with the original sequential Controller.

The Modal functions are now dynamically configured through the factory pattern
to support runtime configuration of container scaling, timeouts, and other parameters.
"""

import asyncio
import dataclasses
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import modal

from openevolve.config import Config, load_config
from openevolve.database import Program, ProgramDatabase
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.modal_app import (
    app,
    cpu_image,
    database_volume,
    evaluation_volume,
    inference_secret,
)
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import (
    apply_diff,
    extract_code_language,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)

# Modal app and shared resources are imported from modal_app module
# Modal functions are now dynamically configured in modal_factory.py

# ============================================================================
# Modal Functions - Distributed Components
#
# NOTE: The actual Modal functions and classes are now dynamically configured
# in modal_factory.py. This allows runtime configuration of container scaling,
# timeouts, and other parameters through the Config system.
#
# The implementations are in modal_impl.py and the factory applies decorators
# in register_modal_components().
# ============================================================================


# ============================================================================
# DistributedController - Main orchestrator with full feature parity
# ============================================================================


class DistributedController:
    """
    Distributed version of OpenEvolve controller with full feature parity.

    Maintains identical interface and semantics as the original Controller
    while implementing evolution using distributed Modal Functions.
    """

    def __init__(
        self,
        initial_program_path: str,
        evaluation_file: str,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        output_dir: Optional[str] = None,
        modal_app_name: str = "openevolve",
    ):
        # Load configuration (same as original)
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)

        # Set up output directory
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(initial_program_path), "openevolve_output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging (same as original)
        self._setup_logging()

        # Set random seed for reproducibility (same as original)
        if self.config.random_seed is not None:
            import hashlib
            import random

            import numpy as np

            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

            # Propagate to LLM configurations
            base_seed = str(self.config.random_seed).encode("utf-8")
            llm_seed = int(hashlib.md5(base_seed + b"llm").hexdigest()[:8], 16) % (
                2**31
            )
            self.config.llm.random_seed = llm_seed

            # Propagate seed to individual model configurations
            for model_cfg in self.config.llm.models:
                if (
                    not hasattr(model_cfg, "random_seed")
                    or model_cfg.random_seed is None
                ):
                    model_cfg.random_seed = llm_seed
            for model_cfg in self.config.llm.evaluator_models:
                if (
                    not hasattr(model_cfg, "random_seed")
                    or model_cfg.random_seed is None
                ):
                    model_cfg.random_seed = llm_seed

            logger.info(
                f"Set random seed to {self.config.random_seed} for reproducibility"
            )
            logger.debug(f"Generated LLM seed: {llm_seed}")

        # Load initial program
        self.initial_program_path = initial_program_path
        self.initial_program_code = self._load_initial_program()
        self.language = extract_code_language(self.initial_program_code)
        self.evaluation_file = evaluation_file

        # Upload evaluation script to Modal volume once
        from openevolve.sandboxed_execution import upload_evaluation_script

        self.evaluation_volume = upload_evaluation_script(evaluation_file)

        # Store the evaluation file separately since we need to pass it to workers
        self.evaluation_file_path = evaluation_file

        logger.info("Uploaded evaluation script to Modal volume")

        # Extract file extension
        self.file_extension = os.path.splitext(initial_program_path)[1] or ".py"

        # Modal app reference
        self.modal_app = app
        self.modal_app_name = modal_app_name

        # Deployed app and class references (will be set during deployment)
        self.deployed_app = None
        self.hub_cls = None

        # Producer state
        self.running = False
        self.producer_task = None

        logger.info(
            f"Initialized DistributedController with Modal app: {modal_app_name}"
        )

    def _setup_logging(self) -> None:
        """Set up logging (identical to original)"""
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))

        log_file = os.path.join(
            log_dir, f"openevolve_distributed_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(console_handler)

        logger.info(f"Logging to {log_file}")

    def _load_initial_program(self) -> str:
        """Load the initial program from file"""
        with open(self.initial_program_path, "r") as f:
            return f.read()

    async def _deploy_app(self):
        """Deploy the Modal app and setup lookups for classes and functions"""
        if self.deployed_app is not None:
            return  # Already deployed

        logger.info(f"Deploying Modal app: {self.modal_app_name}")

        # Deploy the app
        self.deployed_app = await self.modal_app.deploy.aio()

        # Setup lookups for deployed classes and functions
        self.hub_cls = modal.Cls.from_name("openevolve", "ControllerHub")
        self.evolve_worker_func = modal.Function.from_name(
            "openevolve", "evolve_worker"
        )
        self.llm_generate_func = modal.Function.from_name("openevolve", "llm_generate")
        self.modal_evaluate_func = modal.Function.from_name(
            "openevolve", "modal_evaluate"
        )

        logger.info("Modal app deployed and lookups configured")

    def _get_hub(self):
        """Get a ControllerHub instance from the deployed app"""
        if self.hub_cls is None:
            raise RuntimeError("App not deployed. Call _deploy_app() first.")
        return self.hub_cls()

    async def _initialize_database(self):
        """Initialize the distributed database with the initial program"""
        hub = self._get_hub()

        # Setup database with proper config
        import dataclasses

        await hub.setup_database.remote.aio(dataclasses.asdict(self.config.database))

        # Check if we need to add initial program
        stats = await hub.get_stats.remote.aio()

        if stats["num_programs"] == 0:
            logger.info("Adding initial program to distributed database")

            # Evaluate initial program
            initial_id = str(uuid.uuid4())
            metrics, artifacts = await self.modal_evaluate_func.remote.aio(
                self.initial_program_code,
                initial_id,
                "/eval/evaluator.py",
                dataclasses.asdict(self.config.evaluator),
            )

            # Create initial program
            initial_program = Program(
                id=initial_id,
                code=self.initial_program_code,
                language=self.language,
                parent_id=None,
                generation=0,
                metrics=metrics,
                metadata={"source": "initial"},
            )

            # Commit with proper prompt structure
            initial_prompt = {"system": "Initial program", "user": "Initial program"}
            await hub.commit_child.remote.aio(
                0,  # generation_idx
                0,  # candidate_idx
                initial_program,  # child_program
                initial_prompt,  # prompt
                "Initial program",  # llm_response
                artifacts,  # artifacts
            )

            logger.info(
                f"Added initial program with score: {metrics.get('score', 0):.4f}"
            )

    async def _producer_loop(self, max_generations: int):
        """
        Producer loop with proper generation management.

        Args:
            max_generations: Maximum number of generations to run
        """
        logger.info(f"Starting producer loop for {max_generations} generations")

        # Calculate population size from config
        pop_size = self.config.database.population_size
        # can't buffer more than pop_size in a single generation
        buffer_size = min(self.config.modal.worker_buffer_size, pop_size)

        for generation in range(max_generations):
            if not self.running:
                break

            try:
                hub = self._get_hub()

                # Start new generation
                await hub.start_generation.remote.aio(pop_size)
                logger.info(
                    f"Started generation {generation} with population size {pop_size}"
                )

                # Keep spawning workers until generation is complete
                while self.running:
                    stats = await hub.get_stats.remote.aio()

                    # Check if generation is complete
                    if stats["generation_complete"]:
                        logger.info(f"Generation {generation} completed")
                        break

                    # Calculate how many tasks to spawn
                    pending_tasks = stats["pending_tasks"]
                    tasks_scheduled = stats["tasks_scheduled"]
                    remaining_tasks = pop_size - tasks_scheduled

                    if remaining_tasks > 0:
                        # Spawn workers up to buffer size or remaining tasks
                        tasks_to_spawn = min(
                            buffer_size - pending_tasks, remaining_tasks
                        )

                        if tasks_to_spawn > 0:
                            for _ in range(tasks_to_spawn):
                                self.evolve_worker_func.spawn(
                                    self.config, "/eval/evaluator.py"
                                )
                                await hub.track_pending_task.remote.aio(1)

                            logger.debug(
                                f"Spawned {tasks_to_spawn} workers for generation {generation}"
                            )

                    # Wait before next check
                    await asyncio.sleep(self.config.modal.status_poll_interval)

                # Log progress
                stats = await hub.get_stats.remote.aio()
                logger.info(
                    f"Generation {generation} complete: {stats['num_programs']} total programs, "
                    f"best score: {stats['best_score']:.4f}"
                )

            except Exception as e:
                logger.error(f"Producer loop error in generation {generation}: {e}")
                await asyncio.sleep(5.0)

        logger.info("Producer loop completed")

    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
    ) -> Program:
        """
        Run the distributed evolution process.

        Maintains identical interface and semantics as the original Controller.

        Args:
            iterations: Maximum number of generations (maintains compatibility)
            target_score: Target score to reach (continues until reached if specified)
        """
        max_generations = iterations or self.config.max_iterations

        logger.info(f"Starting distributed evolution for {max_generations} generations")

        # Deploy the Modal app first
        await self._deploy_app()

        # Initialize database with initial program
        await self._initialize_database()

        # Start the producer loop
        self.running = True

        try:
            # Run producer in background
            self.producer_task = asyncio.create_task(
                self._producer_loop(max_generations)
            )

            # Monitor progress and target score
            hub = self._get_hub()

            while self.running:
                stats = await hub.get_stats.remote.aio()
                current_generation = stats["generation_idx"]
                current_best_score = stats["best_score"]
                generation_complete = stats["generation_complete"]

                # Check if target score reached
                if target_score is not None and current_best_score >= target_score:
                    logger.info(
                        f"Target score {target_score} reached! Best score: {current_best_score:.4f}"
                    )
                    break

                # Check if all generations completed
                if current_generation >= max_generations and generation_complete:
                    logger.info(f"All {max_generations} generations completed")
                    break

                # Wait and check again
                await asyncio.sleep(10.0)

        finally:
            # Clean shutdown
            self.running = False
            if self.producer_task:
                self.producer_task.cancel()
                try:
                    await self.producer_task
                except asyncio.CancelledError:
                    pass

        # Get final best program
        hub = self._get_hub()
        best_program = await hub.get_best_program.remote.aio()

        if best_program is None:
            logger.warning("No programs found, returning initial program")
            best_program = Program(
                id="initial",
                code=self.initial_program_code,
                language=self.language,
                parent_id=None,
                generation=0,
                metrics={"score": 0.0},
                metadata={"source": "initial"},
            )

        # Save best program to local filesystem (like original controller)
        await self._save_best_program_locally(best_program)

        logger.info(
            f"Distributed evolution completed. Best score: {best_program.metrics.get('score', 0):.4f}"
        )
        return best_program

    async def _save_best_program_locally(self, best_program: Program):
        """Save best program to local filesystem for easy access"""
        if not best_program:
            return

        # Create best program directory locally
        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Use the extension from the initial program file
        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(best_program.code)

        # Save complete program info including metrics
        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            import json
            import time

            json.dump(
                {
                    "id": best_program.id,
                    "generation": best_program.generation,
                    "iteration": best_program.iteration_found,
                    "timestamp": best_program.timestamp,
                    "parent_id": best_program.parent_id,
                    "metrics": best_program.metrics,
                    "language": best_program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(
            f"Saved best program to {code_path} with program info to {info_path}"
        )


# Alias for backwards compatibility
DistributedOpenEvolve = DistributedController
