"""
Distributed controller for OpenEvolve using Modal for auto-scaling

This implementation splits the evolution loop into distributed Modal Functions
while maintaining full compatibility with the original sequential Controller.
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
from openevolve.modal_app import app, cpu_image, database_volume, evaluation_volume, inference_secret
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

# ============================================================================
# Modal Functions - Distributed Components
# ============================================================================


@app.cls(
    image=cpu_image,
    volumes={"/db": database_volume},
    timeout=60 * 60,  # Keep alive for 1 hour idle
    max_containers=1,
)
@modal.concurrent(max_inputs=999)
class ControllerHub:
    """
    Centralized database management with single-writer semantics.
    Maintains the authoritative ProgramDatabase with full feature parity.
    """

    @modal.enter()
    def setup(self):
        """Initialize database from persistent volume with proper config"""
        from openevolve.config import DatabaseConfig

        # Create proper database config
        self.db_config = DatabaseConfig(
            db_path="/db/programs.sqlite",
            in_memory=False,
            population_size=1000,
            archive_size=100,
            num_islands=5,
            migration_interval=50,
            migration_rate=0.1,
            elite_selection_ratio=0.1,
            exploration_ratio=0.2,
            exploitation_ratio=0.7,
            feature_dimensions=["score", "complexity"],
            feature_bins=10,
        )

        self.db = ProgramDatabase(self.db_config)
        self.lock = asyncio.Lock()
        self.current_iteration = 0
        self.pending_tasks = 0

        logger.info("ControllerHub initialized with database")

    @modal.method()
    async def get_next_iteration(self) -> int:
        """Get the next sequential iteration number"""
        async with self.lock:
            self.current_iteration += 1
            return self.current_iteration

    @modal.method()
    async def sample_for_task(
        self, iteration_idx: int
    ) -> Tuple[Program, List[Program]]:
        """Sample parent and inspirations for evolution"""
        async with self.lock:
            parent, inspirations = self.db.sample()
            return parent, inspirations

    @modal.method()
    async def get_top_programs(self, n: int = 3) -> List[Program]:
        """Get top N programs from database"""
        async with self.lock:
            return self.db.get_top_programs(n)

    @modal.method()
    async def get_artifacts(self, program_id: str) -> Optional[Dict]:
        """Get artifacts for a program"""
        async with self.lock:
            return self.db.get_artifacts(program_id)

    @modal.method()
    async def commit_child(
        self,
        child_program: Program,
        iteration: int,
        prompt: Dict[str, str],
        llm_response: str,
        artifacts: Optional[Dict] = None,
    ):
        """Commit a child program to the database with full metadata"""
        async with self.lock:
            # Add to database with proper iteration tracking
            self.db.add(child_program, iteration=iteration)

            # Log prompts and responses
            template_key = (
                "diff_user" if "diff" in llm_response else "full_rewrite_user"
            )
            self.db.log_prompt(
                template_key=template_key,
                program_id=child_program.id,
                prompt=prompt,
                responses=[llm_response],
            )

            # Store artifacts if they exist
            if artifacts:
                self.db.store_artifacts(child_program.id, artifacts)

            # Handle island evolution
            self.db.increment_island_generation()

            # Check if migration should occur
            if self.db.should_migrate():
                logger.info(f"Performing migration at iteration {iteration}")
                self.db.migrate_programs()
                self.db.log_island_status()

            # Perform checkpointing
            if iteration % 50 == 0:  # Checkpoint every 50 iterations
                self.db.save(f"/db/checkpoint_{iteration}.db", iteration)
                logger.info(f"Checkpoint saved at iteration {iteration}")

    @modal.method()
    async def get_best_program(self) -> Optional[Program]:
        """Get the best program from the database"""
        async with self.lock:
            if not self.db.programs:
                return None
            return self.db.get_best_program()

    @modal.method()
    async def track_pending_task(self, delta: int = 1):
        """Track pending tasks for buffer management"""
        async with self.lock:
            self.pending_tasks += delta
            return self.pending_tasks

    @modal.method()
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        async with self.lock:
            return {
                "num_programs": len(self.db.programs),
                "current_iteration": self.current_iteration,
                "pending_tasks": self.pending_tasks,
                "best_score": (
                    self.db.get_best_program().metrics.get("score", 0)
                    if self.db.programs
                    else 0
                ),
                "current_island": getattr(self.db, "current_island", 0),
                "best_program_id": (
                    self.db.best_program_id
                    if hasattr(self.db, "best_program_id")
                    else None
                ),
            }


@app.function(
    image=cpu_image,
    min_containers=0,
    max_containers=400,
    timeout=900,
    volumes={"/eval": evaluation_volume},
)
async def evolve_worker(config: Config, evaluation_file: str):
    """
    Stateless evolution worker with full feature parity.
    Performs one complete evolution step: sample → LLM → parse → evaluate → commit.
    """
    try:
        # Get deployed function references
        hub_cls = modal.Cls.from_name("openevolve", "ControllerHub")
        llm_generate_func = modal.Function.from_name("openevolve", "llm_generate")
        modal_evaluate_func = modal.Function.from_name(
            "openevolve", "modal_evaluate"
        )

        hub = hub_cls()

        # 1. Get iteration number
        iteration_idx = await hub.get_next_iteration.remote.aio()

        logger.info(f"Starting evolution iteration {iteration_idx}")

        # 2. Sample parent and inspirations
        parent, inspirations = await hub.sample_for_task.remote.aio(iteration_idx)

        # 3. Get artifacts and top programs for context
        parent_artifacts = await hub.get_artifacts.remote.aio(parent.id)
        top_programs = await hub.get_top_programs.remote.aio(5)

        # 4. Build prompt using proper method
        prompt_sampler = PromptSampler(config.prompt)

        prompt = prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in top_programs[:3]],
            top_programs=[p.to_dict() for p in top_programs],
            inspirations=[p.to_dict() for p in inspirations],
            language=parent.language,
            evolution_round=iteration_idx,
            diff_based_evolution=config.diff_based_evolution,
            program_artifacts=parent_artifacts,
        )

        # 5. LLM generation
        llm_response = await llm_generate_func.remote.aio(prompt, config)

        # 6. Parse the response
        if config.diff_based_evolution:
            diff_blocks = extract_diffs(llm_response)
            if not diff_blocks:
                logger.warning(f"Iteration {iteration_idx}: No valid diffs found")
                return
            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            # Parse full rewrite with correct signature
            child_code = parse_full_rewrite(llm_response, parent.language)
            if not child_code:
                logger.warning(f"Iteration {iteration_idx}: No valid code found")
                return
            changes_summary = "Full rewrite"

        # 7. Check code length
        if len(child_code) > config.max_code_length:
            logger.warning(f"Iteration {iteration_idx}: Code exceeds maximum length")
            return

        # 8. Evaluate the child program
        child_id = str(uuid.uuid4())
        metrics, artifacts = await modal_evaluate_func.remote.aio(
            child_code, child_id, "/eval/evaluator.py", dataclasses.asdict(config.evaluator)
        )

        # 9. Create child program with full metadata
        child_program = Program(
            id=child_id,
            code=child_code,
            language=parent.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=metrics,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
            },
        )

        # 10. Commit back to database
        await hub.commit_child.remote.aio(
            child_program, iteration_idx, prompt, llm_response, artifacts
        )

        logger.info(
            f"Completed evolution iteration {iteration_idx}: score={metrics.get('score', 0):.4f}"
        )

    except Exception as e:
        logger.error(f"Evolution iteration failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Always decrement pending tasks
        try:
            await hub.track_pending_task.remote.aio(-1)
        except:
            pass


@app.function(
    image=cpu_image,
    min_containers=0,
    max_containers=50,  # CPU-only for HTTP calls
    timeout=60 * 10,
    secrets=[inference_secret],
)
async def llm_generate(prompt: Dict[str, str], config: Config) -> str:
    """
    LLM generation using existing deployed Modal vLLM endpoint.
    Makes HTTP calls to the deployed inference server.
    """
    vllm_secret = os.environ.get("VLLM_API_KEY")
    if vllm_secret is None:
        logger.warning(
            "VLLM_API_KEY secret not found in environment. "
            "Please ensure the inference_secret is properly configured in Modal "
            "and contains the VLLM_API_KEY for accessing the LLM endpoint."
        )
    else:
        os.environ["OPENAI_API_KEY"] = vllm_secret

    try:
        # Use existing LLM ensemble logic
        llm_ensemble = LLMEnsemble(config.llm.models)

        system_msg = prompt.get("system", "")
        user_msg = prompt.get("user", "")
        messages = [{"role": "user", "content": user_msg}]

        response = await llm_ensemble.generate_with_context(
            system_message=system_msg,
            messages=messages,
        )

        logger.info(f"Generated LLM response ({len(response)} chars)")
        return response

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        # Return a minimal fallback
        return "# Error generating code\npass"


@app.function(
    image=cpu_image,
    min_containers=0,
    max_containers=256,
    timeout=600,
    retries=3,
    volumes={"/eval": evaluation_volume},
)
@modal.concurrent(max_inputs=64, target_inputs=32)
async def modal_evaluate(
    code: str, 
    program_id: str, 
    evaluation_file: str,
    config_dict: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Modal-native wrapper around evaluation_engine.evaluate_once().
    
    Modal handles:
    - Container isolation
    - Exponential back-off retries
    - Throttling / autoscaling
    
    Args:
        code: The program code to evaluate
        program_id: Unique identifier for the program
        evaluation_file: Path to evaluation script
        config_dict: Serialized EvaluatorConfig
    
    Returns:
        Tuple of (metrics, artifacts)
    """
    logger.info(f"Evaluating program {program_id}")
    
    try:
        # Re-hydrate config & helpers inside the container
        from openevolve.config import EvaluatorConfig
        from openevolve.llm.ensemble import LLMEnsemble
        from openevolve.prompt.sampler import PromptSampler
        from openevolve.evaluation_engine import evaluate_once
        
        config = EvaluatorConfig(**config_dict)
        
        # Set up optional components
        llm_ensemble = None
        prompt_sampler = None
        if config.use_llm_feedback:
            llm_ensemble = LLMEnsemble(config.llm.models)
            prompt_sampler = PromptSampler(config.prompt)
        
        # Set up sandbox executor
        sandbox_executor = None
        if config.use_sandboxed_execution:
            executor_cls = modal.Cls.from_name("openevolve", "SandboxExecutor")
            sandbox_executor = executor_cls(evaluation_file=evaluation_file)
        
        # Use the volume-mounted evaluation file path
        volume_evaluation_file = "/eval/evaluator.py"
        
        # Run the evaluation using the engine
        result = await evaluate_once(
            program_code=code,
            program_id=program_id,
            config=config,
            evaluation_file=volume_evaluation_file,
            llm_ensemble=llm_ensemble,
            prompt_sampler=prompt_sampler,
            database=None,  # database writes happen in the hub
            sandbox_executor=sandbox_executor,
        )
        
        logger.info(
            f"Evaluation completed for {program_id}: score={result.metrics.get('score', 0):.4f}"
        )
        return result.metrics, result.artifacts
        
    except Exception as e:
        logger.error(f"Evaluation failed for {program_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error metrics with failure artifacts
        failure_artifacts = {
            "stderr": str(e),
            "traceback": traceback.format_exc(),
            "failure_stage": "evaluation",
        }
        
        # Return error metrics - Modal will retry if appropriate
        return {"score": 0.0, "error": str(e)}, failure_artifacts


# Backward compatibility alias
@app.function(
    image=cpu_image,
    min_containers=0,
    max_containers=256,
    timeout=600,
    volumes={"/eval": evaluation_volume},
)
@modal.concurrent(max_inputs=64, target_inputs=32)
async def evaluate_program(
    code: str, program_id: str, evaluation_file: str
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Backward compatibility wrapper for the old evaluate_program function.
    Uses basic evaluation without advanced features.
    """
    # Use basic config for backward compatibility
    from openevolve.config import EvaluatorConfig
    basic_config = EvaluatorConfig(
        use_sandboxed_execution=True,
        cascade_evaluation=False,
        use_llm_feedback=False,
    )
    
    return await modal_evaluate(code, program_id, "/eval/evaluator.py", dataclasses.asdict(basic_config))


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

            logger.info(
                f"Set random seed to {self.config.random_seed} for reproducibility"
            )

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
                initial_program, 0, initial_prompt, "Initial program", artifacts
            )

            logger.info(
                f"Added initial program with score: {metrics.get('score', 0):.4f}"
            )

    async def _producer_loop(self, max_iterations: int):
        """
        Producer loop with proper task tracking and buffer management.
        """
        logger.info(f"Starting producer loop for {max_iterations} iterations")

        while self.running:
            try:
                hub = self._get_hub()
                stats = await hub.get_stats.remote.aio()

                current_iteration = stats["current_iteration"]
                pending_tasks = stats["pending_tasks"]

                # Check completion conditions
                if current_iteration >= max_iterations:
                    if pending_tasks == 0:
                        logger.info("All iterations completed and tasks finished")
                        break
                    else:
                        logger.info(
                            f"Waiting for {pending_tasks} pending tasks to complete"
                        )
                        await asyncio.sleep(5.0)
                        continue

                # Calculate how many tasks to spawn
                buffer_size = 50  # Reasonable buffer to maintain throughput
                desired_pending = min(buffer_size, max_iterations - current_iteration)

                if pending_tasks < desired_pending:
                    tasks_to_spawn = desired_pending - pending_tasks

                    # Spawn evolution workers and track them
                    hub = self._get_hub()
                    for _ in range(tasks_to_spawn):
                        self.evolve_worker_func.spawn(self.config, "/eval/evaluator.py")
                        # Increment pending task count when we spawn
                        await hub.track_pending_task.remote.aio(1)

                    logger.info(f"Spawned {tasks_to_spawn} evolution workers")

                # Wait before next check
                await asyncio.sleep(2.0)

                # Log progress periodically
                if current_iteration % 10 == 0 and current_iteration > 0:
                    logger.info(
                        f"Progress: iteration {current_iteration}/{max_iterations}, "
                        f"pending: {pending_tasks}, best score: {stats['best_score']:.4f}"
                    )

            except Exception as e:
                logger.error(f"Producer loop error: {e}")
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
        """
        max_iterations = iterations or self.config.max_iterations

        logger.info(f"Starting distributed evolution for {max_iterations} iterations")

        # Deploy the Modal app first
        await self._deploy_app()

        # Initialize database with initial program
        await self._initialize_database()

        # Start the producer loop
        self.running = True

        try:
            # Run producer in background
            self.producer_task = asyncio.create_task(
                self._producer_loop(max_iterations)
            )

            # Monitor progress and target score
            hub = self._get_hub()

            while self.running:
                stats = await hub.get_stats.remote.aio()
                current_iteration = stats["current_iteration"]
                current_best_score = stats["best_score"]
                pending_tasks = stats["pending_tasks"]

                # Check if target score reached
                if target_score is not None and current_best_score >= target_score:
                    logger.info(
                        f"Target score {target_score} reached! Best score: {current_best_score:.4f}"
                    )
                    break

                # Check if all iterations completed
                if current_iteration >= max_iterations and pending_tasks == 0:
                    logger.info(f"All {max_iterations} iterations completed")
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

        logger.info(
            f"Distributed evolution completed. Best score: {best_program.metrics.get('score', 0):.4f}"
        )
        return best_program


# Alias for backwards compatibility
DistributedOpenEvolve = DistributedController
