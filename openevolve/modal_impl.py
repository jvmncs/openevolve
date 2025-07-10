"""
Undecorated implementations of Modal functions and classes.

This module contains the business logic without Modal decorators,
allowing the decorators to be applied dynamically with configuration.
"""

import asyncio
import dataclasses
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import modal

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import (
    apply_diff,
    extract_code_language,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)


@dataclass
class _GenerationState:
    """State tracking for a single generation in MAP-Elites"""

    generation_idx: int = 0
    pop_size: int = 0  # how many children we expect this generation
    tasks_scheduled: int = 0  # how many workers we gave a parent to
    tasks_committed: int = 0  # how many commits we have received
    frozen_population: List[Program] = field(default_factory=list)
    staging_children: List[Program] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Check if this generation is complete"""
        return self.tasks_committed >= self.pop_size

    def is_full(self) -> bool:
        """Check if this generation is full (no more tasks to schedule)"""
        return self.tasks_scheduled >= self.pop_size

    def reset_for_next_generation(self):
        """Reset state for the next generation"""
        self.generation_idx += 1
        # Mark previous generation as finished until `start_generation`
        # sets the new desired population size.
        self.pop_size = 0
        self.tasks_scheduled = 0
        self.tasks_committed = 0
        self.frozen_population.clear()
        self.staging_children.clear()


class ControllerHub:
    """
    Centralized database management with single-writer semantics.
    Maintains the authoritative ProgramDatabase with full feature parity.
    """

    @modal.enter()
    def setup(self):
        """Initialize database from persistent volume with proper config"""
        # Database config will be set via setup_database method
        self.db_config = None
        self.db = None
        self.lock = asyncio.Lock()
        self.generation_state = _GenerationState()
        self.pending_tasks = 0

        logger.info("ControllerHub initialized (database setup pending)")

    @modal.method()
    async def setup_database(self, config_dict: dict):
        """Setup database with proper configuration from the main process"""
        async with self.lock:
            if self.db is not None:
                return  # Already initialized

            from openevolve.config import DatabaseConfig

            # Create database config from provided config
            self.db_config = DatabaseConfig(**config_dict)
            # Ensure db_path is set to volume location
            self.db_config.db_path = "/db/programs.sqlite"

            self.db = ProgramDatabase(self.db_config)

            # Try to resume from checkpoint
            await self._try_resume_from_checkpoint()

            logger.info(f"Database setup complete with config: {self.db_config}")

    async def _try_resume_from_checkpoint(self):
        """Try to resume from the latest checkpoint"""
        import os
        import glob

        # Find the latest checkpoint file
        checkpoint_files = glob.glob("/db/checkpoint_gen_*.db")
        if not checkpoint_files:
            logger.info("No checkpoint found, starting fresh")
            return

        # Get the latest checkpoint
        latest_checkpoint = max(
            checkpoint_files, key=lambda f: int(f.split("_")[-1].split(".")[0])
        )
        latest_gen = int(latest_checkpoint.split("_")[-1].split(".")[0])

        logger.info(f"Found checkpoint for generation {latest_gen}")

        # Set generation state to resume from next generation
        self.generation_state.generation_idx = latest_gen + 1

        logger.info(f"Resuming from generation {self.generation_state.generation_idx}")

    @modal.method()
    async def start_generation(self, pop_size: int) -> int:
        """
        Start a new generation with the given population size.

        Args:
            pop_size: Number of children to produce in this generation

        Returns:
            The generation index that was started

        Raises:
            RuntimeError: If a generation is already running
        """
        async with self.lock:
            if (
                not self.generation_state.is_complete()
                and self.generation_state.pop_size > 0
            ):
                raise RuntimeError(
                    f"Generation {self.generation_state.generation_idx} is still running"
                )

            # Start new generation
            self.generation_state.pop_size = pop_size
            self.generation_state.tasks_scheduled = 0
            self.generation_state.tasks_committed = 0
            self.generation_state.staging_children.clear()

            # Create frozen snapshot of current population for sampling
            # Use the same sampling logic as the original controller
            all_programs = list(self.db.programs.values())
            self.generation_state.frozen_population = all_programs.copy()

            logger.info(
                f"Started generation {self.generation_state.generation_idx} with pop_size={pop_size}, frozen_population={len(self.generation_state.frozen_population)}"
            )

            return self.generation_state.generation_idx

    @modal.method()
    async def request_parent(self) -> Tuple[int, int, Optional[Program], List[Program]]:
        """
        Request a parent program for evolution.

        Returns:
            (generation_idx, candidate_idx, parent, inspirations)
            or (-1, -1, None, []) if generation is full
        """
        async with self.lock:
            if self.generation_state.is_full():
                return (-1, -1, None, [])

            # Sample from frozen population using existing logic
            if not self.generation_state.frozen_population:
                # No programs to sample from
                return (-1, -1, None, [])

            # Use the database's sampling logic but with frozen population
            # Temporarily set the programs to our frozen population
            original_programs = self.db.programs
            temp_programs = {p.id: p for p in self.generation_state.frozen_population}
            self.db.programs = temp_programs

            try:
                parent, inspirations = self.db.sample()
            finally:
                # Restore original programs
                self.db.programs = original_programs

            candidate_idx = self.generation_state.tasks_scheduled
            self.generation_state.tasks_scheduled += 1

            return (
                self.generation_state.generation_idx,
                candidate_idx,
                parent,
                inspirations,
            )

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
        generation_idx: int,
        candidate_idx: int,
        child_program: Program,
        prompt: Dict[str, str],
        llm_response: str,
        artifacts: Optional[Dict] = None,
    ):
        """Commit a child program to the database with full metadata"""
        async with self.lock:
            # Validate generation
            if generation_idx != self.generation_state.generation_idx:
                logger.warning(
                    f"Received commit for wrong generation: {generation_idx} != {self.generation_state.generation_idx}"
                )
                return

            # Add to staging (not main database yet)
            self.generation_state.staging_children.append(child_program)
            self.generation_state.tasks_committed += 1

            # Log prompts and responses (can be done immediately)
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

            logger.debug(
                f"Committed child {candidate_idx} for generation {generation_idx} ({self.generation_state.tasks_committed}/{self.generation_state.pop_size})"
            )

            # Check if generation is complete (barrier)
            if self.generation_state.is_complete():
                logger.info(
                    f"Generation {generation_idx} complete! Processing {len(self.generation_state.staging_children)} children"
                )

                # Process all staged children atomically
                current_island_counter = 0
                programs_per_island = max(
                    1,
                    self.generation_state.pop_size // (self.db_config.num_islands * 2),
                )

                for child in self.generation_state.staging_children:
                    # Add to database with proper iteration tracking
                    # Use generation_idx as iteration for compatibility
                    self.db.add(child, iteration=generation_idx)

                    # Handle island evolution for each child
                    self.db.increment_island_generation()

                    # Island rotation logic (match original controller)
                    current_island_counter += 1
                    if current_island_counter >= programs_per_island:
                        self.db.next_island()
                        current_island_counter = 0
                        logger.debug(f"Switched to island {self.db.current_island}")

                # Check if migration should occur (once per generation)
                if self.db.should_migrate():
                    logger.info(f"Performing migration at generation {generation_idx}")
                    self.db.migrate_programs()
                    self.db.log_island_status()

                # Perform checkpointing (once per generation)
                # Use configured interval
                checkpoint_interval = getattr(
                    self.db_config, "checkpoint_interval_generations", 10
                )
                if generation_idx % checkpoint_interval == 0:
                    self.db.save(
                        f"/db/checkpoint_gen_{generation_idx}.db", generation_idx
                    )
                    logger.info(f"Checkpoint saved at generation {generation_idx}")

                # Export best program periodically
                # Use configured interval
                export_interval = getattr(
                    self.db_config, "export_best_interval_generations", 5
                )
                if generation_idx % export_interval == 0:
                    await self._export_best_program(generation_idx)

                # Prepare for next generation
                self.generation_state.reset_for_next_generation()

                # Reset pending tasks counter for new generation
                self.pending_tasks = 0

                logger.info(
                    f"Generation {generation_idx} barrier complete, ready for generation {self.generation_state.generation_idx}"
                )

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
    async def report_failed_task(self, generation_idx: int):
        """
        Called by a worker that obtained a parent but exited
        without committing a child program.
        Frees the slot so the producer can schedule a replacement.
        """
        async with self.lock:
            # Ignore stray failures from old generations
            if generation_idx != self.generation_state.generation_idx:
                return
            # Never go below zero
            if self.generation_state.tasks_scheduled > 0:
                self.generation_state.tasks_scheduled -= 1

    async def _export_best_program(self, generation_idx: int):
        """Export best program to volume for easy access"""
        best_program = self.db.get_best_program()
        if not best_program:
            return

        # Create best program directory on volume
        import os

        os.makedirs("/db/best", exist_ok=True)

        # Write best program code
        with open("/db/best/best_program.py", "w") as f:
            f.write(best_program.code)

        # Write metadata
        import json
        import time

        with open("/db/best/best_program_info.json", "w") as f:
            json.dump(
                {
                    "id": best_program.id,
                    "generation": best_program.generation,
                    "iteration": best_program.iteration_found,
                    "current_generation": generation_idx,
                    "metrics": best_program.metrics,
                    "language": best_program.language,
                    "timestamp": best_program.timestamp,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"Exported best program at generation {generation_idx}")

    @modal.method()
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        async with self.lock:
            if self.db is None:
                return {
                    "num_programs": 0,
                    "generation_idx": 0,
                    "generation_complete": False,
                    "tasks_scheduled": 0,
                    "tasks_committed": 0,
                    "pop_size": 0,
                    "pending_tasks": 0,
                    "best_score": 0,
                    "current_island": 0,
                    "best_program_id": None,
                    "database_ready": False,
                }

            return {
                "num_programs": len(self.db.programs),
                "generation_idx": self.generation_state.generation_idx,
                "generation_complete": self.generation_state.is_complete(),
                "tasks_scheduled": self.generation_state.tasks_scheduled,
                "tasks_committed": self.generation_state.tasks_committed,
                "pop_size": self.generation_state.pop_size,
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
                "database_ready": True,
            }


async def evolve_worker(config: Config, evaluation_file: str):
    """
    Stateless evolution worker with full feature parity.
    Performs one complete evolution step: sample → LLM → parse → evaluate → commit.
    """
    got_parent = False        # did we get a parent?
    committed = False         # did we reach commit_child() successfully?
    generation_idx = -1       # generation index for failure reporting
    
    try:
        # Get deployed function references
        hub_cls = modal.Cls.from_name("openevolve", "ControllerHub")
        llm_generate_func = modal.Function.from_name("openevolve", "llm_generate")
        modal_evaluate_func = modal.Function.from_name("openevolve", "modal_evaluate")

        hub = hub_cls()

        # 1. Request parent and get generation/candidate indices
        (
            generation_idx,
            candidate_idx,
            parent,
            inspirations,
        ) = await hub.request_parent.remote.aio()

        # Check if generation is full
        if generation_idx < 0:
            logger.debug("Generation is full, worker returning")
            return

        got_parent = True     # from this point on a slot is reserved

        logger.info(
            f"Starting evolution generation {generation_idx}, candidate {candidate_idx}"
        )

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
            evolution_round=generation_idx,
            diff_based_evolution=config.diff_based_evolution,
            program_artifacts=parent_artifacts,
        )

        # 5. LLM generation
        llm_response = await llm_generate_func.remote.aio(prompt, config)

        # 6. Parse the response
        if config.diff_based_evolution:
            diff_blocks = extract_diffs(llm_response)
            if not diff_blocks:
                logger.warning(
                    f"Generation {generation_idx}, candidate {candidate_idx}: No valid diffs found"
                )
                return
            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            # Parse full rewrite with correct signature
            child_code = parse_full_rewrite(llm_response, parent.language)
            if not child_code:
                logger.warning(
                    f"Generation {generation_idx}, candidate {candidate_idx}: No valid code found"
                )
                return
            changes_summary = "Full rewrite"

        # 7. Check code length
        if len(child_code) > config.max_code_length:
            logger.warning(
                f"Generation {generation_idx}, candidate {candidate_idx}: Code exceeds maximum length"
            )
            return

        # 8. Evaluate the child program
        child_id = str(uuid.uuid4())
        metrics, artifacts = await modal_evaluate_func.remote.aio(
            child_code,
            child_id,
            "/eval/evaluator.py",
            dataclasses.asdict(config.evaluator),
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
            generation_idx,
            candidate_idx,
            child_program,
            prompt,
            llm_response,
            artifacts,
        )
        
        committed = True      # successful commit

        logger.info(
            f"Completed evolution generation {generation_idx}, candidate {candidate_idx}: score={metrics.get('score', 0):.4f}"
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
        
        # If we reserved a slot but never committed, tell the hub
        if got_parent and not committed:
            try:
                await hub.report_failed_task.remote.aio(generation_idx)
            except:
                pass


async def llm_generate(prompt: Dict[str, str], config: Config) -> str:
    """
    LLM generation using existing deployed Modal vLLM endpoint.
    Makes HTTP calls to the deployed inference server.
    """
    logger.info(f"llm_generate called with prompt keys: {list(prompt.keys())}")
    logger.info(f"config type: {type(config)}, has llm attr: {hasattr(config, 'llm')}")
    if hasattr(config, "llm"):
        logger.info(
            f"config.llm type: {type(config.llm)}, has models attr: {hasattr(config.llm, 'models')}"
        )
        if hasattr(config.llm, "models"):
            logger.info(
                f"config.llm.models length: {len(config.llm.models) if config.llm.models else 'None'}"
            )

    vllm_secret = os.environ.get("VLLM_API_KEY")
    if vllm_secret is None:
        logger.warning(
            "VLLM_API_KEY secret not found in environment. "
            "Please ensure the inference_secret is properly configured in Modal "
            "and contains the VLLM_API_KEY for accessing the LLM endpoint."
        )
    else:
        os.environ["OPENAI_API_KEY"] = vllm_secret
        logger.info("Set OPENAI_API_KEY from VLLM_API_KEY secret")

    try:
        # Use existing LLM ensemble logic
        logger.info(f"Initializing LLMEnsemble with {len(config.llm.models)} models")
        llm_ensemble = LLMEnsemble(config.llm.models)

        system_msg = prompt.get("system", "")
        user_msg = prompt.get("user", "")
        messages = [{"role": "user", "content": user_msg}]

        logger.info(
            f"Starting LLM generation with system_msg length: {len(system_msg)}, user_msg length: {len(user_msg)}"
        )
        response = await llm_ensemble.generate_with_context(
            system_message=system_msg,
            messages=messages,
        )

        logger.info(f"Generated LLM response ({len(response)} chars)")
        return response

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        import traceback

        logger.error(f"LLM generation traceback: {traceback.format_exc()}")
        # Return a minimal fallback
        return "# Error generating code\npass"


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
