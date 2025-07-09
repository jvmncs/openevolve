"""
Core evaluation engine for OpenEvolve - stateless evaluation logic
"""

import asyncio
import importlib.util
import json
import logging
import os
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import re

from openevolve.config import EvaluatorConfig
from openevolve.database import ProgramDatabase
from openevolve.evaluation_result import EvaluationResult
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.sandboxed_execution import SandboxExecutor
from openevolve.utils.format_utils import format_metrics_safe

logger = logging.getLogger(__name__)


async def evaluate_once(
    *,
    program_code: str,
    program_id: str,
    config: EvaluatorConfig,
    evaluation_file: str,
    llm_ensemble: Optional[LLMEnsemble] = None,
    prompt_sampler: Optional[PromptSampler] = None,
    database: Optional[ProgramDatabase] = None,
    sandbox_executor: Optional[SandboxExecutor] = None,
) -> EvaluationResult:
    """
    Execute exactly one evaluation pass.

    Args:
        program_code: Code to evaluate
        program_id: Optional ID for logging
        config: Evaluator configuration
        evaluation_file: Path to evaluation script
        llm_ensemble: Optional LLM ensemble for feedback
        prompt_sampler: Optional prompt sampler for LLM feedback
        database: Optional database for prompt logging
        sandbox_executor: Optional sandbox executor for sandboxed execution

    Returns:
        EvaluationResult with metrics and artifacts

    Raises:
        Exception: Any evaluation failure (for Modal to retry)
    """
    start_time = time.time()
    program_id_str = f" {program_id}" if program_id else ""

    # Check if artifacts are enabled
    artifacts_enabled = config.enable_artifacts

    # Create a temporary file for the program
    with tempfile.NamedTemporaryFile(suffix=config.temp_file_suffix, delete=False) as temp_file:
        temp_file.write(program_code.encode("utf-8"))
        temp_file_path = temp_file.name

    try:
        # Run evaluation
        if config.cascade_evaluation:
            # Run cascade evaluation
            result = await _cascade_evaluate(
                temp_file_path, config, evaluation_file, sandbox_executor
            )
        else:
            # Run direct evaluation
            result = await _direct_evaluate(
                temp_file_path, config, evaluation_file, sandbox_executor
            )

        # Process the result based on type
        eval_result = _process_evaluation_result(result)

        # Add LLM feedback if configured
        if config.use_llm_feedback and llm_ensemble and prompt_sampler:
            llm_result = await _llm_evaluate(
                program_code, llm_ensemble, prompt_sampler, database, program_id
            )
            llm_eval_result = _process_evaluation_result(llm_result)

            # Combine metrics
            for name, value in llm_eval_result.metrics.items():
                eval_result.metrics[f"llm_{name}"] = value * config.llm_feedback_weight

            # Merge artifacts
            if llm_eval_result.has_artifacts():
                eval_result.artifacts.update(llm_eval_result.artifacts)

        elapsed = time.time() - start_time
        logger.info(
            f"Evaluated program{program_id_str} in {elapsed:.2f}s: "
            f"{format_metrics_safe(eval_result.metrics)}"
        )

        return eval_result

    except asyncio.TimeoutError:
        # Handle timeout specially
        logger.warning(f"Evaluation timed out after {config.timeout}s")

        timeout_artifacts = {}
        if artifacts_enabled:
            timeout_artifacts = {
                config.artifact_keys["timeout"]: True,
                config.artifact_keys["timeout_duration"]: config.timeout,
                config.artifact_keys["failure_stage"]: "evaluation",
                config.artifact_keys["error_type"]: "timeout",
            }

        return EvaluationResult(
            metrics={"error": 0.0, "timeout": True}, artifacts=timeout_artifacts
        )

    except Exception as e:
        # Let other exceptions bubble up for caller to handle (Modal will retry, Evaluator will handle)
        logger.error(f"Evaluation failed for program{program_id_str}: {str(e)}")
        raise

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


async def _direct_evaluate(
    program_path: str,
    config: EvaluatorConfig,
    evaluation_file: str,
    sandbox_executor: Optional[SandboxExecutor] = None,
) -> Union[Dict[str, float], EvaluationResult]:
    """
    Directly evaluate a program using the evaluation function with timeout

    Args:
        program_path: Path to the program file
        config: Evaluator configuration
        evaluation_file: Path to evaluation script
        sandbox_executor: Optional sandbox executor

    Returns:
        Dictionary of metric name to score or EvaluationResult

    Raises:
        asyncio.TimeoutError: If evaluation exceeds timeout
        Exception: If evaluation function raises an exception
    """
    # If sandbox executor is available, use it
    if sandbox_executor:
        # Read the program code from the temporary file
        with open(program_path, "r") as f:
            program_code = f.read()

        # Generate a unique ID for this evaluation
        program_id = str(uuid.uuid4())

        # Use sandbox executor
        print("Evaluating program through SandboxExecutor...")
        logger.info("Evaluating program through SandboxExecutor...")
        return await sandbox_executor.evaluate_program.remote.aio(
            program_code, program_id
        )

    # Otherwise use the original local execution
    # Load evaluation function
    evaluation_function = _load_evaluation_function(evaluation_file)

    # Create a coroutine that runs the evaluation function in an executor
    async def run_evaluation():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, evaluation_function, program_path)

    # Run the evaluation with timeout - let exceptions bubble up
    result = await asyncio.wait_for(run_evaluation(), timeout=config.timeout)

    # Validate result
    if not isinstance(result, dict):
        logger.warning(f"Evaluation returned non-dictionary result: {result}")
        return {"error": 0.0}

    return result


async def _cascade_evaluate(
    program_path: str,
    config: EvaluatorConfig,
    evaluation_file: str,
    sandbox_executor: Optional[SandboxExecutor] = None,
) -> Union[Dict[str, float], EvaluationResult]:
    """
    Run cascade evaluation with increasingly challenging test cases

    Args:
        program_path: Path to the program file
        config: Evaluator configuration
        evaluation_file: Path to evaluation script
        sandbox_executor: Optional sandbox executor

    Returns:
        Dictionary of metrics or EvaluationResult with metrics and artifacts
    """
    # Import the evaluation module to get cascade functions if they exist
    try:
        # Add the evaluation file's directory to Python path
        eval_dir = os.path.dirname(os.path.abspath(evaluation_file))
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
            logger.debug(f"Added {eval_dir} to Python path for cascade evaluation")

        spec = importlib.util.spec_from_file_location(
            "evaluation_module", evaluation_file
        )
        if spec is None or spec.loader is None:
            return await _direct_evaluate(
                program_path, config, evaluation_file, sandbox_executor
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check if cascade functions exist
        if not hasattr(module, "evaluate_stage1"):
            return await _direct_evaluate(
                program_path, config, evaluation_file, sandbox_executor
            )

        # Run first stage with timeout
        try:
            # Use sandbox executor if available
            if sandbox_executor:
                # Read the program code for sandbox execution
                with open(program_path, "r") as f:
                    program_code = f.read()
                
                # Generate a unique ID for this evaluation
                import uuid
                stage1_program_id = str(uuid.uuid4())
                
                # Use sandbox executor with specific function
                stage1_result = await sandbox_executor.evaluate_program_with_function.remote.aio(
                    program_code, stage1_program_id, "evaluate_stage1"
                )
                stage1_eval_result = _process_evaluation_result(stage1_result)
            else:
                # Fall back to local execution
                async def run_stage1():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, module.evaluate_stage1, program_path
                    )

                stage1_result = await asyncio.wait_for(run_stage1(), timeout=config.timeout)
                stage1_eval_result = _process_evaluation_result(stage1_result)
        except asyncio.TimeoutError:
            logger.warning(f"Stage 1 evaluation timed out after {config.timeout}s")
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0, "timeout": True},
                artifacts={
                    config.artifact_keys["failure_stage"]: "stage1",
                    config.artifact_keys["timeout"]: True,
                },
            )
        except Exception as e:
            logger.error(f"Error in stage 1 evaluation: {str(e)}")
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0},
                artifacts={
                    config.artifact_keys["stderr"]: str(e),
                    config.artifact_keys["traceback"]: traceback.format_exc(),
                    config.artifact_keys["failure_stage"]: "stage1",
                },
            )

        # Check threshold
        if not _passes_threshold(
            stage1_eval_result.metrics, config.cascade_thresholds[0]
        ):
            return stage1_eval_result

        # Check if second stage exists
        if not hasattr(module, "evaluate_stage2"):
            return stage1_eval_result

        # Run second stage with timeout
        try:
            # Use sandbox executor if available
            if sandbox_executor:
                # Read the program code for sandbox execution
                with open(program_path, "r") as f:
                    program_code = f.read()
                
                # Generate a unique ID for this evaluation
                import uuid
                stage2_program_id = str(uuid.uuid4())
                
                # Use sandbox executor with specific function
                stage2_result = await sandbox_executor.evaluate_program_with_function.remote.aio(
                    program_code, stage2_program_id, "evaluate_stage2"
                )
                stage2_eval_result = _process_evaluation_result(stage2_result)
            else:
                # Fall back to local execution
                async def run_stage2():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, module.evaluate_stage2, program_path
                    )

                stage2_result = await asyncio.wait_for(run_stage2(), timeout=config.timeout)
                stage2_eval_result = _process_evaluation_result(stage2_result)
        except asyncio.TimeoutError:
            logger.warning(f"Stage 2 evaluation timed out after {config.timeout}s")
            stage1_eval_result.artifacts.update(
                {
                    config.artifact_keys["stage2_timeout"]: True,
                    config.artifact_keys["failure_stage"]: "stage2",
                }
            )
            stage1_eval_result.metrics["stage2_passed"] = 0.0
            stage1_eval_result.metrics["timeout"] = True
            return stage1_eval_result
        except Exception as e:
            logger.error(f"Error in stage 2 evaluation: {str(e)}")
            stage1_eval_result.artifacts.update(
                {
                    config.artifact_keys["stage2_stderr"]: str(e),
                    config.artifact_keys["stage2_traceback"]: traceback.format_exc(),
                    config.artifact_keys["failure_stage"]: "stage2",
                }
            )
            stage1_eval_result.metrics["stage2_passed"] = 0.0
            return stage1_eval_result

        # Merge results from stage 1 and 2
        merged_metrics = {}
        for name, value in stage1_eval_result.metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_metrics[name] = float(value)

        for name, value in stage2_eval_result.metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_metrics[name] = float(value)

        # Merge artifacts
        merged_artifacts = {}
        merged_artifacts.update(stage1_eval_result.artifacts)
        merged_artifacts.update(stage2_eval_result.artifacts)

        merged_result = EvaluationResult(
            metrics=merged_metrics, artifacts=merged_artifacts
        )

        # Check threshold for stage 3
        if len(config.cascade_thresholds) < 2 or not _passes_threshold(
            merged_result.metrics, config.cascade_thresholds[1]
        ):
            return merged_result

        # Check if third stage exists
        if not hasattr(module, "evaluate_stage3"):
            return merged_result

        # Run third stage with timeout
        try:
            # Use sandbox executor if available
            if sandbox_executor:
                # Read the program code for sandbox execution
                with open(program_path, "r") as f:
                    program_code = f.read()
                
                # Generate a unique ID for this evaluation
                import uuid
                stage3_program_id = str(uuid.uuid4())
                
                # Use sandbox executor with specific function
                stage3_result = await sandbox_executor.evaluate_program_with_function.remote.aio(
                    program_code, stage3_program_id, "evaluate_stage3"
                )
                stage3_eval_result = _process_evaluation_result(stage3_result)
            else:
                # Fall back to local execution
                async def run_stage3():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, module.evaluate_stage3, program_path
                    )

                stage3_result = await asyncio.wait_for(run_stage3(), timeout=config.timeout)
                stage3_eval_result = _process_evaluation_result(stage3_result)
        except asyncio.TimeoutError:
            logger.warning(f"Stage 3 evaluation timed out after {config.timeout}s")
            merged_result.artifacts.update(
                {
                    config.artifact_keys["stage3_timeout"]: True,
                    config.artifact_keys["failure_stage"]: "stage3",
                }
            )
            merged_result.metrics["stage3_passed"] = 0.0
            merged_result.metrics["timeout"] = True
            return merged_result
        except Exception as e:
            logger.error(f"Error in stage 3 evaluation: {str(e)}")
            merged_result.artifacts.update(
                {
                    config.artifact_keys["stage3_stderr"]: str(e),
                    config.artifact_keys["stage3_traceback"]: traceback.format_exc(),
                    config.artifact_keys["failure_stage"]: "stage3",
                }
            )
            merged_result.metrics["stage3_passed"] = 0.0
            return merged_result

        # Merge stage 3 results
        for name, value in stage3_eval_result.metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_result.metrics[name] = float(value)

        merged_result.artifacts.update(stage3_eval_result.artifacts)

        return merged_result

    except Exception as e:
        logger.error(f"Error in cascade evaluation: {str(e)}")
        return EvaluationResult(
            metrics={"stage1_passed": 0.0, "error": 0.0},
            artifacts={
                config.artifact_keys["stderr"]: str(e),
                config.artifact_keys["traceback"]: traceback.format_exc(),
                config.artifact_keys["failure_stage"]: "cascade_setup",
            },
        )


async def _llm_evaluate(
    program_code: str,
    llm_ensemble: LLMEnsemble,
    prompt_sampler: PromptSampler,
    database: Optional[ProgramDatabase] = None,
    program_id: str = "",
) -> Union[Dict[str, float], EvaluationResult]:
    """
    Use LLM to evaluate code quality

    Args:
        program_code: Code to evaluate
        llm_ensemble: LLM ensemble for generation
        prompt_sampler: Prompt sampler for building prompts
        database: Optional database for logging
        program_id: Optional ID for logging

    Returns:
        Dictionary of metric name to score or EvaluationResult
    """
    try:
        # Create prompt for LLM
        prompt = prompt_sampler.build_prompt(
            current_program=program_code, template_key=config.llm_evaluation_template
        )

        # Get LLM response
        responses = await llm_ensemble.generate_all_with_context(
            prompt["system"], [{"role": "user", "content": prompt["user"]}]
        )

        # Log prompt and response to database
        if database and program_id:
            database.log_prompt(
                program_id=program_id,
                template_key=config.llm_evaluation_template,
                prompt=prompt,
                responses=responses,
            )

        # Extract JSON from response
        try:
            # Try to find JSON block - use configured patterns
            json_patterns = config.json_extract_patterns

            artifacts = {}
            avg_metrics = {}
            for i, response in enumerate(responses):
                json_str = None
                
                # Try each configured pattern
                for pattern in json_patterns:
                    json_match = re.search(pattern, response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1) if json_match.groups() else json_match.group(0)
                        break
                
                if json_str is None:
                    # Fallback: try to extract JSON directly
                    json_str = response
                    # Remove non-JSON parts
                    start_idx = json_str.find("{")
                    end_idx = json_str.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = json_str[start_idx:end_idx]

                # Parse JSON
                result = json.loads(json_str)

                # All non-numeric values are artifacts, all numeric values are metrics
                metrics = {}
                for key, value in result.items():
                    if not isinstance(value, (int, float)):
                        artifacts[key] = value
                    else:
                        metrics[key] = float(value)

                # Weight of the model in the ensemble
                weight = llm_ensemble.weights[i] if llm_ensemble.weights else 1.0

                # Average the metrics
                for name, value in metrics.items():
                    if name in avg_metrics:
                        avg_metrics[name] += value * weight
                    else:
                        avg_metrics[name] = value * weight

            return EvaluationResult(
                metrics=avg_metrics,
                artifacts=artifacts,
            )

        except Exception as e:
            logger.warning(f"Error parsing LLM response: {str(e)}")
            return EvaluationResult(metrics={}, artifacts={})

    except Exception as e:
        logger.error(f"Error in LLM evaluation: {str(e)}")
        traceback.print_exc()
        return EvaluationResult(metrics={}, artifacts={})


def _process_evaluation_result(result: Any) -> EvaluationResult:
    """
    Process evaluation result to handle both dict and EvaluationResult returns

    Args:
        result: Raw result from evaluation function

    Returns:
        EvaluationResult instance
    """
    if isinstance(result, dict):
        # Backward compatibility - wrap dict in EvaluationResult
        return EvaluationResult.from_dict(result)
    elif isinstance(result, EvaluationResult):
        # New format - use directly
        return result
    else:
        # Error case - return error metrics
        logger.warning(f"Unexpected evaluation result type: {type(result)}")
        return EvaluationResult(metrics={"error": 0.0})


def _passes_threshold(metrics: Dict[str, float], threshold: float) -> bool:
    """
    Check if metrics pass a threshold

    Args:
        metrics: Dictionary of metric name to score
        threshold: Threshold to pass

    Returns:
        True if metrics pass threshold
    """
    if not metrics:
        return False

    # Calculate average score, skipping non-numeric values and 'error' key
    valid_metrics = []
    for name, value in metrics.items():
        # Skip 'error' keys and ensure values are numeric
        if name != "error" and isinstance(value, (int, float)):
            try:
                valid_metrics.append(float(value))
            except (TypeError, ValueError):
                logger.warning(f"Skipping non-numeric metric: {name}={value}")
                continue

    if not valid_metrics:
        return False

    avg_score = sum(valid_metrics) / len(valid_metrics)
    return avg_score >= threshold


def _load_evaluation_function(evaluation_file: str):
    """Load the evaluation function from the evaluation file"""
    if not os.path.exists(evaluation_file):
        raise ValueError(f"Evaluation file {evaluation_file} not found")

    try:
        # Add the evaluation file's directory to Python path
        eval_dir = os.path.dirname(os.path.abspath(evaluation_file))
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
            logger.debug(f"Added {eval_dir} to Python path for local imports")

        spec = importlib.util.spec_from_file_location(
            "evaluation_module", evaluation_file
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec from {evaluation_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["evaluation_module"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "evaluate"):
            raise AttributeError(
                f"Evaluation file {evaluation_file} does not contain an 'evaluate' function"
            )

        return module.evaluate
    except Exception as e:
        logger.error(f"Error loading evaluation function: {str(e)}")
        raise
