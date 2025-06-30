"""
Test Modal sandbox integration in OpenEvolve.
"""

import asyncio
import pytest
from pathlib import Path

from openevolve.config import EvaluatorConfig
from openevolve.evaluator import Evaluator
from openevolve.sandboxed_execution import (
    SandboxExecutor,
    detect_sandbox_config,
    load_sandbox_image,
)


EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "function_minimization"
EVALUATION_FILE = EXAMPLE_DIR / "evaluator.py"


@pytest.fixture
def test_program():
    """A simple test program for evaluation."""
    return """
import numpy as np

def search_algorithm(iterations=100, bounds=(-5, 5)):
    # Simple test implementation
    return 0.0, 0.0, 0.0

def evaluate_function(x, y):
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20

def run_search():
    x, y, value = search_algorithm()
    return x, y, value
"""


@pytest.fixture
def sandbox_image():
    """Load the sandbox image for testing."""
    return load_sandbox_image(str(EXAMPLE_DIR))


@pytest.fixture
def sandbox_executor(sandbox_image):
    """Create a SandboxExecutor for testing."""
    executor = SandboxExecutor(sandbox_image, str(EVALUATION_FILE))
    return executor


def test_sandbox_detection():
    """Test that sandbox configuration detection works."""
    assert detect_sandbox_config(str(EXAMPLE_DIR))

    # Test with non-existent directory
    assert not detect_sandbox_config("/tmp/nonexistent")


def test_load_sandbox_image():
    """Test loading sandbox image from example."""
    image = load_sandbox_image(str(EXAMPLE_DIR))
    assert image is not None


@pytest.mark.asyncio
async def test_sandbox_executor_single_evaluation(sandbox_executor, test_program):
    """Test evaluating a single program in a sandbox."""
    result = await sandbox_executor.evaluate_program(test_program, "test-1")

    assert isinstance(result, dict)
    assert len(result) > 0
    # Check that evaluation succeeded (not just error: 0.0)
    assert not (len(result) == 1 and result.get("error") == 0.0), (
        f"Evaluation failed with error result: {result}"
    )
    assert all(isinstance(v, (int, float)) for v in result.values())


@pytest.mark.asyncio
async def test_evaluator_with_sandbox():
    """Test Evaluator integration with sandbox execution."""
    config = EvaluatorConfig()
    config.use_sandboxed_execution = True
    config.cascade_evaluation = False
    config.parallel_evaluations = 4

    evaluator = Evaluator(config, str(EVALUATION_FILE))

    assert evaluator.sandbox_executor is not None

    # Test evaluating a program
    test_program = """
# EVOLVE-BLOCK-START
import numpy as np

def search_algorithm(iterations=500, bounds=(-5, 5)):
    best_x = -1.7  # Close to known optimum
    best_y = 0.68
    best_value = evaluate_function(best_x, best_y)

    for i in range(iterations):
        x = best_x + np.random.uniform(-0.1, 0.1)
        y = best_y + np.random.uniform(-0.1, 0.1)
        value = evaluate_function(x, y)

        if value < best_value:
            best_x, best_y, best_value = x, y, value

    return best_x, best_y, best_value
# EVOLVE-BLOCK-END

def evaluate_function(x, y):
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20

def run_search():
    x, y, value = search_algorithm()
    return x, y, value
"""

    result = await evaluator.evaluate_program(test_program, "test-evaluator-1")
    assert isinstance(result, dict)
    # For a successful evaluation, we should have metrics other than just 'error'
    assert len(result) > 0, f"Expected non-empty result, got: {result}"
    # If we only have error: 0.0, that indicates a failure
    assert not (len(result) == 1 and result.get("error") == 0.0), (
        f"Evaluation appears to have failed with only error metric: {result}"
    )


@pytest.mark.asyncio
async def test_evaluator_without_sandbox():
    """Test that Evaluator works without sandbox when not configured."""
    config = EvaluatorConfig()
    config.use_sandboxed_execution = False
    config.cascade_evaluation = False

    evaluator = Evaluator(config, str(EVALUATION_FILE))

    assert evaluator.sandbox_executor is None

    # Should still work with local execution
    test_program = """
import numpy as np

def search_algorithm(iterations=10, bounds=(-5, 5)):
    return 0.0, 0.0, 0.0

def evaluate_function(x, y):
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20

def run_search():
    x, y, value = search_algorithm()
    return x, y, value
"""

    result = await evaluator.evaluate_program(test_program, "test-no-sandbox-1")
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_evaluator_batch_with_sandbox():
    """Test batch evaluation with sandboxes."""
    config = EvaluatorConfig()
    config.use_sandboxed_execution = True
    config.cascade_evaluation = False

    evaluator = Evaluator(config, str(EVALUATION_FILE))

    programs = [
        (
            """
import numpy as np
def search_algorithm(iterations=10, bounds=(-5, 5)):
    return -1.7, 0.68, -1.5
def evaluate_function(x, y):
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
def run_search():
    return search_algorithm()
""",
            "batch-1",
        ),
        (
            """
import numpy as np
def search_algorithm(iterations=10, bounds=(-5, 5)):
    return 0.0, 0.0, 0.0
def evaluate_function(x, y):
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
def run_search():
    return search_algorithm()
""",
            "batch-2",
        ),
    ]

    results = await evaluator.evaluate_multiple(programs)

    assert len(results) == len(programs)
    for i, result in enumerate(results):
        assert isinstance(result, dict)
        assert len(result) > 0
        assert not (len(result) == 1 and result.get("error") == 0.0), (
            f"Evaluation {i} failed with error result: {result}"
        )
