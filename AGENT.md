# OpenEvolve Development Guide

## Commands

**Test Commands:**
- `uv run python -m unittest discover -s tests -p "test_*.py"` - Run all tests
- `uv run python -m unittest tests.test_database` - Run single test module  
- `uv run python -m unittest tests.test_database.TestProgramDatabase.test_add_and_get` - Run single test

**Build/Lint Commands:**
- `uv run black openevolve tests scripts examples` - Format code with Black
- `uv sync` - Install dependencies and create virtual environment
- `uv run mypy openevolve` - Type checking

## Architecture

OpenEvolve is an evolutionary coding system with these core components:
- **Controller** (`controller.py`) - Main orchestration engine
- **Database** (`database.py`) - Program storage and retrieval with MAP-Elites archives
- **LLM Ensemble** (`llm/`) - Language model wrappers and ensemble logic
- **Evaluator** (`evaluator.py`) - Code execution and fitness evaluation with sandboxing
- **Prompt Sampler** (`prompt/`) - Context-rich prompt generation from program history

## Code Style

- **Formatting:** Black with 100 character line length
- **Imports:** isort with Black profile 
- **Types:** Required type hints (mypy strict mode)
- **Testing:** unittest framework with async support via pytest-asyncio
- **Error Handling:** Use EvaluationResult for evaluation errors with artifacts channel
