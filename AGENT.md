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

## Modal SDK Guidelines

### Core Concepts Used in OpenEvolve

**Apps and Functions:**
```python
import modal

app = modal.App("openevolve-distributed")

@app.function()
def my_function():
    pass

@app.cls()
class MyClass:
    @modal.enter()
    def setup(self):
        pass

    @modal.method()
    def my_method(self):
        pass
```

**Images:**
```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("package1", "package2")
    .run_commands("command1", "command2")
    .env({"VAR": "value"})
)
```

**Volumes:**
```python
volume = modal.Volume.from_name("volume-name", create_if_missing=True)

@app.function(volumes={"/mount/path": volume})
def use_volume():
    pass

# Volume batch upload
with volume.batch_upload(force=True) as batch:
    batch.put_file(io.BytesIO(data), "filename")
```

**Secrets:**
```python
secret = modal.Secret.from_name("secret-name")

@app.function(secrets=[secret])
def use_secret():
    import os
    value = os.environ["SECRET_KEY"]
```

**Sandboxes:**
```python
sandbox = modal.Sandbox.create(
    app=app,
    image=image,
    volumes={"/path": volume},
    timeout=600
)

process = sandbox.exec("python", "-c", "code")
stdout = process.stdout.read()
sandbox.terminate()
```

**GPU Support:**
```python
@app.function(gpu="A100")
def gpu_function():
    pass
```

**Web Servers:**
```python
@app.function()
@modal.web_server(port=8081)
def serve():
    pass
```

**Function Parameters:**
- `timeout` - Maximum execution time
- `max_containers` - Scaling limit
- `min_containers` - Keep warm containers
- `volumes` - Mount volumes
- `secrets` - Access secrets
- `gpu` - GPU type

**Calling Methods:**
- `.remote()` - Remote execution
- `.spawn()` - Background execution
- `.lookup()` - Find existing app
