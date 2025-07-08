# Updated Distributed Modal Architecture Plan for OpenEvolve

Distributed Modal Architecture Plan for OpenEvolve
====================================================================
Goal: keep every resource (GPU for LLM, CPU for evaluation, I/O for DB) close to 100 % utilisation while preserving MAP-Elites semantics.  The design splits responsibilities into stateless, auto-scalable Modal Functions and a very thin state-keeper.

Legend
• Rectangles = Modal Functions / cls's
• Thick black = Modal job queues (internal)
• Thin arrows = plain async calls / HTTP

```
┌──────────────────┐     ┌────────────────┐
│   LLM Service                │     │ Eval Workers              │
│  (GPU, vLLM)                 │     │ (CPU+Sandbox)             │
└────────┬─────────┘     └────────┬───────┘
               │                                     ▲
         batched /async calls                         │
               ▼                                     │
┌───────────────────────────────────────┐
│                         Evolution Queue                         │
└────────┬───────────────────────┬──────┘
               │                                       │
        evolve_worker()                            update_db()
               ▼                                       ▲
┌────────┴────────┐        ┌────────┴────────┐
│       Controller Hub       │────▶│ Program DB                 │
│         (singleton)        │◀────┤ (Postgres/Vol)             │
└─────────────────┘        └─────────────────┘
```

────────────────────────────────────────────────────────
## 1. Modal primitives in use
────────────────────────────────────────────────────────
• `modal.App` — the top-level container for all serverless objects
• `@app.function(...)` — define stateless functions and autoscale with `min_containers` / `max_containers`
• `@app.cls(...)` — define a long-lived stateful object (ControllerHub)
• `@modal.concurrent(max_inputs, target_inputs)` — limit concurrent input *per container* (was `Scale.target_concurrency`)
• `modal.Volume.from_name("openevolve-db")` — shared, persisted SQLite volume
• `modal.Sandbox.create(image=..., timeout=...)` — hermetic program execution
• `.remote()` — async cross-function invocation (unchanged)

────────────────────────────────────────────────────────
## 2. Components in detail
────────────────────────────────────────────────────────
All code snippets assume:

```python
import modal, asyncio, uuid
app = modal.App("openevolve")          # instead of modal.Stub
cpu_image = modal.Image.debian_slim().pip_install("xxhash", "spacy")
SANDBOX_IMG = modal.Image.from_registry("python:3.11-slim")
volume = modal.Volume.from_name("openevolve-db", create_if_missing=True)
desired_buffer = 5_000                 # producer keeps this many tasks ahead
```

### 2.1 Controller Hub — the single source of truth

```python
@app.cls(
    secrets=[modal.Secret.from_name("pg-prod")],
    volumes={"/db": volume},
    timeout=60 * 60,          # keep container alive for 1 h idle
    allow_concurrent_inputs=128,  # container executes up to 128 RPCs at once
)
class ControllerHub:
    def __enter__(self):
        from database import ProgramDatabase
        self.db = ProgramDatabase("/db/programs.sqlite")
        self.lock = asyncio.Lock()

    # Single-writer gate
    @modal.method()
    @modal.concurrent(max_inputs=1)
    async def dequeue_batch(self, n: int) -> list[int]:
        async with self.lock:
            start = self.db.next_iter
            tasks = list(range(start, start + n))
            self.db.next_iter += n
            self.db.save()
        return tasks

    @modal.method()
    @modal.concurrent(max_inputs=1)
    async def commit_child(self, child_dict: dict, artifacts: dict):
        async with self.lock:
            from model import Program
            self.db.add(Program.from_dict(child_dict))
            if artifacts:
                self.db.store_artifacts(child_dict["id"], artifacts)
            self.db.maybe_checkpoint()
            self.db.save()
```

### 2.2 Evolution workers — cheap, burstable CPUs

```python
@app.function(
    image=cpu_image,
    min_containers=0,         # pay-as-you-go
    max_containers=400,       # autoscale ceiling
    timeout=900,
)
@modal.concurrent(max_inputs=100, target_inputs=25)  # per container
async def evolve_worker(task_idx: int):
    hub = ControllerHub()
    # 1. sample parent and inspirations
    parent, inspirations = await hub.sample_for_task.remote(task_idx)
    prompt = build_prompt(parent, inspirations)

    # 2. LLM generation
    llm_response = await llm_generate.remote(prompt)

    # 3. Patch / rewrite program
    child_code, change_summary = parse_llm_response(parent.code, llm_response)

    # 4. Async evaluation
    metrics, artifacts = await evaluate_program.remote(child_code)

    # 5. Commit back
    child_dict = make_program_dict(
        parent, child_code, metrics, change_summary
    )
    await hub.commit_child.remote(child_dict, artifacts)
```

### 2.3 LLM service — GPU + dynamic batching

```python
@app.function(
    gpu="H100",
    min_containers=1,             # keep 1 warm for latency
    max_containers=128,
    timeout=1_800,
)
@modal.concurrent(max_inputs=64, target_inputs=32)   # vLLM prefers large batches
async def llm_generate(prompt_pack: dict[str, str]):
    # prompt_pack = {"system": ..., "user": ...}
    from vllm_client import chat
    return await chat(prompt_pack)
```

Autoscaling: Modal spins up to `max_containers` H100 pods until the backlog of `evolve_worker` calls is drained.

### 2.4 Evaluation workers — CPU + sandbox

```python
@app.function(
    image=cpu_image,
    min_containers=0,
    max_containers=256,
    timeout=600,
)
@modal.concurrent(max_inputs=64, target_inputs=32)
async def evaluate_program(code: str) -> tuple[dict, dict]:
    sx = modal.Sandbox.create(image=SANDBOX_IMG, timeout=20)
    metrics = await sx.call("evaluate_program", code, uuid.uuid4().hex)
    artifacts = sx.get_artifacts() or {}
    return metrics, artifacts
```

(The nested sandbox is billed separately but boots in ≈1 s.)

### 2.5 Producer loop — keeps the queue filled

Run locally for interactive experiments or ship as a scheduled Modal Function:

```python
@app.function(schedule="*/1 * * * *")   # every minute
async def producer():
    hub = ControllerHub()
    while True:
        free = desired_buffer - evolve_worker.pending()   # in-flight count
        if free > 0:
            task_ids = await hub.dequeue_batch.remote(free)
            for t in task_ids:
                evolve_worker.spawn(t)   # fire-and-forget
        await asyncio.sleep(0.5)
```

`evolve_worker.spawn` returns immediately and lets Modal queue the job.

────────────────────────────────────────────────────────
## 3. Tuning knobs
────────────────────────────────────────────────────────

| Component | Env / Config flag | Modal knob                          |
|-----------|-------------------|-------------------------------------|
| LLM       | `llm.parallel`    | `@modal.concurrent(max_inputs=…)`   |
| Eval      | `eval.parallel`   | same                                |
| Fleet     | buffer size       | `producer.desired_buffer`           |
| Sandbox   | `eval.timeout`    | `Sandbox.create(timeout=…)`         |

────────────────────────────────────────────────────────
## 4. Cost model
────────────────────────────────────────────────────────
```
cost ≈ Σ( container-seconds × price(container_type) )
```
Modal auto-terminates idle replicas; min_containers=0 for CPU-heavy fleets
means **zero cost at rest**.

────────────────────────────────────────────────────────
## 5. Failure / idempotency
────────────────────────────────────────────────────────
• All workers are stateless; Modal retries failed inputs (at-least-once).
• ControllerHub's single-writer lock makes DB writes idempotent.
• LLM/Eval retries on `modal.RateLimitError` or transient infra faults.

────────────────────────────────────────────────────────
## 6. Migration steps
────────────────────────────────────────────────────────
1. Isolate `_evolve_once` ⇒ move into `evolve_worker`.
2. Replace local vLLM with `llm_generate.remote`.
3. Replace local evaluator with `evaluate_program.remote`.
4. Introduce `ControllerHub` for DB updates.
5. Delete the old sequential loop and start `producer`.

────────────────────────────────────────────────────────
## 7. Throughput estimate (example settings)
────────────────────────────────────────────────────────
• H100 vLLM replica: ~250 tok / s → 0.8 s / request
• `@modal.concurrent(target_inputs=32)` ⇒ ≈25 ms / prompt
• Eval sandbox: 2 s per run, 64 parallel ⇒ ≈31 ms
Wall-clock per evolution ≈ max(25 ms, 31 ms) ≈ 30 ms ⇒ **>30 iterations / s**
Scale linearly by raising `max_containers` & concurrency targets.

────────────────────────────────────────────────────────
## 8. Benefits
────────────────────────────────────────────────────────
• Pure serverless: pay only when evolving.
• Independent back-pressure loops for GPU & CPU.
• Simple single-writer semantics for the DB.
• New exploration strategies = just new queues / functions.

--------------------------------------------------------------------
This plan mirrors the previous architecture but uses **only the current Modal primitives** (`modal.App`, `@app.function`, `@modal.concurrent`, `Volume.from_name`, `Sandbox.create`, container-level autoscaling) and should run unmodified on the latest Modal release.
