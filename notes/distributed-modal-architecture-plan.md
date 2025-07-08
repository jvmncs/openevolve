# Distributed Modal Architecture Plan for OpenEvolve

Proposed Distributed Architecture for OpenEvolve on Modal
========================================================
Goal: keep every resource (GPU for LLM, CPU for evaluation, I/O for DB) close to 100 % utilisation while preserving the MAP-Elites semantics.  The design splits responsibilities into stateless, auto-scalable Modal Functions and a very thin stateful layer.

────────────────────────────────────────────────────────
## 1. High-level services
────────────────────────────────────────────────────────

```
┌──────────────────┐     ┌────────────────┐
│   LLM Service    │     │ Eval Workers   │
│  (GPU, vLLM)     │     │ (CPU + sandbox)│
└────────┬─────────┘     └────────┬───────┘
         │                           ▲
  batched /async calls               │
         ▼                           │
┌────────────────────────────────────────┐
│            Evolution Queue            │
└────────┬───────────────────────┬──────┘
         │                       │
   evolve_worker()          update_db()
         ▼                       ▲
┌────────┴────────┐     ┌────────┴────────┐
│  Controller Hub │────▶│ Program DB      │
│  (singleton)    │◀────┤ (Postgres/Vol)  │
└─────────────────┘     └─────────────────┘
```

**Legend:**
- Rectangles = Modal Functions/cls's  
- Thick black = Modal job queues (internal)  
- Thin arrows = plain async calls / HTTP

**Core ideas:**
1. Split the "monolithic loop" into *independent iterations* (`evolve_once`) that can run safely in parallel; queue them.
2. Push all heavy lifting (LLM call, diff parsing, evaluation, artefact upload) into Modal Functions that scale with Modal's `scale()` primitive.
3. Keep a single tiny process (§ 2.1) that owns the *authoritative* ProgramDatabase and hands out iteration jobs; it never blocks on GPU/CPU work.

────────────────────────────────────────────────────────
## 2. Components in detail
────────────────────────────────────────────────────────

### 2.1 Controller Hub (state keeper)

```python
@modal.cls(secrets=[…], volumes={"/db": modal.Volume.persisted("openevolve-db")})
class ControllerHub:
    def __enter__(self):
        # Loads /db/programs.sqlite → self.database  (existing ProgramDatabase)
        self.lock = asyncio.Lock()

    @modal.method(concurrency_limit=1)   # single-writer semantics
    async def dequeue_batch(self, n: int) -> list[int]:
        """Pick the next `n` iteration indices, return task descriptors."""
        async with self.lock:
            start = self.database.next_iter
            tasks = list(range(start, start+n))
            self.database.next_iter += n
            self.database.save("/db")    # cheap; sqlite WAL
        return tasks

    @modal.method(concurrency_limit=1)
    async def commit_child(self, child_dict: dict, artifacts: dict):
        """Write evaluation result, migrate, checkpoint etc."""
        async with self.lock:
            self.database.add(Program.from_dict(child_dict))
            if artifacts:
                self.database.store_artifacts(child_dict["id"], artifacts)
            # migrate / checkpoint logic unchanged
            self.database.maybe_checkpoint()
            self.database.save("/db")
```

- Only **two** RPCs (`dequeue_batch`, `commit_child`) → small contention window.  
- Uses Modal persistent Volume (`/db`) so every container sees the same DB file.

### 2.2 Evolution workers (stateless, many)

```python
stub = modal.Stub("openevolve-evolve")

# Pay-per-ms CPUs, cheap burst scaling
@stub.function(
    image=cpu_image,
    scale=modal.Scale(target_concurrency=100,  # desired steady-state
                      max_concurrency=400),    # autoscale ceiling
    timeout=900,
)
async def evolve_worker(task_idx: int):
    hub = ControllerHub()
    # 1. sample parent and inspirations
    parent, inspirations = await hub.sample_for_task.remote(task_idx)
    prompt = build_prompt(parent, inspirations, …)

    # 2. LLM call to GPU service
    llm_response = await llm_generate.remote(prompt)   # see § 2.3

    # 3. Patch / rewrite program
    child_code, change_summary = parse_llm_response(parent.code, llm_response)

    # 4. Schedule evaluation (separate queue)
    metrics, artifacts = await evaluate_program.remote(child_code)

    # 5. Commit back
    child_dict = make_program_dict(parent, child_code, metrics, change_summary)
    await hub.commit_child.remote(child_dict, artifacts)
```

`evolve_worker` itself is cheap (CPU-only); thousands can run in parallel.

### 2.3 LLM service (GPU)

Already exists: `openevolve/llm/modal.py` uses vLLM.  Wrap it in a Modal Function with batching:

```python
@stub.function(
    gpu="H100",
    scale=modal.Scale(target_concurrency=32, max_concurrency=128),
    concurrency_limits={"max_batch_size": 64},   # vLLM dynamic batching
)
async def llm_generate(prompt_pack):
    # prompt_pack = {system, user}
    return await vllm_client.chat(prompt_pack)
```

Because workers are *fire-and-forget*, Modal will autoscale GPU replicas until
• backlog≈0 **or** `max_concurrency` reached.

### 2.4 Evaluation workers (CPU + sandbox)

```python
@stub.function(
    image=cpu_image,   # can include docker-in-docker if sandbox needs it
    scale=modal.Scale(target_concurrency=64, max_concurrency=256),
    timeout=600,
)
async def evaluate_program(code: str) -> tuple[dict, dict]:
    sx = SandboxExecutor(SANDBOX_IMG, EVAL_FILE)   # 1-sec sandbox boot
    metrics = await sx.evaluate_program(code, uuid.uuid4().hex)
    artifacts = sx.get_pending_artifacts()   # may be None
    return metrics, (artifacts or {})
```

This uses Modal's "nested sandbox" — perfectly legal: a Modal Function may spawn `modal.Sandbox` objects; they run on fresh micro-VMs and are billed separately.  Because sandboxes boot in ~1 s and terminate automatically, CPU cost is minimal during idle.

### 2.5 Task production loop

A **tiny** cron job keeps the worker fleet fed:

```python
async def producer():
    hub = ControllerHub()
    while True:
        free = desired_buffer - in_flight    # back-pressure metric
        if free > 0:
            task_ids = await hub.dequeue_batch.remote(free)
            for t in task_ids:
                evolve_worker.spawn(t)       # returns immediately
        await asyncio.sleep(0.5)
```

Run `producer` either locally (for interactive use) or as another Modal Function with `schedule="* * * * *"` to keep it serverless.

────────────────────────────────────────────────────────
## 3. Concurrency & cost knobs
────────────────────────────────────────────────────────

| Component | Config field | Modal knob |
|-----------|--------------|------------|
| LLM | llm.parallel_generations | Scale.target_concurrency (GPU) |
| Eval | evaluator.parallel_evaluations | Scale.target_concurrency (CPU) |
| Ctrl | controller.task_buffer | producer.desired_buffer |
| Sandbox timeout | evaluator.timeout | function.timeout / Sandbox(timeout) |

Tune them independently; Modal will spin up/kill containers so you pay roughly:

```
cost ≈ Σ( container-seconds × price(container_type) )
```

────────────────────────────────────────────────────────
## 4. Failure / idempotency considerations
────────────────────────────────────────────────────────

- Every worker is **stateless**; if it crashes Modal retries it (at-least-once).  
- `commit_child` guarded by lock makes DB writes idempotent (ignore duplicate IDs).  
- Checkpointing is still deterministic (runs inside the lock).  
- If LLM/Eval throw `modal.RateLimitError` the worker catches, backs off, re-queues itself, or simply lets Modal retry.

────────────────────────────────────────────────────────
## 5. Migration path from current code
────────────────────────────────────────────────────────

1. Apply the refactor already sketched in `llm-parallelization-plan.md` to isolate `_evolve_once`.  
2. Move that coroutine into `evolve_worker` above; replace direct DB access with RPCs to `ControllerHub`.  
3. Replace local async vLLM client with the remote `llm_generate` call.  
4. Swap `Evaluator.evaluate_program` call for the remote `evaluate_program`.  
5. Delete the old sequential `for i in range…` loop and start the `producer`.

No change is needed to `SandboxExecutor`, `LLMEnsemble`, templates, metrics etc.—they run verbatim inside Modal containers.

────────────────────────────────────────────────────────
## 6. Expected throughput
────────────────────────────────────────────────────────

**Assume:**
- vLLM replica serves ~250 tok/s on H100, avg 200 tok request → 0.8 s latency.  
- `parallel_generations` = 32 → effective per-prompt latency ≈ 0.8 s / 32 ≈ 25 ms.  
- Evaluation (sandbox) takes 2 s, `parallel_evaluations` = 64 → 2 s / 64 ≈ 31 ms.  

So wall-clock per evolution step ≈ max(25 ms, 31 ms) ≈ 30 ms ⇒ **>30 iterations/sec sustained**, linear-scaling with knobs.

────────────────────────────────────────────────────────
## 7. Benefits
────────────────────────────────────────────────────────

- **Truly serverless**: zero cost at rest, millisecond-level horizontal scaling.  
- **Tight, independent back-pressure loops** for GPU and CPU; saturation is trivial to tune.  
- **Controller logic stays simple**; DB consistency guaranteed by single-writer pattern.  
- **Any future island-per-GPU or multi-model experiments** = just spin another worker class with different queue tag.

This architecture brings OpenEvolve to "infinite horizontal scale" on Modal with only a few hundred lines of additional glue code while re-using almost all existing business logic.
