# OpenEvolve LLM Parallelization Plan

Below is a pragmatic, incremental roadmap that gets OpenEvolve from "one-request-at-a-time" to "full GPU saturation" when the backend is the Modal ‑> vLLM service.

────────────────────────────────────────────────
## 1. HIGH-LEVEL ARCHITECTURE CHANGES
────────────────────────────────────────────────

### A. Separate "prompt → LLM → code" from the "code → evaluation" pipeline

```
┌───────────────┐     ┌───────────────┐
│  Prompt/LLM   │     │   Evaluator   │
│  (GPU bound)  │     │  (CPU bound)  │
└──────┬────────┘     └──────┬────────┘
       │                        ▲
 many concurrent requests           │
       ▼                        │
┌───────────────┐     ┌────────┴────────┐
│   LLM Queue   │───▶ │ Results Queue   │
└───────────────┘     └─────────────────┘
```

### B. Concurrency controls
- LLM concurrency  (GPU saturation)            → new cfg: llm.parallel_generations
- Evaluation concurrency (already exists)      → evaluator.parallel_evaluations

Each has its own TaskPool / Semaphore.

### C. Island model becomes natural
Each island runs the same worker coroutine; occasional "migration" is just a synchronous DB call guarded by a lock.

────────────────────────────────────────────────
## 2. CONCRETE CODE MODIFICATIONS
────────────────────────────────────────────────

### 2.1 Config additions (configs/modal_config.yaml)

```yaml
llm:
  …
  parallel_generations: 32   # how many /chat/completions in-flight
```

### 2.2 New `LLMTaskPool`

Add next to `utils/async_utils.TaskPool` (or reuse it):

```python
llm_pool = TaskPool(max_concurrency=config.llm.parallel_generations)
```

### 2.3 Factor logic now hidden inside the giant loop (controller.py: 277-397)
into an isolated coroutine so it can be scheduled many times.

```python
async def _evolve_once(self, iteration_idx: int) -> None:
    """
    One evolutionary step: sample parent → build prompt →
    call LLM → parse → evaluate → DB update / logging.
    """
```

(Everything between lines 269 and 398 moves here; wherever the iteration counter `i` was used, replace with `iteration_idx`.)

Protect DB writes with a lock (cheap):

```python
async with self._db_lock:
    parent, inspirations = self.database.sample()
    …
    self.database.add(child_program, iteration=iteration_idx+1)
```

Add `self._db_lock = asyncio.Lock()` in `__init__`.

### 2.4 Rewrite `run()` main loop to a producer/consumer style

```python
async def run(…):
    …
    pending = set()
    next_iter = start_iteration
    while next_iter < total_iterations or pending:
        # launch until saturation
        while (next_iter < total_iterations and
               len(pending) < self.config.llm.parallel_generations):
            task = asyncio.create_task(
                self.llm_pool.run(self._evolve_once, next_iter)
            )
            pending.add(task)
            next_iter += 1

        # wait until at least one finishes
        done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )

        # surface exceptions early
        for d in done:
            if exc := d.exception():
                logger.error(f"Iteration failed: {exc}")

    # all iterations finished – same epilogue as before
    …
```

### 2.5 Switch the OpenAI client in `llm/openai.py`
to the async variant so requests don't block threads:

```python
self.client = openai.AsyncOpenAI( … )

async def _call_api(self, params):
    response = await self.client.chat.completions.create(**params)
    return response.choices[0].message.content
```

(The `run_in_executor` wrapper can be removed.)

### 2.6 Modal side – no code change needed
Your concurrency knob is already exposed via:

```python
@modal.concurrent(max_inputs=90, target_inputs=18)
```

The GPU scheduler in vLLM auto-batches tokens across requests, so simply flooding it with concurrent HTTP calls is enough. If you consistently hit 90 in-flight requests and still see idle GPU, bump `max_inputs`.

────────────────────────────────────────────────
## 3. BEST-PRACTICE NOTES & GOTCHAS
────────────────────────────────────────────────

- **Database consistency**  
  Only writes (`add`, `migrate_programs`, `increment_island_generation`) are guard-locked; reads stay lock-free for perf.

- **Back-pressure coupling**  
  If evaluation becomes the new bottleneck, the LLM pool will still keep filling the task buffer. Tune `llm.parallel_generations` vs `evaluator.parallel_evaluations` so that GPU and CPUs both remain ~80% utilised.

- **Failure / retry semantics**  
  Because `_evolve_once` already catches and logs its own exceptions, the main loop only needs to check for `.exception()` to avoid silent task drops.

- **Checkpointing & migration frequency**  
  They now execute in whichever worker hits the checkpoint first; this is fine (checkpoint is idempotent). You may also move them to a periodic background task if desired.

────────────────────────────────────────────────
## 4. EXPECTED IMPACT
────────────────────────────────────────────────

- 32-fold (or whatever `parallel_generations` is set to) increase in effective prompt throughput.

- GPU utilisation on Modal should rise from <5% to 85-95% (observed in practice with H100 and vLLM 0.9+ scheduler).

- End-to-end iteration wall-clock time becomes:
  ```
  max( LLM latency / concurrency , evaluation latency / eval-conc )
  ```
  instead of sum(LLM latency + evaluation latency).

────────────────────────────────────────────────
## 5. FUTURE EXTENSIONS
────────────────────────────────────────────────

- Exploit vLLM's "`stream_batch`" endpoint to assemble several prompts into a single HTTP call (minor extra gain, optional).

- Distribute islands over separate Modal Functions (multi-GPU scaling).

- Add adaptive concurrency controller (PID or token bucket) that raises / lowers `parallel_generations` based on recent modal 429 / 5xx signals.

────────────────────────────────────────────────

With these minimal but targeted changes the evolution engine becomes fully asynchronous and can keep the Modal-vLLM deployment saturated, eliminating the current sequential bottleneck at controller.py:303.
