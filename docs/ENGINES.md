# Engines

All numbers here were measured on an M2 Max with 64 GB, one request at a time.

## llama.cpp (`engine: "llama.cpp"`)

Use mainline [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) built with Metal.
Recommended flags:

```
llama-server -m <model.gguf> --alias <id> --host 127.0.0.1 --port {port}
  -c <ctx> -ngl 999 --parallel 1 -b 2048 -ub 2048 -fa on
  --reasoning-format none --metrics
```

| Flag | Why |
|---|---|
| `--reasoning-format none` | **required.** The model emits raw think tags and the proxy is the single place reasoning is split. Letting both split produces doubled or empty reasoning. |
| `--parallel 1` | one slot gets the whole context; the hub is single-user |
| `-b 2048 -ub 2048` | large prefill batches; noticeably faster prompt processing on Apple GPUs |
| `-fa on` | flash attention; smaller KV and faster long context |
| `-ngl 999` | everything on the GPU |
| `--metrics` | Prometheus `/metrics` for external monitoring |
| `--reasoning-effort high` | optional per-model default when clients send none |

Notes:

- Some llama.cpp builds reject `response_format: json_schema` ("Failed to initialize
  samplers"). The proxy rewrites simple schemas to a GBNF grammar, so client title
  generation keeps working.
- Chat templates disagree on accepted `reasoning_effort` values. The proxy reads them live
  from `/props`.
- **Hybrid Mamba/attention MoE models** (Nemotron-3.5-Lightning-30B-A3B, `NEMOTRON_H_MOE`)
  run on mainline. Only the attention layers keep a KV cache, so 128K of context costs
  far less memory than on a dense transformer. Its Q8_0 file is 33.6 GB, needs about 39 GB
  resident at 128K, and decodes at about 65 tok/s (3B active parameters). Thinking is on by
  default; `reasoning_effort: none` turns it off. NVIDIA recommends temperature 1.0 and
  top_p 0.95.

## mtplx (`engine: "mtplx"`)

mtplx (Homebrew `mtplx`) is an MLX engine with multi-token-prediction
(MTP) speculative decoding. It drafts several tokens ahead and verifies them in one pass.

```
mtplx serve --model <repo-or-path> --port {port} --preserve-thinking off
            --tool-prompt-mode native --no-auth
```

- **`--no-auth` is required.** mtplx enables auth by default and the proxy would get 401s.
- It emits reasoning already split (`reasoning_content`), so the proxy passes it through.
- `reasoning_effort` maps to `xhigh` natively; `none` sends `reasoning: off`.
- Structured output is enforced natively, but only with `llguidance` installed in mtplx's
  venv. A Homebrew upgrade creates a **new** venv, so reinstall it after upgrading or
  `response_format` requests return 400.
- Its `prompt_per_second` divides *total* prompt tokens by the time spent on *new* tokens,
  which gives absurd rates on cache hits. The proxy recomputes prefill speed ([STATS.md](STATS.md)).
- The server runs as a child process that can outlive its launchd job. `llm_switch.py`
  kills such strays.
- Decode is about 2× llama.cpp on the same 27B weights for prose (34 vs 17 tok/s). Prefill
  is roughly equal.
- If a turn's thinking runs past `max_tokens` the answer comes back empty. Use realistic
  budgets (16K) or `--reasoning-effort medium`.

## TensorFold (`engine: "tensorfold"`)

An MLX engine with parallel speculative drafts from a small drafter model (DFlash2).

```
tensorfold serve <mlx-repo> --drafter auto --host 127.0.0.1 --port {port}
                 --context 65536 --name <id> --no-update-check
```

- Reports `tensorfold` telemetry (decode rate, prefill seconds, TTFT) and `speculative`
  stats (acceptance rate, tokens per round). The proxy surfaces both in `/stats`.
- Structured output is native with the `tensorfold[grammar]` extra installed.
- **Version note for M1–M4 Macs:** the native (Zig) 1.2.x builds cap prompt chunks at 128
  rows on chips without the M5 tensor units, and disable multi-checkpoint prompt caching
  when a drafter is attached. In our tests that meant about 110 tok/s prefill and about
  30 s time-to-first-token on every turn of a 3.4K-token agent prompt. The Python 0.6.x
  engine uses 2,048-token chunks and caches the full prompt, so pin 0.6.x on pre-M5
  hardware.

## Generic (`engine: "openai"`)

Any OpenAI-compatible server (oMLX, LM Studio, vLLM, …). Reasoning handling auto-detects
per turn: if `reasoning_content` deltas appear, they're passed through; otherwise think
tags in `content` are split. `response_format` gets the llama.cpp GBNF rewrite, so set
`engine` precisely if your server handles schemas natively.

## Memory sizing (one model at a time)

| Model class | Resident (approx.) |
|---|---|
| 7B Q4 | 5 GB + KV |
| 27B Q4 GGUF | 16–17 GB + KV |
| 27B MLX (mtplx/TensorFold) | 19–21 GB + paged KV |
| 36B-A4B MoE Q4 | 21 GB + ~191 KB/token (F16 KV) → ~45 GB at 128K |
| 30B-A3B hybrid Mamba MoE Q8 | ~39 GB at 128K |

By default macOS lets the GPU wire only part of RAM. Raise the limit for big contexts (see
[OPERATIONS.md](OPERATIONS.md#metal-wired-memory-limit)).
