# Stats: how the numbers are computed

Counting SSE deltas undercounts tokens, because one delta can carry several. Engines also
disagree on what "prompt tokens per second" means under prompt caching. So the proxy uses
the engine's own end-of-turn numbers as ground truth and the live counts only for the
in-progress display.

## Sources

| Engine | Final-turn data used |
|---|---|
| llama.cpp | `timings`: `prompt_n, prompt_ms, prompt_per_second, predicted_n, predicted_ms, predicted_per_second` |
| mtplx | llama-style `timings` + `usage` (+ `mtplx_progress` for token totals) |
| TensorFold | `tensorfold` (`tokens_per_second, seconds, prefill_seconds, time_to_first_token`) + `speculative` + `usage` |
| all | `usage.prompt_tokens`, `usage.prompt_tokens_details.cached_tokens`, `usage.completion_tokens` |

## Per-turn fields (`/stats.last`)

| Field | Formula |
|---|---|
| `prefill_tps` | **new (uncached) tokens ÷ prefill time** = `(prompt_tokens − cached_tokens) / prompt_ms`. A full cache hit gives `null` (nothing was prefilled). Values above 5,000 tok/s are dropped as unmeasurable. |
| `decode_tps` | the engine's authoritative generation rate |
| `think_tps`, `gen_tps` | the authoritative decode total split into thinking and answer: tokens by the live delta ratio, time by the proxy's wall clock per phase. A phase too short to time (< 8 tokens or < 0.5 s) reports `decode_tps`; `null` means that phase never happened. |
| `ttft` | request start → first token (or the engine's own TTFT when it reports one) |
| `prompt_tokens`, `predicted_tokens`, `reason_tokens`, `content_tokens`, `duration` | as named |

Thinking and answering are the same decode loop, so `think_tps` and `gen_tps` both land
near `decode_tps`. The split that matters for performance is **prefill vs decode**.

Live (in-progress) rates are wall-clock estimates. `tps()` requires at least 3 tokens and
0.1 s, and rejects anything above 1,000 tok/s, so bursty SSE arrivals can't show impossible
speeds.

## Session and all-time counters (router)

The router aggregates generated tokens across every model, so the totals survive swaps.

- **Generated tokens per turn** = max(`timings.predicted_n`, `usage.completion_tokens`,
  `mtplx_progress.completion_tokens`), parsed from the last 256 KB of the stream. All three
  are cumulative, so the max holds even if a large final stats chunk is cut off. Thinking
  tokens are included. Non-streaming responses are read from their JSON `usage`.
- **Session** resets when a new conversation starts, detected by a SHA-1 fingerprint of
  the first `user` message. The fingerprint stays constant as a chat grows and across an
  agent's tool-loop sub-requests. (A "≤ 2 messages means new chat" heuristic fails for
  agents that resend each turn's growing history.) Only the hash is kept, never the text.
- **Auxiliary calls** (`response_format` set, e.g. title generation) are not counted and
  never reset the session.
- **All-time** is written atomically to `state_dir/usage.json` after every turn and survives
  restarts.

Known limits:

- Client-side history compression can rewrite the first user message. That causes one
  spurious session reset.
- Sub-agents that start their own conversation count as a new session. All-time is
  unaffected.
