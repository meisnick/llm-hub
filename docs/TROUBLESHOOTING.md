# Troubleshooting

Each entry here is a failure we actually hit while running the hub.

### Every request has a ~15 s delay, and the model seems to restart each time
Two engine servers were listening at once (a stray child process survived its launchd
job). Active-model detection takes the first `up` proxy, the request was for the other
model, and the router switched on every request. `llm_switch.py` now kills anything left
on other models' server ports. Check with `llm_switch.py --list`; only one port should be
listening.

### Short replies arrive all at once; long ones arrive in bursts
A forwarding loop used `resp.read(4096)`, which waits for 4 KB. The router uses `read1()`.
If you add your own proxy in front, make sure it streams (no response buffering, e.g.
`proxy_buffering off` in nginx).

### Empty answer, reasoning has content
- The `max_tokens` budget ran out while the model was still thinking
  (`finish_reason: length`). Raise `max_tokens` or lower `reasoning_effort`.
- Pre-split engines: if the answer shows up as reasoning, check `engine` in `hub.json`.
  The proxy must know the engine pre-splits, or it re-tags the answer.

### Empty reply / HTTP 500 mentioning Jinja or "reasoning_effort"
The chat template rejected the effort value. The proxy maps values using the template's
own "Supported values" list from `/props`. A template without such a list gets the
normalized value unchanged. Check the proxy log line `[reasoning] mapped … -> …`.

### `response_format` returns 400
- llama.cpp: the schema was too complex for the built-in GBNF conversion (nested objects or
  arrays), so it fell back to `json_object`. Your build may also not support that.
- mtplx: `llguidance` is missing from the current venv (common after `brew upgrade`).
- TensorFold: install the `tensorfold[grammar]` extra.

### Think tags show up in the answer text
The model emitted a `<think>` block even with thinking off. The proxy's
`ContentThinkRouter` moves such blocks to `reasoning_content` wherever they appear. If you
still see raw tags, the model uses an opener/closer not in `OPENERS`/`CLOSERS` in
`llm_proxy.py`; add it there.

### Prefill speed shows 10,000+ tok/s
That's an engine dividing total prompt tokens by the time spent on the uncached part. The
proxy recomputes from `usage.cached_tokens`; a full cache hit shows `–`. If you see absurd
values, the engine isn't reporting `cached_tokens`.

### Dashboard says "offline" though the API works
- The dashboard looks for the router on the same host, port 8090. Open it via the router
  (`http://<mac>:8090/`) or pass `?hub=http://<mac>:8090`.
- When editing `index.html`: removing an element while its `$("id")` update stays in the
  poll loop throws on every tick and makes the whole page show offline.

### Session token counter resets every turn
Your client's first user message changes between requests (some clients inject a timestamp
or rewrite history). Session detection fingerprints the first user message. Enable
`"session_log": true` to see the fingerprints per request.

### Model never comes up (503 `llmhub_load_timeout`)
Read `<key>-server.log`. Common causes:
- wrong model path,
- not enough wired memory (raise `iogpu.wired_limit_mb`),
- the port is still held by a dying process,
- for mtplx, auth enabled (missing `--no-auth`) — the proxy's health check gets 401.

Very large models may simply need more than `load_timeout` seconds on a cold disk cache.

### A model shows 256K context in the client but the server runs 128K
The client uses its provider-wide default. Add a per-model `context_length` override
([CLIENTS.md](CLIENTS.md)).
