# Architecture

```
                    ┌──────────────────────────── one Mac ─────────────────────────────┐
 client ──HTTP──►   │ :8090 llm_router.py                                              │
                    │   ├─ GET  /v1/models ── from hub.json                            │
                    │   ├─ POST /v1/*  ── ensure_loaded(model) ──► llm_switch.py <key> │
                    │   │                      │  bootout other servers, kill strays   │
                    │   │                      └► bootstrap + kickstart target server  │
                    │   └─ stream ──► :<proxy_port> llm_proxy.py ──► :<server_port>    │
                    │                    (per model, always up)       engine (1 at a time)
                    └──────────────────────────────────────────────────────────────────┘
```

## Processes

| Process | Count | Lifetime | Binds |
|---|---|---|---|
| Router (`llm_router.py`) | 1 | always (launchd KeepAlive) | `listen_host:listen_port` (default `0.0.0.0:8090`) |
| Proxy (`llm_proxy.py --model <key>`) | 1 per model | always | `127.0.0.1:<proxy_port>` |
| Engine server | 1 per model, **only one running** | on demand | `<upstream_host>:<server_port>` |

Proxies are cheap (a few MB each) and stay up even when their server is down. A proxy
then reports `"up": false` on `/stats` and returns 502 for requests. Keeping them always-on
means the router's detection and the dashboard never have to start anything.

Server plists live in `server_plist_dir` (default `~/.config/llmhub/launchd`), **not**
`~/Library/LaunchAgents`. launchd therefore never starts them at login, so a reboot can't
try to load every model at once. Only `llm_switch.py` bootstraps a server.

## Request flow

1. The client sends `POST /v1/chat/completions` with `"model": "<id>"`.
2. The router fingerprints the conversation (SHA-1 of the first user message) to keep a
   per-chat session counter (see [STATS.md](STATS.md)).
3. **`ensure_loaded(id)`**, under a lock so concurrent requests can't interleave switches:
   - If that model's proxy already reports `up`, it continues immediately.
   - Otherwise it runs `llm_switch.py <key>`, then polls the proxy's `/stats` once a second
     until `up` is true or `load_timeout` (default 180 s) passes. On timeout the client gets
     a 503 with `type: llmhub_load_timeout`.
4. The body is forwarded unchanged to `127.0.0.1:<proxy_port>`. The proxy rewrites what it
   must and forwards it to the engine.
5. The response is streamed back with `read1()`, which forwards whatever bytes have
   arrived. (`read(4096)` would hold short replies until the end and deliver long ones
   in bursts.)
6. The last 256 KB of the stream is parsed for the turn's generated-token count, which
   feeds the session and all-time counters. Requests with `response_format` (title
   generation and similar auxiliary calls) are not counted.

Requests without a known `model` go to whatever is currently loaded.

## Self-healing active-model detection

`detect_active()` asks every proxy for `/stats` and takes the first one reporting
`up: true`. The result is cached for about 2 s. This one rule covers:

- a model switched by hand (`llm_switch.py` on the box),
- an engine crash (KeepAlive restarts it, and detection follows),
- a router restart (no stale "last known" model),
- requesting the model that's already loaded (no needless reload).

Because detection takes the *first* `up` proxy, two servers running at once would make it
flap. `llm_switch.py` prevents that: after booting out the other agents it kills anything
still listening on another model's server port. Some engines leave a child process behind
when their launchd job stops (mtplx's server does).

## The proxy pipeline

For each chat request (`hub/llm_proxy.py`):

1. **Reasoning control.** `reasoning_effort` (or `extra_body.reasoning.effort`) is
   normalized to `none|low|medium|high|xhigh`:
   - `none` also sets `chat_template_kwargs.enable_thinking=false` (and `reasoning: off`
     for mtplx).
   - For llama.cpp the proxy reads the chat template from `GET /props` (cached 30 s), parses
     its "Supported values/types" list and maps to the nearest allowed level, rounding ties
     up. Some templates accept `high` but not `xhigh`, others the reverse, and an
     unsupported value is a Jinja error that surfaces as an empty reply.
2. **Structured output.** For `response_format: json_schema|json_object` thinking is turned
   off, because a think trace would eat the token budget.
   - llama.cpp: a flat schema of primitive properties is compiled to a GBNF `grammar`;
     anything else falls back to `json_object`.
   - mtplx / TensorFold: the request is passed through, and they enforce it natively.
3. **Streaming split.**
   - *Tag-based* (llama.cpp with `--reasoning-format none`): the `ThinkSplitter` moves
     everything before the closing think tag into `reasoning_content`. It handles openers,
     closers and tags split across chunks.
   - *Pre-split* (mtplx, TensorFold, thinking off): reasoning already arrives in
     `reasoning_content` and is passed through. A `ContentThinkRouter` still watches
     `content` for a stray `<think>…</think>` block (some models think anyway with thinking
     off) and moves it to reasoning. Unchanged chunks are forwarded byte-for-byte.
   - `tool_calls`, `finish_reason`, `usage` and `timings` are always preserved.
4. **Keepalive.** During long prefills the proxy sends an SSE comment every 15 s so clients
   and intermediaries don't time out.
5. **Stats.** Phase (`prefill → thinking → generating → idle`), TTFT and live token counts
   update as chunks arrive. The engine's final `timings`/`usage` replace estimates at the
   end of the turn ([STATS.md](STATS.md)).

Non-streaming responses get the same think-tag split applied to the final message.

## HTTP API

### Router (`:8090`)

| Method | Path | Description |
|---|---|---|
| GET | `/v1/models` | all configured models (`owned_by: llmhub`) |
| POST | `/v1/chat/completions`, `/v1/*` | auto-swap + forward |
| GET | `/sysinfo` | `active_model`, `models[]`, `session`, `alltime`, memory, disk, optional temps |
| GET | `/stats` | the active model's proxy `/stats` |
| POST | `/restart` | re-run the switcher for the active model |
| GET | `/`, `/dashboard` | dashboard HTML |
| GET | `/health` | `{"ok": true}` |

### Proxy (`:<proxy_port>`)

| Method | Path | Description |
|---|---|---|
| GET | `/stats` | `{model, up, phase, active, turn, last, totals, history, think_history}` |
| * | anything else | forwarded to the engine (chat completions are rewritten as above) |

`last` holds `ttft, prefill_tps, prompt_tokens, think_tps, gen_tps, decode_tps,
reason_tokens, content_tokens, predicted_tokens, duration`, plus `speculative` (acceptance
rate, tokens per round) on TensorFold.
