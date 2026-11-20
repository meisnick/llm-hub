# Clients

The hub is a plain OpenAI-compatible API at `http://<mac>:8090/v1`. No API key is needed;
send any non-empty string if your client insists.

## Hermes Agent

```yaml
# config.yaml
model:
  default: MTPLX-Qwen3.8-27B        # any model id from /v1/models
  provider: llmhub
  base_url: http://<mac>:8090/v1
  api_mode: chat_completions
  max_tokens: 16384

providers:
  llmhub:
    name: Local LLM Hub
    base_url: http://<mac>:8090/v1
    api_mode: chat_completions
    api_key: not-needed
    discover_models: true           # new hub models appear in /model automatically
    context_length: 262144          # default for models without an override
    models:                         # per-model context windows (match the server's -c)
      Nemotron-3.5-Lightning-30B:
        context_length: 131072
      TF-Qwen3.8-27B:
        context_length: 65536

delegation:
  model: MTPLX-Qwen3.8-27B          # keep equal to model.default to avoid swap-thrash
  provider: llmhub
```

Tips:

- **Set per-model `context_length`.** Otherwise the client assumes the provider default and
  only compresses history after the server has already rejected an over-long prompt.
- **Use the same model for the main loop and delegation.** Different models make every
  sub-agent call trigger a full model swap.
- Hermes' reasoning ladder (`none … ultra`) is clamped per model by the proxy, so any
  setting is safe.
- Restart the client or start a new session after config changes.

## Open WebUI / LibreChat / anything OpenAI-compatible

Add an OpenAI connection with base URL `http://<mac>:8090/v1`. The model list comes from
`/v1/models`. The first message to a model that isn't loaded waits for the swap: seconds for
small GGUFs, about 30 s for 30B-class models. Raise the client's request timeout if needed.

## curl

```bash
curl -s http://<mac>:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Nemotron-3.5-Lightning-30B",
  "messages": [{"role": "user", "content": "Hello"}],
  "reasoning_effort": "none",
  "stream": true }'
```

## Python (openai SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="http://<mac>:8090/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="TF-Qwen3.8-27B",
    messages=[{"role": "user", "content": "Explain speculative decoding in two sentences."}],
    reasoning_effort="low",
)
print(r.choices[0].message.content)
print(getattr(r.choices[0].message, "reasoning_content", None))   # the thinking, if any
```

## Behavior clients can rely on

| Feature | Behavior |
|---|---|
| Reasoning | arrives in `reasoning_content` (stream deltas and final message); `content` never contains think tags |
| `reasoning_effort` | `none` disables thinking; other values are mapped to the nearest level the model supports |
| Tools | `tools` / `tool_calls` / `finish_reason: tool_calls` pass through unchanged |
| Structured output | `response_format` `json_schema` / `json_object` works on all engines (thinking is turned off for these calls) |
| Long prefills | SSE comment keepalives every 15 s while the prompt is processed |
| Model swap timeout | HTTP 503, `error.type = "llmhub_load_timeout"` |
