#!/usr/bin/env python3
"""End-to-end smoke test of a running hub - the request shapes agent clients actually send.

    python3 tools/check_hub.py --hub http://localhost:8090 --model Nemotron-3.5-Lightning-30B

Checks: /v1/models lists the model, plain chat, streaming (reasoning split from answer),
tool calling, json_schema structured output (title generation), and reasoning_effort=none.
Requesting a model that is not loaded makes the hub switch to it (can take a while).
"""
import argparse, json, sys, time, urllib.request


def post(hub, body, timeout):
    req = urllib.request.Request(hub + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default="http://localhost:8090")
    ap.add_argument("--model", required=True)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    hub, M, T = a.hub.rstrip("/"), a.model, a.timeout
    results = []

    def check(name, fn):
        t = time.time()
        try:
            detail = fn()
            results.append((True, name, detail, time.time() - t))
        except Exception as e:
            results.append((False, name, f"{type(e).__name__}: {e}", time.time() - t))
        ok, n, d, dt = results[-1]
        print(f"{'PASS' if ok else 'FAIL'}  {n:<22} {dt:6.1f}s  {d}")

    def models():
        ids = [m["id"] for m in json.load(urllib.request.urlopen(hub + "/v1/models", timeout=10))["data"]]
        assert M in ids, f"{M} not in {ids}"
        return f"{len(ids)} models listed"

    def chat():
        r = json.load(post(hub, {"model": M, "max_tokens": 400,
                                 "messages": [{"role": "user", "content": "Reply with exactly: hub ok"}]}, T))
        msg = r["choices"][0]["message"]
        assert "hub ok" in (msg.get("content") or "").lower(), repr(msg.get("content"))
        return f"content={msg['content']!r} reasoning={len(msg.get('reasoning_content') or '')} chars"

    def stream():
        resp = post(hub, {"model": M, "max_tokens": 400, "stream": True,
                          "messages": [{"role": "user", "content": "Count from 1 to 5, comma separated."}]}, T)
        content, reasoning = "", ""
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("error"):
                raise RuntimeError(d["error"])
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
        assert "<think>" not in content and "</think>" not in content, "think tags leaked into content"
        assert "3" in content, repr(content)
        return f"content={content.strip()!r} reasoning={len(reasoning)} chars"

    def tools():
        r = json.load(post(hub, {"model": M, "max_tokens": 800,
            "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
            "tools": [{"type": "function", "function": {"name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                               "required": ["city"]}}}]}, T))
        c = r["choices"][0]
        calls = c["message"].get("tool_calls") or []
        assert calls and calls[0]["function"]["name"] == "get_weather", c
        return f"finish={c['finish_reason']} args={calls[0]['function']['arguments']}"

    def structured():
        r = json.load(post(hub, {"model": M, "max_tokens": 200,
            "messages": [{"role": "user", "content": "Give a short title for a chat about rain gauges."}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": {
                "type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}}}}, T))
        obj = json.loads(r["choices"][0]["message"]["content"])
        assert isinstance(obj.get("title"), str) and obj["title"], obj
        return json.dumps(obj)

    def no_reasoning():
        r = json.load(post(hub, {"model": M, "max_tokens": 200, "reasoning_effort": "none",
                                 "messages": [{"role": "user", "content": "What is 2+2? One word."}]}, T))
        msg = r["choices"][0]["message"]
        return f"content={(msg.get('content') or '').strip()!r} reasoning={len(msg.get('reasoning_content') or '')} chars"

    for name, fn in (("models listed", models), ("chat", chat), ("stream split", stream),
                     ("tool call", tools), ("json_schema", structured), ("reasoning none", no_reasoning)):
        check(name, fn)
    bad = [r for r in results if not r[0]]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
