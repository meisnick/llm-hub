#!/usr/bin/env python3
"""LLM Hub adapter + live-stats proxy (one instance per model).

    client -> router :8090 -> THIS PROXY (proxy_port) -> engine server (server_port)

What it does to every request/response:
  1. Reasoning normalization. Tag-based engines (llama.cpp) stream raw ``<think>...</think>``
     inside ``content``; the ThinkSplitter turns that into ``reasoning_content`` + clean
     ``content``. Pre-split engines (mtplx, TensorFold, oMLX) already separate them and are
     passed through. ``tool_calls`` are never touched.
  2. reasoning_effort clamping. Clients send a wide ladder (none..ultra); each chat template
     accepts a different subset. The proxy reads the live template from ``GET /props`` and maps
     to the nearest supported level so the engine never 400s/500s on an unknown value.
  3. Structured-output repair. Some llama.cpp builds reject ``response_format: json_schema``;
     the proxy rewrites simple schemas to a GBNF grammar (thinking off).
  4. Live ``/stats``. Because it sees every token it exposes per-turn TTFT, prefill/decode
     speeds and think/answer token splits for the dashboard, using the engine's own
     ``timings``/``usage`` as ground truth.

Configuration (either form):
  * ``llm_proxy.py --model <key> [--config hub.json]`` - read everything from the hub config
  * environment: LLMHUB_UPSTREAM_PORT, LLMHUB_LISTEN_PORT, LLMHUB_MODEL, LLMHUB_ENGINE,
    LLMHUB_UPSTREAM_HOST, LLMHUB_LOG, LLMHUB_DUMP_BODY (legacy K2_* names are also accepted)
"""
import argparse, http.server, http.client, json, time, threading, os, re, sys


def _env(name, default=None):
    return os.environ.get("LLMHUB_" + name, os.environ.get("K2_" + name, default))


def _configure():
    ap = argparse.ArgumentParser(description="LLM Hub per-model adapter proxy")
    ap.add_argument("--model", help="model key or id from the hub config")
    ap.add_argument("--config", help="path to hub.json (default: LLMHUB_CONFIG / ~/.config/llmhub/hub.json)")
    a, _ = ap.parse_known_args()
    conf = {}
    if a.model:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hubconfig
        cfg = hubconfig.load(a.config)
        m = hubconfig.find(cfg, a.model)
        if not m:
            raise SystemExit(f"llm_proxy: model '{a.model}' not in {cfg['_path']}")
        conf = {"UPSTREAM_HOST": m["upstream_host"], "UPSTREAM_PORT": m["server_port"],
                "LISTEN_PORT": m["proxy_port"], "MODEL": m["id"], "ENGINE": m["engine"],
                "LOG": os.path.join(cfg["state_dir"], f"{m['key']}-proxy.log"),
                "STATE_DIR": cfg["state_dir"]}
    get = lambda k, d: conf.get(k) if conf.get(k) is not None else _env(k, d)
    return conf, get


if __name__ == "__main__":
    _CONF, _get = _configure()
else:  # imported (tests): environment/defaults only
    _CONF, _get = {}, (lambda k, d: _env(k, d))

STATE_DIR = os.path.expanduser(_get("STATE_DIR", "~/.local/state/llmhub"))
UP_HOST = _get("UPSTREAM_HOST", "127.0.0.1")
UP_PORT = int(_get("UPSTREAM_PORT", "8081"))
LISTEN_PORT = int(_get("LISTEN_PORT", "8091"))
MODEL = _get("MODEL", "local-model")
LOG = os.path.expanduser(_get("LOG", os.path.join(STATE_DIR, "proxy-%d.log" % LISTEN_PORT)))
# Engine family decides reasoning/structured-output handling. Explicit is better than the
# old name/port sniffing; it is still the fallback when LLMHUB_ENGINE is unset.
ENGINE = (_get("ENGINE", "") or "").lower()
# Debug only: write the full last chat request body (incl. message text) to the state dir.
# OFF by default for privacy; enable with LLMHUB_DUMP_BODY=1.
DUMP_BODY = str(_get("DUMP_BODY", "0")).lower() in ("1", "true", "yes", "on")
try:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
except Exception:
    pass
CLOSERS = ["</think>", "</ifm|think_faster>", "</ifm|think_fast>", "</ifm|think>"]
OPENERS = ["<think>", "<ifm|think_faster>", "<ifm|think_fast>", "<ifm|think>"]
MAXTAG = max(len(c) for c in CLOSERS)

LOCK = threading.Lock()
STATS = {
    "phase": "idle",            # idle | prefill | thinking | generating
    "active": False,
    "turn_start": 0.0, "ttft": None, "think_start": None, "gen_start": None,
    "last_tok": None,           # wall-clock time of the most recent generated token
    "reason_tokens": 0, "content_tokens": 0,   # SSE-delta counts (used for the think/answer SPLIT ratio only)
    "last": None,               # {ttft, prefill_tps, prompt_tokens, think_tps, gen_tps, decode_tps, reason_tokens, content_tokens, predicted_tokens, duration}
    "totals": {"turns": 0, "reason_tokens": 0, "content_tokens": 0},
    "history": [],              # last N gen_tps
    "think_history": [],        # last N think_tps
}

def log(s):
    try:
        with open(LOG, "a") as f: f.write(s + "\n")
    except Exception: pass

_health = {"t": 0.0, "ok": False}
def upstream_up():
    # cached (~2s) health probe of the llama-server this proxy fronts
    now = time.time()
    if now - _health["t"] < 2.0:
        return _health["ok"]
    ok = False
    try:
        c = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=1)
        c.request("GET", "/health"); r = c.getresponse(); ok = (r.status == 200)
        r.read(); c.close()
    except Exception:
        ok = False
    _health.update(t=now, ok=ok)
    return ok

# ---- per-model reasoning_effort support, read live from the upstream chat template ----
# Templates validate effort differently, e.g.
#   K2-Horizon : "Supported values: high, medium, low"
#   Qwen3.8    : "Supported types are xhigh (default), medium, and low"
# so we read GET /props -> chat_template and parse the allowed set (cached 30s).
EFFORT_RANK = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}
_eff = {"t": 0.0, "set": None, "known": False}
def supported_efforts():
    now = time.time()
    if _eff["known"] and now - _eff["t"] < 30.0:
        return _eff["set"]
    s, known = None, False
    try:
        c = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=2)
        c.request("GET", "/props"); r = c.getresponse(); raw = r.read(); c.close()
        if r.status == 200:
            tmpl = json.loads(raw).get("chat_template") or ""
            known = True
            m = re.search(r"Supported (?:values|types)[^\n'\"]{0,120}", tmpl)
            if m:
                found = set(re.findall(r"\b(minimal|low|medium|high|xhigh)\b", m.group(0)))
                s = found or None
    except Exception:
        pass
    _eff.update(t=now, set=s, known=known)
    return s

def map_effort(req, allowed):
    # nearest supported level; ties prefer MORE thinking
    if not allowed or req in allowed:
        return req
    r = EFFORT_RANK.get(req, 2)
    return min(allowed, key=lambda a: (abs(EFFORT_RANK[a] - r), -EFFORT_RANK[a]))

def count_text_tokens(text):
    if not text:
        return 0
    m = re.findall(r"\w+|[^\w\s]", text)
    return max(1, len(m)) if text.strip() else 1

def tps(tokens, dt):
    # guarded tokens/sec: need a real window + a few tokens; reject nonsense bursts
    if not tokens or tokens < 3 or not dt or dt < 0.1:
        return None
    r = tokens / dt
    return None if r > 1000 else round(r, 1)

def split_tps(tok, ms, decode):
    # per-phase (thinking / answer) rate from the authoritative decode total,
    # apportioned by wall-clock. A small phase can't be timed from bursty SSE
    # arrivals (you'd see a rate faster than the real decode loop), so fall back
    # to the authoritative decode rate; None only when the phase never happened.
    if not tok:
        return None
    if tok < 8 or not ms or ms < 500.0:
        return decode
    r = tps(tok, ms / 1000.0)
    return r if r else decode

def snap():
    with LOCK:
        s = dict(STATS)
        now = time.time()
        turn = None
        if s["active"]:
            elapsed = now - s["turn_start"]
            gtps = tps(s["content_tokens"], (now - s["gen_start"]) if s["gen_start"] else 0)
            tend = s["gen_start"] or now
            ttps = tps(s["reason_tokens"], (tend - s["think_start"]) if s["think_start"] else 0)
            live = ttps if s["phase"] == "thinking" else gtps
            turn = {"elapsed": round(elapsed, 1),
                    "ttft": round(s["ttft"], 2) if s["ttft"] else None,
                    "reason_tokens": s["reason_tokens"],
                    "content_tokens": s["content_tokens"],
                    "think_tps": ttps,
                    "gen_tps": gtps,
                    "tps": live}
        return {"phase": s["phase"], "active": s["active"], "turn": turn,
                "last": s["last"], "totals": s["totals"],
                "history": s["history"][-60:], "think_history": s["think_history"][-60:]}

class ThinkSplitter:
    def __init__(self):
        self.phase = "think"; self.buf = ""; self.opener_done = False
    def _strip_opener(self):
        # strip a leading <ifm|think*> opener (+ newlines) once, tolerating chunk splits
        s = self.buf.lstrip("\n")
        for op in OPENERS:
            if s.startswith(op):
                self.buf = s[len(op):].lstrip("\n"); self.opener_done = True; return True
        for op in OPENERS:                       # a partial opener may still be forming
            if len(s) < len(op) and op.startswith(s):
                return False                     # wait for more bytes
        self.opener_done = True; return True     # no opener present
    def feed(self, text):
        out = []; self.buf += text
        if self.phase == "think":
            if not self.opener_done and not self._strip_opener():
                return out
            idx, found = -1, None
            for cl in CLOSERS:
                i = self.buf.find(cl)
                if i != -1 and (idx == -1 or i < idx): idx, found = i, cl
            if idx != -1:
                before, after = self.buf[:idx], self.buf[idx+len(found):]
                if before: out.append(("reasoning", before))
                self.phase = "content"; self.buf = ""
                if after: out.append(("content", after))
            else:
                if len(self.buf) > MAXTAG:
                    emit = self.buf[:-MAXTAG]; self.buf = self.buf[-MAXTAG:]
                    if emit: out.append(("reasoning", emit))
        else:
            if self.buf: out.append(("content", self.buf)); self.buf = ""
        return out
    def flush(self):
        if self.buf:
            ch = "reasoning" if self.phase == "think" else "content"
            seg = [(ch, self.buf)]; self.buf = ""; return seg
        return []

class ContentThinkRouter:
    """Pre-split / thinking-off path: route a think block found INSIDE the content to reasoning.
    Qwen3.8-27B-Uncensored sometimes thinks anyway at reasoning 'none' and writes the whole
    <think>...</think> block as content; the old head-only filter missed it whenever the first
    chunk was just newlines, so clients showed it as the answer (and a voice client read it aloud).
    Empty blocks vanish, stray closers are dropped, partial tags are held across chunks, and
    whitespace before the answer starts is dropped."""
    def __init__(self):
        self.buf, self.think, self.started, self.r_started = "", False, False, False
    @staticmethod
    def _partial(s, tags):  # longest suffix of s that could still become one of tags
        best = 0
        for t in tags:
            for k in range(min(len(t) - 1, len(s)), best, -1):
                if s.endswith(t[:k]):
                    best = k; break
        return best
    def _add(self, out, piece):
        if not piece: return
        if self.think:
            if not self.r_started:  # the template's empty block is just newlines: emit nothing for it
                piece = piece.lstrip()
                if not piece: return
                self.r_started = True
            out.append(("reasoning", piece)); return
        if not self.started:
            piece = piece.lstrip()
            if not piece: return
            self.started = True
        out.append(("content", piece))
    def feed(self, text):
        self.buf += text; out = []
        while self.buf:
            tags = CLOSERS if self.think else OPENERS + CLOSERS
            hits = [(self.buf.find(t), t) for t in tags if t in self.buf]
            if not hits:
                keep = self._partial(self.buf, tags)
                self._add(out, self.buf[:len(self.buf) - keep])
                self.buf = self.buf[len(self.buf) - keep:]
                break
            i, tag = min(hits)
            self._add(out, self.buf[:i])
            self.buf = self.buf[i + len(tag):]
            self.think = tag in OPENERS
            if self.think: self.r_started = False
        return out
    def flush(self):
        out, rest, self.buf = [], self.buf, ""
        self._add(out, rest)
        return out

def schema_to_gbnf(schema):
    """Minimal JSON-schema -> GBNF for an object of primitive props (covers title-gen).
    Returns a grammar string, or None to signal 'fall back to json_object'."""
    try:
        if not isinstance(schema, dict) or schema.get("type") != "object":
            return None
        props = schema.get("properties") or {}
        if not props:
            return None
        parts = []
        for k, spec in props.items():
            t = (spec or {}).get("type")
            v = {"string": "string", "number": "number", "integer": "number",
                 "boolean": "boolean"}.get(t)
            if not v:
                return None
            keytok = r'"\"' + k + r'\""'          # GBNF literal for "key"
            parts.append(keytok + ' ws ":" ws ' + v)
        body = ' ws "," ws '.join(parts)
        root = 'root ::= "{" ws ' + body + ' ws "}"\n'
        helpers = (
            r'string ::= "\"" ( [^"\\] | "\\" . )* "\""' + "\n"
            r'number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?' + "\n"
            r'boolean ::= "true" | "false"' + "\n"
            r'ws ::= [ \t\n]*' + "\n"
        )
        return root + helpers
    except Exception:
        return None

def sse(obj):
    return ("data: " + json.dumps(obj, separators=(",", ":")) + "\n\n").encode("utf-8")

def cors(h):
    h.send_header("Access-Control-Allow-Origin", "*")

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _stats(self):
        d = snap(); d["model"] = MODEL; d["up"] = upstream_up()
        body = json.dumps(d).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        cors(self)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try: self.wfile.write(body)
        except Exception: pass

    def _proxy(self, method):
        if self.path.rstrip("/") == "/stats":
            return self._stats()
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        is_chat = self.path.endswith("/chat/completions")
        streaming = False
        if is_chat and body:
            try:
                j = json.loads(body); streaming = bool(j.get("stream"))

                # Normalize reasoning controls for the active backend engine.
                # Hermes emits: none, minimal, low, medium, high, xhigh, max, ultra.
                re_in = j.get("reasoning_effort")
                if re_in is None and isinstance(j.get("extra_body"), dict):
                    re_in = j["extra_body"].get("reasoning", {}).get("effort") or j["extra_body"].get("reasoning_effort")

                re_norm = None
                if ENGINE:
                    is_mtplx = ENGINE == "mtplx"
                    is_tf = ENGINE == "tensorfold"
                    # generic OpenAI servers (oMLX, vLLM, ...) split reasoning themselves and
                    # enforce response_format natively: handle them like TensorFold (passthrough)
                    is_tf = is_tf or ENGINE == "openai"
                else:  # legacy heuristic when no engine is configured
                    is_mtplx = "mtplx" in MODEL.lower()
                    is_tf = MODEL.lower().startswith("tf-") or "tensorfold" in MODEL.lower()

                if re_in is not None:
                    re_str = str(re_in).lower().strip()
                    if re_str in ("none", "off", "false", "0"):
                        re_norm = "none"
                    elif re_str in ("minimal", "low"):
                        re_norm = "low"
                    elif re_str == "medium":
                        re_norm = "medium"
                    elif re_str in ("high", "xhigh", "max", "ultra"):
                        # K2 and llama-server templates require "high" (low/medium/high only).
                        # mtplx natively accepts "xhigh".
                        re_norm = "xhigh" if (is_mtplx and re_str in ("xhigh", "max", "ultra")) else "high"
                    else:
                        re_norm = "medium"

                    # Clamp to what THIS upstream's template actually accepts (read live).
                    if re_norm != "none" and not is_mtplx:
                        want = "xhigh" if re_str in ("xhigh", "max", "ultra") else re_norm
                        allowed = supported_efforts()
                        if allowed:
                            re_norm = map_effort(want, allowed)

                    j["reasoning_effort"] = re_norm
                    if re_norm == "none":
                        ctk = j.get("chat_template_kwargs") or {}
                        ctk["enable_thinking"] = False
                        j["chat_template_kwargs"] = ctk
                        if is_mtplx:
                            j["reasoning"] = "off"
                    log(f"[reasoning] mapped {re_in!r} -> {re_norm!r} (model={MODEL})")

                # This llama.cpp build throws on response_format=json_schema
                # ("Failed to initialize samplers"). json_object works — rewrite it.
                rf = j.get("response_format")
                if isinstance(rf, dict) and rf.get("type") in ("json_schema", "json_object"):
                    # structured calls want JSON directly, not a think trace that eats
                    # the token budget -> disable thinking
                    ctk = j.get("chat_template_kwargs") or {}
                    ctk["enable_thinking"] = False
                    j["chat_template_kwargs"] = ctk
                    j["reasoning_effort"] = "none"
                    if is_mtplx: j["reasoning"] = "off"
                    gbnf = None
                    if is_mtplx or is_tf:
                        # mtplx/TensorFold ignore GBNF but enforce response_format natively
                        # (needs llguidance / tensorfold[grammar] in their venvs) -> pass through
                        log(f"[rewrite] {rf.get('type')} passthrough + reasoning off ({'mtplx' if is_mtplx else 'tf'})")
                    else:
                        if rf.get("type") == "json_schema":
                            gbnf = schema_to_gbnf((rf.get("json_schema") or {}).get("schema"))
                        if gbnf:
                            j["grammar"] = gbnf                 # strict schema via grammar (works here)
                            j.pop("response_format", None)
                            log(f"[rewrite] json_schema -> grammar + enable_thinking:false ({self.path})")
                        else:
                            j["response_format"] = {"type": "json_object"}
                            log(f"[rewrite] -> json_object + enable_thinking:false ({self.path})")
                body = json.dumps(j).encode("utf-8")
            except Exception as e:
                log(f"[req rewrite err] {e}")

        try:
            conn = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=600)
            hdrs = {k: v for k, v in self.headers.items()
                    if k.lower() not in ("host", "content-length", "connection")}
            
            keepalive_run = [False]
            if is_chat:
                with LOCK:
                    STATS.update(phase="prefill", active=True, turn_start=time.time(),
                                 ttft=None, think_start=None, gen_start=None, last_tok=None,
                                 reason_tokens=0, content_tokens=0)
                if streaming:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    
                    keepalive_run[0] = True
                    def keepalive_thread():
                        while keepalive_run[0]:
                            for _ in range(15):
                                if not keepalive_run[0]: return
                                time.sleep(1)
                            if not keepalive_run[0]: return
                            try:
                                log(f"[{time.time()}] FIRING KEEPALIVE")
                                self.connection.sendall(b": proxy-prefill-keepalive\n\n")
                            except Exception as e:
                                log(f"[{time.time()}] KEEPALIVE FAILED: {e}")
                                keepalive_run[0] = False
                                break
                    threading.Thread(target=keepalive_thread, daemon=True).start()

            if is_chat and DUMP_BODY:
                with open(os.path.join(STATE_DIR, "last_proxy_body.json"), "w") as bf:
                    bf.write(body.decode("utf-8"))
            conn.request(method, self.path, body=body, headers=hdrs)
            resp = conn.getresponse()
        except Exception as e:
            if 'keepalive_run' in locals(): keepalive_run[0] = False
            log(f"[upstream err] {e}")
            if is_chat:
                with LOCK: STATS.update(phase="idle", active=False)
            
            if is_chat and streaming:
                try:
                    self.wfile.write(f'data: {{"error": {{"message": "proxy upstream failed: {e}", "type": "proxy_error"}}}}\n\n'.encode())
                    self.wfile.write(b'data: [DONE]\n\n')
                except: pass
            else:
                self.send_response(502); self.end_headers()
                try: self.wfile.write(b'{"error":"proxy upstream failed"}')
                except: pass
            return

        if is_chat and streaming:
            if resp.status != 200:
                keepalive_run[0] = False
                raw = resp.read()
                try: conn.close()
                except: pass
                with LOCK: STATS.update(phase="idle", active=False)
                try:
                    d = json.loads(raw)
                    if "error" not in d: d = {"error": {"message": raw.decode('utf-8','replace'), "type": "upstream_error"}}
                except:
                    d = {"error": {"message": raw.decode('utf-8','replace'), "type": "upstream_error"}}
                
                try:
                    self.wfile.write(f'data: {json.dumps(d)}\n\n'.encode())
                    self.wfile.write(b'data: [DONE]\n\n')
                except: pass
                
                jb = json.loads(body) if body else {}
                keys = {k: jb.get(k) for k in ("model","stream","temperature","top_k","top_p","min_p",
                        "max_tokens","n_predict","response_format","grammar","json_schema") if k in jb}
                log(f"\n##### {time.strftime('%H:%M:%S')} NON-200 status={resp.status} path={self.path} #####")
                log("REQ KEYS: " + json.dumps(keys, default=str)[:1500])
                log("RESP BODY: " + raw.decode('utf-8','replace')[:800])
                return

            sp = ThinkSplitter(); rbuf = b""
            timings = None      # llama.cpp's authoritative per-turn timings (final chunk)
            usage = None        # OpenAI usage block (prompt_tokens + cached_tokens)
            tf_data = None      # TensorFold telemetry
            spec_data = None    # TensorFold speculative decode stats
            pre_split = is_mtplx or is_tf or (re_norm == "none") or (j.get("chat_template_kwargs") or {}).get("enable_thinking") is False
            # thinking-off / pre-split content can still carry a think block (empty from the template,
            # or a real one from a model that thinks anyway): route it to reasoning wherever it appears.
            router = ContentThinkRouter()
            def emit(base, segs):
                for ch, txt in segs:
                    o = {"id": base.get("id"), "object": base.get("object"),
                         "created": base.get("created"), "model": base.get("model"),
                         "choices": [{"index": 0,
                                      "delta": ({"reasoning_content": txt} if ch == "reasoning" else {"content": txt}),
                                      "finish_reason": None}]}
                    try: self.wfile.write(sse(o))
                    except Exception: return
            try:
                while True:
                    chunk = resp.read(1024)
                    if not chunk:
                        keepalive_run[0] = False
                        break
                    rbuf += chunk
                    while b"\n\n" in rbuf:
                        raw_evt, rbuf = rbuf.split(b"\n\n", 1)
                        line = raw_evt.decode("utf-8", "replace").strip()
                        if line == ": keep-alive":
                            self.wfile.write(b": keep-alive\n\n"); self.wfile.flush()
                            continue
                        if not line.startswith("data:"): continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            if not pre_split:
                                for ch, txt in sp.flush(): emit({}, [(ch, txt)])
                            else:
                                rest = router.flush()
                                if rest: emit({}, rest)
                            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                            continue
                        try: d = json.loads(payload)
                        except Exception:
                            self.wfile.write((line + "\n\n").encode()); continue
                        # capture authoritative timings/usage whenever the engine attaches them
                        if isinstance(d, dict) and d.get("timings"):
                            timings = d["timings"]
                        if isinstance(d, dict) and d.get("usage"):
                            usage = d["usage"]
                        if isinstance(d, dict) and d.get("tensorfold"):
                            tf_data = d["tensorfold"]
                        if isinstance(d, dict) and d.get("speculative"):
                            spec_data = d["speculative"]
                        ch0 = (d.get("choices") or [{}])[0]
                        delta = ch0.get("delta", {}) or {}
                        cont = delta.get("content")
                        rcont = delta.get("reasoning_content")
                        other = {k: v for k, v in delta.items() if k not in ("content", "reasoning_content", "role")}
                        finish = ch0.get("finish_reason")
                        if rcont is not None and not cont:
                            # PRE-SPLIT upstream (mtplx/oMLX): reasoning arrives already separated.
                            # Bypass the ThinkSplitter; pass it through and count it as thinking.
                            pre_split = True
                            now = time.time()
                            toks = count_text_tokens(rcont)
                            with LOCK:
                                STATS["last_tok"] = now
                                if STATS["ttft"] is None: STATS["ttft"] = now - STATS["turn_start"]
                                if STATS["think_start"] is None: STATS["think_start"] = now
                                STATS["reason_tokens"] += toks
                                STATS["phase"] = "thinking"
                            self.wfile.write((line + "\n\n").encode())
                        elif cont is not None and not other and pre_split:
                            # answer content from a pre-split upstream (or thinking off): route any think
                            # block inside it to reasoning_content, count each part as what it is
                            segs = router.feed(cont)
                            if finish is not None:
                                segs += router.flush()
                            now = time.time()
                            with LOCK:
                                for ch, txt in segs:
                                    STATS["last_tok"] = now
                                    if STATS["ttft"] is None: STATS["ttft"] = now - STATS["turn_start"]
                                    if ch == "reasoning":
                                        if STATS["think_start"] is None: STATS["think_start"] = now
                                        STATS["reason_tokens"] += count_text_tokens(txt)
                                        STATS["phase"] = "thinking"
                                    else:
                                        if STATS["gen_start"] is None: STATS["gen_start"] = now
                                        STATS["content_tokens"] += count_text_tokens(txt)
                                        STATS["phase"] = "generating"
                            if segs == [("content", cont)]:
                                self.wfile.write((line + "\n\n").encode())  # unchanged: keep the chunk as sent
                            else:
                                if segs: emit(d, segs)
                                if finish is not None or "timings" in d or "usage" in d:
                                    d2 = json.loads(payload); d2["choices"][0]["delta"] = {}  # keep finish/timings/usage
                                    self.wfile.write(sse(d2))
                        elif cont is not None and not other:
                            now = time.time()
                            toks = count_text_tokens(cont)
                            with LOCK:
                                STATS["last_tok"] = now
                                if STATS["ttft"] is None:
                                    STATS["ttft"] = now - STATS["turn_start"]
                                # classify by splitter phase BEFORE feeding
                                if sp.phase == "think":
                                    if STATS["think_start"] is None: STATS["think_start"] = now
                                    STATS["reason_tokens"] += toks
                                    STATS["phase"] = "thinking"
                                else:
                                    if STATS["gen_start"] is None: STATS["gen_start"] = now
                                    STATS["content_tokens"] += toks
                                    STATS["phase"] = "generating"
                            segs = sp.feed(cont)
                            # if we just crossed into content, mark gen_start
                            if sp.phase == "content":
                                with LOCK:
                                    if STATS["gen_start"] is None and any(c=="content" for c,_ in segs):
                                        STATS["gen_start"] = time.time()
                            if segs: emit(d, segs)
                            if finish is not None:
                                d2 = json.loads(payload); d2["choices"][0]["delta"] = {}
                                self.wfile.write(sse(d2))
                        else:
                            fseg = [] if pre_split else sp.flush()
                            if fseg: emit(d, fseg)
                            self.wfile.write((line + "\n\n").encode())
                        self.wfile.flush()
            except Exception as e:
                log(f"[stream err] {e}")
            finally:
                keepalive_run[0] = False
                try: conn.close()
                except: pass
                with LOCK:
                    now = time.time()
                    dur = now - STATS["turn_start"]
                    rd, cd = STATS["reason_tokens"], STATS["content_tokens"]   # delta counts (split ratio)
                    tstart, gstart = STATS["think_start"], STATS["gen_start"]
                    ltok = STATS["last_tok"] or now                            # last real token, not post-stream flush
                    # proxy wall-clock phase spans
                    think_wall = (gstart - tstart) if (tstart and gstart) else ((ltok - tstart) if tstart else 0.0)
                    ans_wall   = (ltok - gstart) if gstart else 0.0

                    tg = timings or {}
                    prompt_ms   = tg.get("prompt_ms")
                    prompt_n    = tg.get("prompt_n")
                    prefill_tps = tg.get("prompt_per_second")
                    pred_n      = tg.get("predicted_n")
                    pred_ms     = tg.get("predicted_ms")
                    decode_tps  = tg.get("predicted_per_second")

                    # Handle TensorFold telemetry
                    if tf_data:
                        pred_ms = (tf_data.get("seconds") or 0.0) * 1000.0
                        decode_tps = tf_data.get("tokens_per_second")
                        if tf_data.get("prefill_seconds"):
                            prompt_ms = tf_data["prefill_seconds"] * 1000.0
                        if tf_data.get("time_to_first_token"):
                            STATS["ttft"] = tf_data["time_to_first_token"]
                        if usage and usage.get("completion_tokens") is not None:
                            pred_n = usage["completion_tokens"]
                        elif decode_tps and pred_ms:
                            pred_n = round(decode_tps * (pred_ms / 1000.0))
                    # Prefer usage-based prefill speed = NEW (uncached) tokens / prefill time.
                    # Engines disagree on prompt_per_second under prompt caching: mtplx divides
                    # TOTAL prompt tokens by NEW-token time -> absurd rates on partial cache hits.
                    if usage:
                        if pred_n is None and usage.get("completion_tokens") is not None:
                            pred_n = usage["completion_tokens"]
                        pt = usage.get("prompt_tokens")
                        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                        if pt is not None:
                            prompt_n = pt
                            if cached is not None and prompt_ms:
                                new_prefill = pt - cached
                                prefill_tps = (new_prefill / (prompt_ms / 1000.0)) if (prompt_ms > 0 and new_prefill > 0) else None
                    # sanity clamp: no single-box 27B prefills faster than this; treat as unmeasurable
                    if prefill_tps and prefill_tps > 5000:
                        prefill_tps = None

                    if pred_n and pred_ms and decode_tps:
                        # AUTHORITATIVE: split llama's real totals by phase (tokens by delta ratio, time by wall-clock)
                        denom = (rd + cd) or 1
                        think_tok = round(pred_n * rd / denom)
                        ans_tok   = pred_n - think_tok
                        total_wall = think_wall + ans_wall
                        if total_wall > 0:
                            think_ms = pred_ms * (think_wall / total_wall)
                            ans_ms   = pred_ms - think_ms
                        else:
                            think_ms, ans_ms = 0.0, pred_ms
                        decode_r = round(decode_tps, 1)
                        ttps = split_tps(think_tok, think_ms, decode_r)
                        gtps = split_tps(ans_tok,   ans_ms,   decode_r)
                        rtok_out, ctok_out = think_tok, ans_tok
                    else:
                        # FALLBACK (non-llama upstream, e.g. mtplx): corrected wall-clock, delta counts
                        gtps = tps(cd, ans_wall)
                        ttps = tps(rd, think_wall)
                        decode_tps = gtps
                        pred_n = (rd + cd) or None
                        rtok_out, ctok_out = rd, cd

                    last_obj = {
                        "ttft": round(STATS["ttft"], 2) if STATS["ttft"] else None,
                        "prefill_tps": round(prefill_tps, 1) if prefill_tps else None,
                        "prompt_tokens": prompt_n,
                        "think_tps": ttps,
                        "gen_tps": gtps,
                        "decode_tps": round(decode_tps, 1) if decode_tps else None,
                        "reason_tokens": rtok_out,
                        "content_tokens": ctok_out,
                        "predicted_tokens": pred_n,
                        "duration": round(dur, 1)}
                    if spec_data:
                        last_obj["speculative"] = {
                            "acceptance_rate": round(spec_data.get("acceptance_rate", 0.0), 3),
                            "tokens_per_round": round(spec_data.get("tokens_per_round", 0.0), 1),
                            "rounds": spec_data.get("rounds"),
                            "drafted": spec_data.get("drafted"),
                            "accepted": spec_data.get("accepted"),
                        }
                    STATS["last"] = last_obj
                    STATS["totals"]["turns"] += 1
                    STATS["totals"]["reason_tokens"] += rtok_out
                    STATS["totals"]["content_tokens"] += ctok_out
                    if gtps: STATS["history"].append(round(gtps, 1))
                    if ttps: STATS["think_history"].append(round(ttps, 1))
                    STATS.update(phase="idle", active=False, think_start=None, gen_start=None, last_tok=None)
            return

        # non-streaming / non-chat / errors
        if is_chat:
            with LOCK: STATS.update(phase="idle", active=False)
        raw = resp.read()
        try: conn.close()
        except: pass
        if resp.status != 200:
            try:
                jb = json.loads(body) if body else {}
                keys = {k: jb.get(k) for k in ("model","stream","temperature","top_k","top_p","min_p",
                        "max_tokens","n_predict","response_format","grammar","json_schema") if k in jb}
                log(f"\n##### {time.strftime('%H:%M:%S')} NON-200 status={resp.status} path={self.path} #####")
                log("REQ KEYS: " + json.dumps(keys, default=str)[:1500])
                log("RESP BODY: " + raw.decode('utf-8','replace')[:800])
            except Exception as e:
                log(f"[non200 log err] {e}")
        if is_chat and resp.status == 200:
            try:
                d = json.loads(raw); m = d["choices"][0]["message"]; c = m.get("content") or ""
                idx, found = -1, None
                for cl in CLOSERS:
                    i = c.find(cl)
                    if i != -1 and (idx == -1 or i < idx): idx, found = i, cl
                if idx != -1:
                    raw_rc = c[:idx].strip("\n")
                    for op in OPENERS:
                        if raw_rc.startswith(op):
                            raw_rc = raw_rc[len(op):].lstrip("\n")
                            break
                    m["reasoning_content"] = (m.get("reasoning_content") or "") + raw_rc
                    m["content"] = c[idx+len(found):].lstrip("\n")
                raw = json.dumps(d).encode("utf-8")
            except Exception as e:
                log(f"[nonstream err] {e}")
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
        cors(self)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        try: self.wfile.write(raw)
        except Exception: pass

    def do_POST(self): self._proxy("POST")
    def do_GET(self):  self._proxy("GET")
    def do_OPTIONS(self):
        self.send_response(204); cors(self)
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Content-Length", "0"); self.end_headers()
    def log_message(self, *a): pass

if __name__ == "__main__":
    log("\n########## adapter+stats proxy started %s (model %s, engine %s, listen %d -> %s:%d) ##########"
        % (time.strftime("%Y-%m-%d %H:%M:%S"), MODEL, ENGINE or "auto", LISTEN_PORT, UP_HOST, UP_PORT))
    # The router talks to proxies over loopback; bind there unless told otherwise.
    bind = _get("LISTEN_HOST", "127.0.0.1")
    http.server.ThreadingHTTPServer((bind, LISTEN_PORT), H).serve_forever()
