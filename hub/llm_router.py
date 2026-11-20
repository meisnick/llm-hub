#!/usr/bin/env python3
"""LLM Hub master router - one OpenAI-compatible endpoint in front of many local models.

    client --> :8090 router --> per-model adapter proxy --> engine server

* ``GET  /v1/models``            every configured model (clients with model discovery see them all)
* ``POST /v1/chat/completions``  (and any other ``/v1/*``) - reads ``model``; if it is not the one
  currently loaded, runs the switcher, waits for that model's proxy to report ``up``, then
  streams the request through.
* ``GET  /sysinfo``              RAM/disk/optional temperatures, detected active model, model list,
  session + all-time token counters
* ``GET  /stats``                the active model's live proxy stats
* ``POST /restart``              re-run the switcher for the active model
* ``GET  /`` or ``/dashboard``   the bundled dashboard (dashboard/index.html)

Only one large model fits in memory at a time, so the router owns the "which model is
loaded" decision. It self-heals: the loaded model is whichever proxy reports its upstream
``up``, so out-of-band switches, crashes and router restarts are all picked up within ~2 s.
"""
import argparse, hashlib, http.server, json, os, platform, re, shutil, subprocess, sys
import threading, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubconfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DASHBOARD = os.path.join(hubconfig.REPO, "dashboard", "index.html")

CFG = None
MODELS = {}            # id -> model dict
ACTIVE = None          # id of the model we believe is loaded
SWITCH_LOCK = threading.Lock()
STATS_LOCK = threading.Lock()
_detect = {"t": 0.0, "model": None}


def proxy_url(m, path):
    return f"http://127.0.0.1:{m['proxy_port']}{path}"


def proxy_stats(m, timeout=1.0):
    try:
        return json.loads(urllib.request.urlopen(proxy_url(m, "/stats"), timeout=timeout).read())
    except Exception:
        return None


def detect_active():
    """The loaded model = the one proxy whose upstream server is actually up (cached ~2 s)."""
    global ACTIVE
    now = time.time()
    if _detect["model"] and now - _detect["t"] < 2.0:
        return _detect["model"]
    found = None
    for mid, m in MODELS.items():
        s = proxy_stats(m)
        if s and s.get("up"):
            found = mid
            break
    _detect["t"] = now
    if found:
        _detect["model"] = found
        ACTIVE = found
    return found or ACTIVE


def run_switch(m):
    """Unload everything else and start model m (see llm_switch.py)."""
    cmd = CFG.get("switch_command")
    if cmd:
        argv = [hubconfig.expand(a) for a in cmd] + [m["key"]]
    else:
        argv = [sys.executable, os.path.join(HERE, "llm_switch.py"), "--config", CFG["_path"], m["key"]]
    subprocess.run(argv)


def ensure_loaded(mid):
    """Switch to model `mid` if needed and wait until its proxy reports the server up."""
    global ACTIVE
    m = MODELS[mid]
    with SWITCH_LOCK:  # concurrent requests for different models must not interleave switches
        detect_active()
        if ACTIVE == mid:
            s = proxy_stats(m)
            if s and s.get("up"):
                return True
        run_switch(m)
        ACTIVE = mid
        _detect.update(t=0.0, model=None)
        deadline = time.time() + float(CFG["load_timeout"])
        while time.time() < deadline:
            s = proxy_stats(m)
            if s and s.get("up"):
                _detect.update(t=time.time(), model=mid)
                return True
            time.sleep(1)
    return False


# ---- usage tracking: per-session (resets on a new chat) + persistent all-time ----
def _usage_file():
    return os.path.join(CFG["state_dir"], "usage.json")


def _load_alltime():
    try:
        with open(_usage_file()) as f:
            d = json.load(f)
        return {"gen_tokens": int(d.get("gen_tokens", 0)), "turns": int(d.get("turns", 0))}
    except Exception:
        return {"gen_tokens": 0, "turns": 0}


ALLTIME = {"gen_tokens": 0, "turns": 0}
SESSION = {"gen_tokens": 0, "turns": 0, "started": time.time()}
_sess = {"fp": None}


def _first_user_fp(messages):
    """Stable fingerprint of a conversation = hash of its first user message.

    Constant while the conversation grows (and across an agent's tool-loop sub-requests);
    changes only when a genuinely different chat starts. Only the hash is kept."""
    for m in (messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if not isinstance(c, str):
                c = json.dumps(c, sort_keys=True, default=str)
            return hashlib.sha1((c or "").encode("utf-8", "replace")).hexdigest()[:12]
    return None


def _save_alltime():
    try:
        tmp = _usage_file() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(ALLTIME, f)
        os.replace(tmp, _usage_file())
    except Exception:
        pass


def maybe_new_session(messages):
    fp = _first_user_fp(messages)
    with STATS_LOCK:
        if fp and fp != _sess.get("fp"):
            SESSION.update(gen_tokens=0, turns=0, started=time.time())
            _sess["fp"] = fp
            return True
    return False


def record_turn(gen_tokens):
    if not gen_tokens or gen_tokens <= 0:
        return
    with STATS_LOCK:
        SESSION["gen_tokens"] += gen_tokens
        SESSION["turns"] += 1
        ALLTIME["gen_tokens"] += gen_tokens
        ALLTIME["turns"] += 1
    _save_alltime()


def extract_gen_tokens(tail_bytes):
    """Generated tokens (INCLUDING thinking) for a turn, from the tail of the SSE stream.

    Every candidate is a cumulative total, so take the max across all of them - robust even
    when a very large final stats chunk is partly cut off by the tail window."""
    gen = 0
    text = tail_bytes.decode("utf-8", "replace")
    if not text.lstrip().startswith("data:"):  # non-streaming: one JSON body
        try:
            d = json.loads(text)
            for cand in ((d.get("timings") or {}).get("predicted_n"),
                         (d.get("usage") or {}).get("completion_tokens")):
                if isinstance(cand, int) and cand > gen:
                    gen = cand
        except Exception:
            pass
        return gen
    try:
        for line in text.split("\n"):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p in ("[DONE]", ""):
                continue
            try:
                d = json.loads(p)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            for cand in ((d.get("timings") or {}).get("predicted_n"),
                         (d.get("usage") or {}).get("completion_tokens"),
                         (d.get("mtplx_progress") or {}).get("completion_tokens")):
                if isinstance(cand, int) and cand > gen:
                    gen = cand
    except Exception:
        pass
    return gen


# ---- host metrics (best effort, never fatal) ----
def _mem_used_mb():
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            out = subprocess.check_output(["top", "-l", "1", "-n", "0"], timeout=5).decode()
            line = next(l for l in out.splitlines() if l.startswith("PhysMem:"))
            used = re.search(r"(\d+)([MG]) used", line)
            free = re.search(r"(\d+)([MG]) unused", line)
            mb = lambda g: float(g.group(1)) * (1024 if g.group(2) == "G" else 1)
            return mb(used), (mb(free) if free else None)
        if sysname == "Linux":
            info = {}
            with open("/proc/meminfo") as f:
                for l in f:
                    k, v = l.split(":", 1)
                    info[k] = int(v.split()[0]) / 1024.0
            return info["MemTotal"] - info.get("MemAvailable", 0), info.get("MemAvailable")
    except Exception:
        pass
    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / 2**20, vm.available / 2**20
    except Exception:
        return None, None


def _temps():
    """Optional: `temp_command` prints lines like 'CPU 61.2' / 'SSD 40' / 'BAT 31'."""
    cmd = CFG.get("temp_command")
    out = {"cpu_temp": None, "ssd_temp": None, "bat_temp": None}
    if not cmd:
        return out
    try:
        text = subprocess.check_output([hubconfig.expand(a) for a in cmd], timeout=5).decode()
    except Exception:
        return out
    buckets = {"cpu_temp": [], "ssd_temp": [], "bat_temp": []}
    pats = CFG.get("temp_patterns") or {"cpu_temp": "CPU", "ssd_temp": "SSD|NAND", "bat_temp": "BAT|battery"}
    for line in text.splitlines():
        v = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°?C)?\s*$", line.strip())
        if not v:
            continue
        for k, pat in pats.items():
            if re.search(pat, line, re.I):
                buckets[k].append(float(v.group(1)))
    for k, vals in buckets.items():
        if vals:
            out[k] = round(sum(vals) / len(vals))
    return out


def sysinfo():
    used, free = _mem_used_mb()
    du = shutil.disk_usage(os.path.expanduser("~"))
    with STATS_LOCK:
        sess, allt = dict(SESSION), dict(ALLTIME)
    active = detect_active()
    if active:  # fold the in-flight turn into the counters so the dashboard ticks live
        s = proxy_stats(MODELS[active], timeout=0.5)
        if s and s.get("active"):
            t = s.get("turn") or {}
            live = (t.get("content_tokens") or 0) + (t.get("reason_tokens") or 0)
            sess["gen_tokens"] += live
            allt["gen_tokens"] += live
    gb = lambda mb: f"{mb / 1024:.1f} GB" if mb is not None else "n/a"
    data = {
        "active_model": active,
        "models": [{"id": m["id"], "key": m["key"], "engine": m["engine"], "size": m["size"],
                    "params": m["params"]} for m in CFG["models"]],
        "session": sess,
        "alltime": allt,
        "memory": f"{gb(used)} used / {gb(free)} free",
        "memory_used_mb": used,
        "disk_free": f"{du.free / 2**30:.0f} GB free",
        "disk_free_gb": round(du.free / 2**30, 1),
        "host": platform.node() if CFG.get("show_hostname") else None,
    }
    data.update(_temps())
    return data


def cors(h):
    h.send_header("Access-Control-Allow-Origin", "*")


class RouterHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); cors(self)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204); cors(self)
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": m["id"], "object": "model", "created": 1720000000, "owned_by": "llmhub"}
                for m in CFG["models"]]})
        elif path == "/sysinfo":
            try:
                self._json(200, sysinfo())
            except Exception as e:
                self._json(200, {"error": str(e)})
        elif path == "/stats":
            m = MODELS.get(detect_active() or "")
            s = proxy_stats(m, timeout=2) if m else None
            if s is None:
                self._json(502, {"error": "stats upstream failed"})
            else:
                self._json(200, s)
        elif path in ("/", "/dashboard", "/index.html"):
            try:
                with open(DASHBOARD, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                self._json(404, {"error": "dashboard not found"})
        elif path == "/health":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        detect_active()
        if self.path == "/restart":
            if ACTIVE in MODELS:
                run_switch(MODELS[ACTIVE])
            self._json(200, {"status": "restarted", "model": ACTIVE})
            return

        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        is_chat = self.path.endswith("/chat/completions")
        is_aux = False
        j = None
        try:
            j = json.loads(body) if body else None
        except Exception:
            j = None

        if is_chat and isinstance(j, dict):
            # auxiliary calls (title generation etc. use response_format) don't count as turns
            is_aux = bool(j.get("response_format"))
            msgs = j.get("messages") or []
            reset = maybe_new_session(msgs) if not is_aux else False
            if CFG.get("session_log"):  # opt-in diagnostics: metadata only, never message text
                try:
                    with open(os.path.join(CFG["state_dir"], "sessions.log"), "a") as lf:
                        lf.write(json.dumps({
                            "t": time.strftime("%H:%M:%S"), "n": len(msgs), "aux": is_aux,
                            "reset": reset, "fp": _first_user_fp(msgs),
                            "roles": [m.get("role") for m in msgs if isinstance(m, dict)][:10],
                            "has_tools": bool(j.get("tools"))}) + "\n")
                except Exception:
                    pass

        target = None
        if self.path.startswith("/v1/") and isinstance(j, dict) and j.get("model") in MODELS:
            mid = j["model"]
            if not ensure_loaded(mid):
                self._json(503, {"error": {"message": f"model {mid} did not come up within "
                                                      f"{CFG['load_timeout']}s", "type": "llmhub_load_timeout"}})
                return
            target = MODELS[mid]
        if target is None:
            target = MODELS.get(ACTIVE) or CFG["models"][0]

        req = urllib.request.Request(proxy_url(target, self.path), data=body, method="POST")
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length"):
                req.add_header(k, v)
        try:
            resp = urllib.request.urlopen(req, timeout=3600)
            self.send_response(resp.status); cors(self)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "connection", "access-control-allow-origin"):
                    self.send_header(k, v)
            self.end_headers()
            tail = b""
            while True:
                # read1 returns whatever has arrived; read(n) would block for n bytes and turn
                # a smooth token stream into bursts.
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                tail = (tail + chunk)[-262144:]  # big enough for large final stats chunks
            if is_chat and not is_aux:
                record_turn(extract_gen_tokens(tail))
        except urllib.error.HTTPError as e:
            self.send_response(e.code); cors(self)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "connection", "access-control-allow-origin"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            try:
                self._json(502, {"error": {"message": str(e), "type": "llmhub_upstream"}})
            except Exception:
                pass


def main():
    global CFG, MODELS, ACTIVE, ALLTIME
    ap = argparse.ArgumentParser(description="LLM Hub master router")
    ap.add_argument("--config", help="path to hub.json")
    a = ap.parse_args()
    CFG = hubconfig.load(a.config)
    os.makedirs(CFG["state_dir"], exist_ok=True)
    MODELS = hubconfig.by_id(CFG)
    ALLTIME = _load_alltime()
    ACTIVE = detect_active()
    if not ACTIVE and CFG.get("default_model"):
        dm = hubconfig.find(CFG, CFG["default_model"])
        if dm:
            threading.Thread(target=ensure_loaded, args=(dm["id"],), daemon=True).start()
    print(f"LLM Hub router on {CFG['listen_host']}:{CFG['listen_port']} - "
          f"{len(MODELS)} models, active={ACTIVE}, config={CFG['_path']}", flush=True)
    http.server.ThreadingHTTPServer((CFG["listen_host"], int(CFG["listen_port"])),
                                    RouterHandler).serve_forever()


if __name__ == "__main__":
    main()
