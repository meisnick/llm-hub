#!/usr/bin/env python3
"""Serve the dashboard with synthetic, animated data - no models, no Mac required.

    python3 tools/demo_server.py [--port 8090]
    open http://localhost:8090/?demo

Used to preview dashboard changes and to produce docs/images/dashboard.png. The model list
comes from config/hub.example.json; every number is generated.
"""
import argparse, http.server, json, math, os, random, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASH = os.path.join(REPO, "dashboard", "index.html")
with open(os.path.join(REPO, "config", "hub.example.json"), encoding="utf-8") as f:
    MODELS = [{"id": m["id"], "key": m["key"], "engine": m["engine"], "size": m["size"],
               "params": m["params"]} for m in json.load(f)["models"]]
ACTIVE = MODELS[0]["id"]
T0 = time.time()
rng = random.Random(7)
HIST = [round(38 + 9 * math.sin(i / 3.0) + rng.uniform(-3, 3), 1) for i in range(60)]
THIST = [round(h + rng.uniform(-2, 2), 1) for h in HIST]
TURN_LEN = 14.0  # seconds per synthetic turn


def stats():
    t = (time.time() - T0) % (TURN_LEN + 3)
    totals = {"turns": 128, "reason_tokens": 211_406, "content_tokens": 96_212}
    last = {"ttft": 0.42, "prefill_tps": 412.6, "prompt_tokens": 6_812, "think_tps": 46.8,
            "gen_tps": 47.9, "decode_tps": 47.5, "reason_tokens": 418, "content_tokens": 352,
            "predicted_tokens": 770, "duration": 16.6,
            "speculative": {"acceptance_rate": 0.74, "tokens_per_round": 3.6}}
    base = {"model": ACTIVE, "up": True, "totals": totals, "last": last,
            "history": HIST, "think_history": THIST}
    if t >= TURN_LEN:
        return dict(base, phase="idle", active=False, turn=None)
    think_end = TURN_LEN * 0.55
    rate = 46 + 4 * math.sin(t)
    if t < 0.4:
        phase, rt, ct = "prefill", 0, 0
    elif t < think_end:
        phase, rt, ct = "thinking", int((t - 0.4) * rate), 0
    else:
        phase, rt, ct = "generating", int((think_end - 0.4) * rate), int((t - think_end) * rate)
    turn = {"elapsed": round(t, 1), "ttft": 0.41 if t > 0.4 else None, "reason_tokens": rt,
            "content_tokens": ct, "think_tps": round(rate, 1),
            "gen_tps": round(rate + 1.2, 1) if ct else None, "tps": round(rate, 1)}
    return dict(base, phase=phase, active=True, turn=turn)


def sysinfo():
    t = time.time() - T0
    return {"active_model": ACTIVE, "models": MODELS,
            "session": {"turns": 14, "gen_tokens": 18_734, "started": T0 - 47 * 60},
            "alltime": {"turns": 2_961, "gen_tokens": 4_382_115},
            "memory": "41.3 GB used / 22.4 GB free", "memory_used_mb": 42_291 + 600 * math.sin(t / 5),
            "disk_free": "612 GB free", "disk_free_gb": 612.0,
            "cpu_temp": int(68 + 6 * math.sin(t / 4)), "ssd_temp": int(41 + math.sin(t / 9)),
            "bat_temp": 31}


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p in ("/", "/dashboard", "/index.html"):
            with open(DASH, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif p == "/stats":
            self._send(200, json.dumps(stats()).encode(), "application/json")
        elif p == "/sysinfo":
            self._send(200, json.dumps(sysinfo()).encode(), "application/json")
        else:
            self._send(404, b"{}", "application/json")

    def do_POST(self):
        self._send(200, b'{"status":"restarted"}', "application/json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    print(f"demo dashboard: http://localhost:{a.port}/?demo")
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
