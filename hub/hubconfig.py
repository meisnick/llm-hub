"""Shared configuration loader for LLM Hub.

Every component (router, proxy, switcher, installer) reads the same JSON file so a model
is described exactly once. Lookup order for the config path:

  1. an explicit path passed by the caller (``--config``)
  2. the ``LLMHUB_CONFIG`` environment variable
  3. ``~/.config/llmhub/hub.json``
  4. ``config/hub.json`` next to this repository checkout

See ``config/hub.example.json`` for a fully commented example and docs/CONFIGURATION.md
for every key.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DEFAULTS = {
    "listen_host": "0.0.0.0",
    "listen_port": 8090,
    "state_dir": "~/.local/state/llmhub",
    "launchd_prefix": "local.llmhub",
    "server_plist_dir": "~/.config/llmhub/launchd",
    "python": "/usr/bin/python3",
    "load_timeout": 180,
    "default_model": None,
    "temp_command": None,
    "session_log": False,
}

ENGINES = ("llama.cpp", "mtplx", "tensorfold", "openai")


def expand(s, **extra):
    """Expand ~, $VARS and {home}/{port}/{repo} placeholders in a string."""
    if not isinstance(s, str):
        return s
    vals = {"home": os.path.expanduser("~"), "repo": REPO}
    vals.update({k: v for k, v in extra.items() if v is not None})
    for k, v in vals.items():
        s = s.replace("{" + k + "}", str(v))
    return os.path.expandvars(os.path.expanduser(s))


def candidate_paths(explicit=None):
    if explicit:
        yield explicit
    env = os.environ.get("LLMHUB_CONFIG")
    if env:
        yield env
    yield os.path.expanduser("~/.config/llmhub/hub.json")
    yield os.path.join(REPO, "config", "hub.json")


def load(explicit=None):
    """Return the parsed config with defaults applied and per-model fields validated."""
    path = None
    for p in candidate_paths(explicit):
        if p and os.path.isfile(os.path.expanduser(p)):
            path = os.path.abspath(os.path.expanduser(p))
            break
    if not path:
        raise SystemExit(
            "LLM Hub: no config found. Copy config/hub.example.json to "
            "~/.config/llmhub/hub.json (or set LLMHUB_CONFIG) and edit it.")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in raw.items() if not k.startswith("_")})
    cfg["_path"] = path
    cfg["state_dir"] = expand(cfg["state_dir"])
    cfg["server_plist_dir"] = expand(cfg["server_plist_dir"])

    models = []
    seen_keys, seen_ports = set(), set()
    for m in raw.get("models", []):
        m = {k: v for k, v in m.items() if not k.startswith("_")}
        for req in ("id", "key", "engine", "server_port", "proxy_port", "command"):
            if req not in m:
                raise SystemExit(f"LLM Hub config: model {m.get('id', '?')} is missing '{req}'")
        if m["engine"] not in ENGINES:
            raise SystemExit(f"LLM Hub config: model {m['id']}: engine must be one of {ENGINES}")
        if m["key"] in seen_keys:
            raise SystemExit(f"LLM Hub config: duplicate model key '{m['key']}'")
        if m["proxy_port"] in seen_ports:
            raise SystemExit(f"LLM Hub config: duplicate proxy_port {m['proxy_port']}")
        seen_keys.add(m["key"])
        seen_ports.add(m["proxy_port"])
        m.setdefault("size", "")
        m.setdefault("params", "")
        m.setdefault("env", {})
        m.setdefault("cwd", None)
        m.setdefault("upstream_host", "127.0.0.1")
        models.append(m)
    if not models:
        raise SystemExit("LLM Hub config: 'models' is empty")
    cfg["models"] = models
    return cfg


def by_id(cfg):
    return {m["id"]: m for m in cfg["models"]}


def by_key(cfg):
    return {m["key"]: m for m in cfg["models"]}


def find(cfg, name):
    """Look a model up by id or key (case-insensitive)."""
    n = (name or "").lower()
    for m in cfg["models"]:
        if m["id"].lower() == n or m["key"].lower() == n:
            return m
    return None


def label(cfg, m, role):
    """launchd label for a model's server/proxy agent, or the router/dashboard."""
    if m is None:
        return f"{cfg['launchd_prefix']}.{role}"
    return f"{cfg['launchd_prefix']}.{m['key']}-{role}"


def server_argv(m):
    return [expand(a, port=m["server_port"]) for a in m["command"]]


def server_env(m):
    return {k: expand(v, port=m["server_port"]) for k, v in (m.get("env") or {}).items()}
