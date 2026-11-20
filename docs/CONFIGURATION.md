# Configuration

Everything is described once in `hub.json`. The router, proxies, switcher and installer all
read the same file. Lookup order:

1. `--config <path>` on the command line
2. `$LLMHUB_CONFIG`
3. `~/.config/llmhub/hub.json`
4. `config/hub.json` inside the checkout (gitignored)

Start from [`config/hub.example.json`](../config/hub.example.json). Keys beginning with `_`
are ignored, so you can use them for comments.

## Top-level keys

| Key | Default | Meaning |
|---|---|---|
| `listen_host` | `0.0.0.0` | router bind address. Use `127.0.0.1` for local-only |
| `listen_port` | `8090` | router port (the dashboard assumes 8090 unless you pass `?hub=`) |
| `dashboard_port` | `null` | optional extra port that serves the same dashboard/API (the page still talks to `listen_port`) |
| `state_dir` | `~/.local/state/llmhub` | logs, `usage.json` (all-time counters) |
| `launchd_prefix` | `local.llmhub` | launchd label prefix: `<prefix>.router`, `<prefix>.<key>-proxy`, `<prefix>.<key>-server` |
| `server_plist_dir` | `~/.config/llmhub/launchd` | where on-demand server plists are written (not auto-loaded) |
| `python` | `/usr/bin/python3` | interpreter used in the generated plists |
| `load_timeout` | `180` | seconds the router waits for a model to come up after a switch |
| `default_model` | `null` | model id/key to load when the router starts and nothing is loaded |
| `switch_command` | built-in | argv prefix replacing `llm_switch.py`; the model key is appended |
| `temp_command` | `null` | optional argv printing temperatures (see below) |
| `temp_patterns` | CPU / SSD\|NAND / BAT\|battery | regexes mapping output lines to `cpu_temp`, `ssd_temp`, `bat_temp` |
| `session_log` | `false` | append per-request metadata (never message text) to `state_dir/sessions.log` |
| `show_hostname` | `false` | include the machine's hostname in `/sysinfo` |

## Model entries (`models[]`)

| Key | Required | Meaning |
|---|---|---|
| `id` | ✓ | the name clients send as `model` and see in `/v1/models` |
| `key` | ✓ | short handle for `llm_switch.py` and launchd labels (`[a-z0-9-]`) |
| `engine` | ✓ | `llama.cpp`, `mtplx`, `tensorfold` or `openai`: selects proxy behavior |
| `server_port` | ✓ | port the engine listens on |
| `proxy_port` | ✓ | port this model's proxy listens on (unique) |
| `command` | ✓ | argv that starts the engine in the foreground |
| `env` | | extra environment for the engine (e.g. `DYLD_LIBRARY_PATH`) |
| `cwd` | | working directory for the engine |
| `upstream_host` | | where the proxy finds the engine (default `127.0.0.1`) |
| `params`, `size` | | labels shown on the dashboard |

### Placeholders

In `command`, `env`, `cwd` and `temp_command`:

- `~` and `{home}` → your home directory
- `{port}` → this model's `server_port`
- `{repo}` → the checkout directory
- `$VAR` → environment variables

Two models may share a `server_port` (only one runs at a time) but never a `proxy_port`.

## Adding a model

1. Download the weights and confirm the engine command works by hand in a terminal.
2. Add an entry to `models[]` with new, unused ports. Convention: server 80xx, proxy 80xx+7.
3. `python3 hub/install_launchd.py`. This rewrites all plists and reloads the router and
   proxies.
4. `python3 hub/llm_switch.py <key>`, then
   `python3 tools/check_hub.py --hub http://localhost:8090 --model <id>`.
5. Clients that discover models pick it up automatically. Tell clients that cache context
   sizes about the new model's context window (see [CLIENTS.md](CLIENTS.md)).

## Temperatures (optional)

The dashboard's CPU/SSD/battery tiles appear only when `temp_command` is set. The command
should print one reading per line. The value is the first number followed by `C` or `°C`
(trailing text is fine), or else the last number on the line.
The default patterns sort lines into buckets and average each bucket:

```
CPU die 1   61.2 C
NAND        40.1 C
battery     31.0 C
```

Any SMC reader works, for example a small CLI built on IOKit, or a third-party tool's
command-line mode. Adjust `temp_patterns` to match its labels. No sensor tool ships with
this repo.

## Proxy environment variables

When run with `--model <key>` the proxy takes everything from `hub.json`. It can also be
run standalone from the environment (the legacy `K2_*` names are accepted too):

| Variable | Meaning |
|---|---|
| `LLMHUB_UPSTREAM_HOST` / `LLMHUB_UPSTREAM_PORT` | engine address |
| `LLMHUB_LISTEN_PORT` / `LLMHUB_LISTEN_HOST` | proxy bind (host default `127.0.0.1`) |
| `LLMHUB_MODEL` | model id reported in `/stats` |
| `LLMHUB_ENGINE` | `llama.cpp` / `mtplx` / `tensorfold` / `openai` |
| `LLMHUB_LOG` | log file |
| `LLMHUB_DUMP_BODY=1` | debug: save the last request body (**contains prompt text**) |
