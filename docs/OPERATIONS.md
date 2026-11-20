# Operations

Commands assume the checkout is at `~/llm-hub` and the default `launchd_prefix`
(`local.llmhub`).

## Status

```bash
curl -s localhost:8090/sysinfo | python3 -m json.tool     # active model, memory, counters
curl -s localhost:8090/stats   | python3 -m json.tool     # live/last-turn speeds
python3 ~/llm-hub/hub/llm_switch.py --list                # which engine ports are listening
launchctl list | grep llmhub                              # agents and PIDs
```

## Switching models

The router switches automatically on request. By hand:

```bash
python3 ~/llm-hub/hub/llm_switch.py nemotron     # load one (unloads the rest)
python3 ~/llm-hub/hub/llm_switch.py --stop       # free all model memory
curl -X POST localhost:8090/restart              # restart the active model
```

A handy shell alias: `alias llm-switch='python3 ~/llm-hub/hub/llm_switch.py'`.

## Logs

Everything is in `state_dir` (default `~/.local/state/llmhub/`):

| File | Contents |
|---|---|
| `router.log` | router stdout/stderr |
| `<key>-proxy.log` | proxy events: reasoning mapping, request rewrites, non-200 upstream replies (request keys only, never prompt text) |
| `<key>-server.log` | engine output (load progress, errors) |
| `usage.json` | persisted all-time counters |

## Restarting services

```bash
U=$(id -u)
launchctl kickstart -k gui/$U/local.llmhub.router
launchctl kickstart -k gui/$U/local.llmhub.<key>-proxy
```

`kickstart -k` has been seen to occasionally not restart an agent, leaving a proxy serving
old code. Verify with a new "proxy started" line in its log or a changed PID
(`lsof -nP -iTCP:<port> -sTCP:LISTEN`).

After editing `hub.json` or updating the code, re-run `python3 hub/install_launchd.py`. It
rewrites every plist and reloads the router and proxies. The running engine is left alone.

## Upgrading engines

- **llama.cpp:** build into a new directory, point one model's `command` at it, then test
  with `check_hub.py` before moving the rest. Keep the old build for rollback.
- **mtplx (Homebrew):** after `brew upgrade mtplx`, reinstall `llguidance` into the new
  versioned venv or structured output breaks.
- **TensorFold:** see the pre-M5 version note in [ENGINES.md](ENGINES.md).

Run `tools/check_hub.py` against each model after any upgrade. It covers chat, streaming
split, tools, `json_schema`, and reasoning off.

## Metal wired-memory limit

By default macOS caps how much unified memory the GPU may wire (roughly 75%). For large
models with long context, raise it:

```bash
sudo sysctl iogpu.wired_limit_mb=57344          # e.g. 56 GB of a 64 GB machine
```

This setting is **lost on reboot**. To persist it:

```bash
python3 hub/install_launchd.py --wired-limit 57344   # writes the LaunchDaemon and prints sudo steps
```

Leave 6–8 GB for macOS itself, or the system will start swapping hard.

## Reboots and login

The router and proxies are **LaunchAgents**, so they start when your user logs in. On a
headless Mac, enable automatic login or nothing listens after a reboot. Engine servers
never auto-start (see [ARCHITECTURE.md](ARCHITECTURE.md#processes)). Set `default_model`
if you want one loaded as soon as the router comes up.

## Uninstall

```bash
python3 hub/llm_switch.py --stop
python3 hub/install_launchd.py --uninstall
rm -rf ~/.config/llmhub ~/.local/state/llmhub
```
