#!/usr/bin/env python3
"""Generate and (re)load the launchd agents for LLM Hub on macOS.

    install_launchd.py              write plists + load router/proxies
    install_launchd.py --dry-run    print what would be written
    install_launchd.py --uninstall  boot everything out and remove the plists
    install_launchd.py --wired-limit 57344   also write a LaunchDaemon that raises the Metal
                                             wired-memory limit at boot (needs sudo to install)

Layout produced:
  ~/Library/LaunchAgents/<prefix>.router.plist         RunAtLoad + KeepAlive (always up)
  ~/Library/LaunchAgents/<prefix>.<key>-proxy.plist    RunAtLoad + KeepAlive (always up; a proxy
                                                       just reports up:false while its server is off)
  <server_plist_dir>/<prefix>.<key>-server.plist       NOT in LaunchAgents: started only by
                                                       llm_switch.py, so a login never loads every
                                                       model at once.
"""
import argparse, os, plistlib, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubconfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = os.path.expanduser("~/Library/LaunchAgents")


def _loaded(dom, label):
    return subprocess.run(["launchctl", "print", f"{dom}/{label}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def reload_agent(dom, label, path):
    """bootout returns before the old job has exited; bootstrapping right away then fails
    (EIO) and leaves the agent unloaded. Wait for it to be gone, then bootstrap with retries."""
    subprocess.run(["launchctl", "bootout", f"{dom}/{label}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        if not _loaded(dom, label):
            break
        time.sleep(0.2)
    for attempt in range(5):
        if subprocess.run(["launchctl", "bootstrap", dom, path],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0 \
                or _loaded(dom, label):
            return
        time.sleep(1)
    print(f"WARNING: could not load {label}; run: launchctl bootstrap {dom} {path}")


def plist(label, argv, log, env=None, cwd=None, interactive=False):
    d = {"Label": label, "ProgramArguments": argv, "RunAtLoad": True, "KeepAlive": True,
         "StandardOutPath": log, "StandardErrorPath": log}
    if env:
        d["EnvironmentVariables"] = env
    if cwd:
        d["WorkingDirectory"] = cwd
    if interactive:  # don't let macOS treat a GPU-heavy server as background work
        d["ProcessType"] = "Interactive"
    return d


def build(cfg):
    py, state = cfg["python"], cfg["state_dir"]
    conf = ["--config", cfg["_path"]]
    out = []  # (path, dict, autoload)
    lab = hubconfig.label(cfg, None, "router")
    out.append((os.path.join(AGENTS, lab + ".plist"),
                plist(lab, [py, os.path.join(HERE, "llm_router.py")] + conf,
                      os.path.join(state, "router.log")), True))
    for m in cfg["models"]:
        lab = hubconfig.label(cfg, m, "proxy")
        out.append((os.path.join(AGENTS, lab + ".plist"),
                    plist(lab, [py, os.path.join(HERE, "llm_proxy.py"), "--model", m["key"]] + conf,
                          os.path.join(state, f"{m['key']}-proxy.stdout.log")), True))
        lab = hubconfig.label(cfg, m, "server")
        env = hubconfig.server_env(m)
        env.setdefault("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")
        out.append((os.path.join(cfg["server_plist_dir"], lab + ".plist"),
                    plist(lab, hubconfig.server_argv(m), os.path.join(state, f"{m['key']}-server.log"),
                          env=env, cwd=hubconfig.expand(m["cwd"]) if m.get("cwd") else None,
                          interactive=True), False))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--wired-limit", type=int, metavar="MB",
                    help="write <prefix>.wiredlimit.plist (LaunchDaemon) setting iogpu.wired_limit_mb")
    a = ap.parse_args()
    cfg = hubconfig.load(a.config)
    dom = f"gui/{os.getuid()}" if hasattr(os, "getuid") else "gui/501"
    items = build(cfg)

    if a.dry_run:
        for path, d, auto in items:
            print(f"# {path}  ({'autoload' if auto else 'on-demand'})")
            print(plistlib.dumps(d).decode())
        return

    if a.uninstall:
        for path, d, _ in items:
            subprocess.run(["launchctl", "bootout", f"{dom}/{d['Label']}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(path):
                os.remove(path)
                print("removed", path)
        return

    os.makedirs(AGENTS, exist_ok=True)
    os.makedirs(cfg["server_plist_dir"], exist_ok=True)
    os.makedirs(cfg["state_dir"], exist_ok=True)
    for path, d, auto in items:
        with open(path, "wb") as f:
            plistlib.dump(d, f)
        print(("loaded   " if auto else "written  ") + path)
        if auto:  # reload so code/config changes take effect
            reload_agent(dom, d["Label"], path)

    if a.wired_limit:
        lab = hubconfig.label(cfg, None, "wiredlimit")
        path = os.path.join(cfg["server_plist_dir"], lab + ".plist")
        with open(path, "wb") as f:
            plistlib.dump({"Label": lab, "RunAtLoad": True, "ProgramArguments":
                           ["/usr/sbin/sysctl", "-w", f"iogpu.wired_limit_mb={a.wired_limit}"]}, f)
        print(f"\nwired-limit daemon written to {path}. Install it (needs sudo):\n"
              f"  sudo cp {path} /Library/LaunchDaemons/ && "
              f"sudo chown root:wheel /Library/LaunchDaemons/{lab}.plist && "
              f"sudo launchctl bootstrap system /Library/LaunchDaemons/{lab}.plist")

    print(f"\nRouter: http://<this-mac>:{cfg['listen_port']}/  (dashboard) and /v1 (API)")
    if cfg.get("default_model"):
        print(f"Default model {cfg['default_model']} loads when the router starts with nothing loaded.")
    else:
        print("Nothing is loaded yet - the first request (or llm_switch.py <key>) loads a model.")


if __name__ == "__main__":
    main()
