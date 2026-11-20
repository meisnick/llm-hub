#!/usr/bin/env python3
"""Swap the loaded model: stop every other engine server, then (re)start the requested one.

    llm_switch.py <key|id>        load a model (unloads all others)
    llm_switch.py --stop          unload everything
    llm_switch.py --list          show models and which server ports are listening

macOS/launchd. Each model's *server* runs as a launchd agent whose plist lives in
``server_plist_dir`` (NOT ~/Library/LaunchAgents) so nothing loads at login and only one
big model is ever resident. ``install_launchd.py`` writes those plists.

Some engines spawn a child that outlives its launchd job (mtplx's server process does), so
after the bootouts anything still listening on another model's server port is a stray: it
gets SIGTERM, then SIGKILL after 20 s. Without this, two models can end up loaded and the
router's "first proxy reporting up" detection flaps between them.
"""
import argparse, os, signal, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hubconfig  # noqa: E402


def sh(argv):
    return subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def listeners(port):
    try:
        out = subprocess.check_output(["lsof", "-nP", f"-tiTCP:{port}", "-sTCP:LISTEN"],
                                      stderr=subprocess.DEVNULL).decode()
        return {int(p) for p in out.split()}
    except Exception:
        return set()


def kill_strays(ports, grace=20):
    pids = set().union(*(listeners(p) for p in ports)) if ports else set()
    if not pids:
        return
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(grace):
        if not set().union(*(listeners(p) for p in ports)):
            return
        time.sleep(1)
    for p in set().union(*(listeners(p) for p in ports)):
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="model key or id")
    ap.add_argument("--config")
    ap.add_argument("--stop", action="store_true", help="unload every model")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cfg = hubconfig.load(a.config)
    uid = os.getuid()
    dom = f"gui/{uid}"

    if a.list or not (a.model or a.stop):
        for m in cfg["models"]:
            up = "LISTENING" if listeners(m["server_port"]) else "-"
            print(f"  {m['key']:<12} {m['id']:<32} {m['engine']:<11} :{m['server_port']} {up}")
        if not (a.model or a.stop):
            print("\nusage: llm_switch.py <key>|--stop")
        return

    target = None
    if not a.stop:
        target = hubconfig.find(cfg, a.model)
        if not target:
            raise SystemExit(f"unknown model '{a.model}'. keys: "
                             + ", ".join(m['key'] for m in cfg["models"]))

    for m in cfg["models"]:
        if m is not target:
            sh(["launchctl", "bootout", f"{dom}/{hubconfig.label(cfg, m, 'server')}"])
    other_ports = sorted({m["server_port"] for m in cfg["models"]
                          if target is None or m["server_port"] != target["server_port"]})
    kill_strays(other_ports)

    if target is None:
        print("-> all models stopped")
        return
    plist = os.path.join(cfg["server_plist_dir"], hubconfig.label(cfg, target, "server") + ".plist")
    if not os.path.isfile(plist):
        raise SystemExit(f"missing {plist} - run install_launchd.py first")
    sh(["launchctl", "bootstrap", dom, plist])  # harmless "already loaded" error is ignored
    sh(["launchctl", "kickstart", "-k", f"{dom}/{hubconfig.label(cfg, target, 'server')}"])
    print(f"-> {target['id']} on :{target['server_port']} (others stopped)")


if __name__ == "__main__":
    main()
