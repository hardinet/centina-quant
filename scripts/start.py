"""Start CENTINA agent and the official Streamlit dashboard.

Usage:
    python scripts/start.py --mode PAPER
    python scripts/start.py --mode ADVISOR --no-dashboard
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
DEFAULT_DASHBOARD = "dashboard/app.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CENTINA launcher")
    parser.add_argument("--mode", choices=["PAPER", "ADVISOR", "SEMI", "AUTO"], default="PAPER")
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--ui", choices=["streamlit", "web"], default="streamlit")
    parser.add_argument("--dashboard", default=DEFAULT_DASHBOARD)
    parser.add_argument("--port", default="8501")
    parser.add_argument("--no-dashboard", action="store_true", help="Do not launch the graphical UI")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def start_dashboard(ui: str, script: str, port: str) -> subprocess.Popen:
    LOGS_DIR.mkdir(exist_ok=True)
    log_name = "web_dashboard.log" if ui == "web" else "streamlit.log"
    dash_log = open(LOGS_DIR / log_name, "a", encoding="utf-8")
    if ui == "web":
        dash_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "dashboard.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            port,
        ]
    else:
        dash_cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            script,
            "--server.port",
            port,
            "--server.headless",
            "true",
            "--theme.base",
            "dark",
            "--theme.backgroundColor",
            "#0B0E11",
            "--theme.primaryColor",
            "#F0B90B",
        ]
    proc = subprocess.Popen(dash_cmd, cwd=str(ROOT), stdout=dash_log, stderr=dash_log)
    target = script if ui == "streamlit" else "dashboard.server:app"
    print(f"[CENTINA] Dashboard: http://localhost:{port} ({target})")
    print(f"[CENTINA] Dashboard logs: logs/{log_name}")
    return proc


def build_agent_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", "src.main", "--mode", args.mode]
    if args.capital is not None:
        cmd += ["--capital", str(args.capital)]
    if args.no_voice:
        cmd += ["--no-voice"]
    if args.debug:
        cmd += ["--debug"]
    return cmd


def main() -> None:
    args = parse_args()
    dashboard_proc = None

    if not args.no_dashboard:
        dashboard_proc = start_dashboard(args.ui, args.dashboard, args.port)
        time.sleep(2)

    print(f"[CENTINA] Starting agent in {args.mode} mode.\n")

    try:
        subprocess.run(build_agent_cmd(args), cwd=str(ROOT), check=False)
    except KeyboardInterrupt:
        pass
    finally:
        if dashboard_proc and dashboard_proc.poll() is None:
            print("\n[CENTINA] Stopping dashboard.")
            dashboard_proc.terminate()
            try:
                dashboard_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard_proc.kill()


if __name__ == "__main__":
    main()
