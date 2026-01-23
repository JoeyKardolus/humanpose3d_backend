import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import socket
import webbrowser
from pathlib import Path


def run(cmd, shell=False, check=True, **kwargs):
    print(f"> {cmd}")
    return subprocess.run(cmd, shell=shell, check=check, **kwargs)


def uv_exists() -> bool:
    """Check if uv is available, including common installation paths."""
    if shutil.which("uv") is not None:
        return True

    # Direct check for common installation locations
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".local", "bin", "uv.exe"),  # Windows
        os.path.join(home, ".local", "bin", "uv"),  # Unix
        os.path.join(home, ".cargo", "bin", "uv.exe"),  # Windows (older)
        os.path.join(home, ".cargo", "bin", "uv"),  # Unix (older)
    ]
    return any(os.path.isfile(c) for c in candidates)


def install_uv_windows():
    run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "irm https://astral.sh/uv/install.ps1 | iex",
        ]
    )


def install_uv_macos_linux():
    run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True)


def ensure_uv_available_in_path():
    """Add common uv installation directories to PATH if they exist."""
    home = os.path.expanduser("~")
    current_path = os.environ.get("PATH", "")

    # Common uv installation paths by platform
    candidate_paths = [
        os.path.join(home, ".cargo", "bin"),  # macOS/Linux (older)
        os.path.join(home, ".local", "bin"),  # Windows and modern uv
    ]

    for candidate in candidate_paths:
        if os.path.isdir(candidate) and candidate not in current_path:
            os.environ["PATH"] = current_path + os.pathsep + candidate
            current_path = os.environ["PATH"]


def get_uv_executable() -> str | None:
    """Return the path to the uv executable, or None if not found."""
    which_result = shutil.which("uv")
    if which_result:
        return which_result

    # Direct check for common installation locations
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".local", "bin", "uv.exe"),  # Windows
        os.path.join(home, ".local", "bin", "uv"),  # Unix
        os.path.join(home, ".cargo", "bin", "uv.exe"),  # Windows (older)
        os.path.join(home, ".cargo", "bin", "uv"),  # Unix (older)
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def fail_if_uv_missing():
    if not uv_exists():
        print(
            "\n❌ uv installed, but not found in PATH.\n"
            "Restart your terminal (or log out/in) and run again.\n"
        )
        sys.exit(1)


def ensure_uv_installed():
    system = platform.system()
    if uv_exists():
        print("uv already installed.")
        return

    print(f"uv not found. Installing for {system}...")
    if system == "Windows":
        install_uv_windows()
    elif system in ("Darwin", "Linux"):
        install_uv_macos_linux()
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    ensure_uv_available_in_path()
    fail_if_uv_missing()
    print("✅ uv installed and available.")


def get_uv_command() -> str:
    """Return the uv command to use (full path if not in PATH)."""
    uv_path = get_uv_executable()
    if uv_path is None:
        raise RuntimeError("uv executable not found")
    return uv_path


def uv_sync():
    print("\nRunning uv sync...")
    uv = get_uv_command()
    run([uv, "sync"])


def is_port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_server(host: str, port: int, max_wait_s: int = 30) -> bool:
    print(f"\nWaiting for server on {host}:{port} (max {max_wait_s}s)...")
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        if is_port_open(host, port):
            print("✅ Server is reachable.")
            return True
        time.sleep(0.25)
    print("❌ Server did not become reachable in time.")
    return False


def open_server_page(host: str, port: int):
    url = f"http://{host}:{port}/"
    print(f"Opening browser: {url}")
    webbrowser.open(url, new=2)  # new tab if possible


def runserver_console(host: str, port: int):
    print("\nStarting Django dev server (console mode)...")
    uv = get_uv_command()
    # This blocks until server exits
    run([uv, "run", "python", "manage.py", "runserver", f"{host}:{port}"])


def runserver_spawn_windows(host: str, port: int):
    print("\nStarting Django dev server (spawn new console)...")
    uv = get_uv_command()
    CREATE_NEW_CONSOLE = 0x00000010
    # Returns immediately
    subprocess.Popen(
        [uv, "run", "python", "manage.py", "runserver", f"{host}:{port}"],
        creationflags=CREATE_NEW_CONSOLE,
    )


def runserver_log(host: str, port: int, log_path: Path):
    print(f"\nStarting Django dev server (logging to {log_path})...")
    uv = get_uv_command()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8")

    p = subprocess.Popen(
        [uv, "run", "python", "manage.py", "runserver", f"{host}:{port}"],
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return p  # caller decides whether to wait


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["console", "spawn", "log"], default="console"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--logfile", default="logs/bootstrap.log")
    parser.add_argument("--open-browser", action="store_true", default=True)
    parser.add_argument("--no-open-browser", action="store_false", dest="open_browser")
    parser.add_argument("--wait-seconds", type=int, default=30)
    args = parser.parse_args()

    print(f"Detected OS: {platform.system()}")
    print(f"Mode: {args.mode}")

    ensure_uv_installed()

    if not args.skip_sync:
        uv_sync()

    system = platform.system()
    host, port = args.host, args.port

    # Start server
    server_proc = None

    uv = get_uv_command()

    if args.mode == "console":
        # In console mode we need to open browser BEFORE blocking,
        # so we start server in background, open browser, then wait.
        server_proc = subprocess.Popen(
            [uv, "run", "python", "manage.py", "runserver", f"{host}:{port}"]
        )

    elif args.mode == "spawn":
        if system != "Windows":
            print(
                "⚠️ spawn mode is mainly for Windows. Falling back to console-mode background start."
            )
            server_proc = subprocess.Popen(
                [uv, "run", "python", "manage.py", "runserver", f"{host}:{port}"]
            )
        else:
            runserver_spawn_windows(host, port)

    elif args.mode == "log":
        server_proc = runserver_log(host, port, Path(args.logfile))

    # Wait until it’s reachable, then open browser
    if args.open_browser:
        if wait_for_server(host, port, max_wait_s=args.wait_seconds):
            open_server_page(host, port)
        else:
            print("Not opening browser because server is not reachable.")

    # Keep the .exe alive when needed
    if args.mode == "spawn":
        # In spawn mode, our process would exit instantly otherwise (bad for windowed exe).
        # So we just keep running while the server is reachable.
        print("\nSpawn mode: keeping launcher alive. Close this app to stop launcher.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    if server_proc is not None:
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            # If user Ctrl+C in console mode, propagate
            try:
                server_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
