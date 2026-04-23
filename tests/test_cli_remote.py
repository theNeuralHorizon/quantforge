"""Tests for the new CLI remote subcommands (jobs, auth, walk-forward)."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest
import uvicorn


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def running_server():
    os.environ["QUANTFORGE_JWT_SECRET"] = "this_is_a_minimum_32_byte_jwt_secret"
    from quantforge.api import auth, jwt_auth
    from quantforge.api import audit as audit_mod
    auth.reset_keys()
    auth.register_raw_key("cli_test_key_12345")
    audit_mod.reset_for_tests()
    jwt_auth.reload_config()

    from quantforge.api.app import create_app
    app = create_app()

    port = _free_port()
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                          access_log=False)
    server = uvicorn.Server(cfg)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("uvicorn didn't start")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


def _run_cli(args: list, env_extra: dict | None = None) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-m", "quantforge", *args],
        capture_output=True, text=True, env=env, timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


class TestCLILocalCommands:
    def test_price(self):
        rc, out, err = _run_cli(["price", "--S", "100", "--K", "100", "--T", "1.0", "--sigma", "0.2"])
        assert rc == 0, err
        assert "Black-Scholes" in out or "black_scholes" in out.lower()

    def test_iv(self):
        rc, out, err = _run_cli(["iv", "--price", "10.45", "--S", "100", "--K", "100",
                                   "--T", "1.0", "--r", "0.04"])
        assert rc == 0, err


class TestCLIRemoteCommands:
    def test_jobs_submit_and_list(self, running_server):
        env = {"QUANTFORGE_CLIENT_KEY": "cli_test_key_12345"}
        rc, out, err = _run_cli(
            ["jobs", "submit", "--api", running_server,
             "--strategy", "ma", "--tickers", "SPY",
             "--start", "2023-01-01", "--end", "2024-01-01"],
            env_extra=env,
        )
        assert rc == 0, err
        job = json.loads(out.split("\n")[0] if "{" not in out.split("\n")[0] else "\n".join(out.split("\n")[:out.count("{")+10]))
        # If multi-line JSON, grab first JSON object
        try:
            job = json.loads(out.strip().split("\n\n")[0])
        except Exception:
            # fallback: find first '{' to matching '}'
            s = out.find("{")
            depth = 0
            end = s
            for i, ch in enumerate(out[s:], start=s):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1; break
            job = json.loads(out[s:end])
        assert "job_id" in job

        # List
        rc, out, err = _run_cli(
            ["jobs", "list", "--api", running_server, "--limit", "5"],
            env_extra=env,
        )
        assert rc == 0
        lst = json.loads(out)
        assert isinstance(lst, list)

    def test_auth_token(self, running_server):
        # Re-apply server-side JWT in case another test module popped the env
        os.environ["QUANTFORGE_JWT_SECRET"] = "this_is_a_minimum_32_byte_jwt_secret"
        from quantforge.api import jwt_auth
        jwt_auth.reload_config()

        env = {"QUANTFORGE_CLIENT_KEY": "cli_test_key_12345",
               "QUANTFORGE_JWT_SECRET": "this_is_a_minimum_32_byte_jwt_secret"}
        rc, out, err = _run_cli(
            ["auth", "--api", running_server,
             "--subject", "cli-user",
             "--scopes", "options:read", "backtest:read",
             "--ttl", "600"],
            env_extra=env,
        )
        assert rc == 0, err
        tok = json.loads(out)
        assert "access_token" in tok
        assert tok["expires_in"] == 600
