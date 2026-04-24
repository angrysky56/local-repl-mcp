"""Venv-aware Python execution.

Runs Python code in a specific project's virtual environment by spawning
`<venv>/bin/python -c CODE` as a subprocess. Unlike run_python_in_repl
(which executes in-process in the MCP server's interpreter), this tool
gives access to whatever packages are installed in the project's venv
without having to install them globally or into the MCP server's env.

Stateless by design: each call is a fresh subprocess. For stateful work
within a single venv, write a single longer script per call, or fall back
to shell_bridge.spawn_shell with `<venv>/bin/python -i` for an
interactive session.

Every call is logged to evolution.db via evolution_memory, same as
shell_bridge — so `query_command_history` surfaces these too.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from .repl import PythonREPL


DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 600
STDOUT_PREVIEW_CHARS = 4000
STDERR_PREVIEW_CHARS = 2000
LOG_PREVIEW_CHARS = 500


class VenvResult(TypedDict):
    """Structured output for venv Python execution."""
    ok: bool
    python_path: str
    cwd: str
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    timed_out: bool
    truncated_stdout: bool
    truncated_stderr: bool
    error: str | None   # set when the call couldn't start (bad venv path, etc.)


def _resolve_python(venv_path: str) -> tuple[str | None, str]:
    """Return (python_binary, error_message).

    Accepts either a path to the venv directory (`.venv`) or directly to
    the python binary. Checks standard layouts for POSIX and Windows.
    """
    expanded = os.path.expanduser(venv_path)
    if not os.path.exists(expanded):
        return None, f"path does not exist: {expanded}"

    # If user passed the python binary directly, use it.
    if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
        return expanded, ""

    # Otherwise treat it as a venv root and look for bin/python or Scripts/python.exe
    candidates = [
        os.path.join(expanded, "bin", "python"),
        os.path.join(expanded, "bin", "python3"),
        os.path.join(expanded, "Scripts", "python.exe"),
    ]
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand, ""

    return None, (
        f"no python binary found under {expanded!r}; "
        f"expected one of: bin/python, bin/python3, Scripts/python.exe"
    )


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    overflow = len(text) - limit
    return text[:limit] + f"\n... [truncated, {overflow} more chars]", True


def _log_soft(**entry) -> None:
    try:
        from . import evolution_memory
        evolution_memory.log_command(**entry)
    except Exception:
        pass


def register_venv_tools(
    mcp: "FastMCP",
    repl_instances: dict[str, "PythonREPL"],
) -> None:
    """Register venv-aware execution tools on the given MCP instance."""

    @mcp.tool()
    def run_python_in_venv(
        code: str,
        venv_path: str,
        repl_id: str | None = None,
        cwd: str | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        env_extra: dict[str, str] | None = None,
    ) -> VenvResult:
        """Run Python code inside a project's virtual environment.

        Spawns a fresh subprocess running `<venv>/bin/python -c CODE`.
        No namespace carries over between calls — each invocation is a
        new interpreter. Use this when the project has dependencies that
        aren't installed in the MCP server's Python environment.

        Args:
            code: Python source to execute.
            venv_path: Path to the venv directory (e.g., "/proj/.venv")
                OR direct path to the python binary. `~` is expanded.
            repl_id: Optional REPL to inherit cwd from. Falls back to
                cwd argument, then to server's cwd.
            cwd: Explicit working directory override.
            timeout_seconds: Kill after N seconds (capped at 600).
            env_extra: Additional env vars merged onto os.environ.
                PYTHONPATH is NOT auto-set — pass it via env_extra if
                the project uses src-layout and needs it.

        Returns:
            VenvResult dict with ok/python_path/cwd/stdout/stderr/
            exit_code/duration_ms/timed_out/truncation flags/error.
        """
        python_bin, err = _resolve_python(venv_path)
        if python_bin is None:
            return VenvResult(
                ok=False, python_path="", cwd="", stdout="", stderr="",
                exit_code=127, duration_ms=0.0, timed_out=False,
                truncated_stdout=False, truncated_stderr=False,
                error=err,
            )

        # Resolve cwd: explicit arg > REPL > server default
        run_cwd = cwd
        if run_cwd is None and repl_id:
            repl = repl_instances.get(repl_id)
            if repl:
                run_cwd = getattr(repl, "cwd", None)
        if run_cwd is None:
            run_cwd = os.getcwd()
        run_cwd = os.path.expanduser(run_cwd)
        if not os.path.isdir(run_cwd):
            return VenvResult(
                ok=False, python_path=python_bin, cwd=run_cwd,
                stdout="", stderr="", exit_code=2, duration_ms=0.0,
                timed_out=False, truncated_stdout=False,
                truncated_stderr=False,
                error=f"cwd does not exist: {run_cwd}",
            )

        clamped_timeout = max(1, min(int(timeout_seconds), MAX_TIMEOUT_SECONDS))
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)

        start = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                [python_bin, "-c", code],
                cwd=run_cwd, env=env,
                capture_output=True, text=True,
                stdin=subprocess.DEVNULL,        # avoid rg-style stdin hangs
                timeout=clamped_timeout, check=False,
            )
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            exit_code = 124
            timed_out = True

        duration_ms = (time.perf_counter() - start) * 1000.0
        stdout, truncated_stdout = _truncate(stdout, STDOUT_PREVIEW_CHARS)
        stderr, truncated_stderr = _truncate(stderr, STDERR_PREVIEW_CHARS)

        # Log as a shell-style row so query_command_history finds it.
        log_cmd = f"[venv:{python_bin}] python -c <code: {len(code)} chars>"
        _log_soft(
            repl_id=repl_id, command=log_cmd, cwd=run_cwd,
            exit_code=exit_code, duration_ms=duration_ms,
            stdout_preview=stdout[:LOG_PREVIEW_CHARS],
            stderr_preview=stderr[:LOG_PREVIEW_CHARS],
            timed_out=timed_out,
            tags=["venv"],
        )

        return VenvResult(
            ok=(exit_code == 0),
            python_path=python_bin, cwd=run_cwd,
            stdout=stdout, stderr=stderr,
            exit_code=exit_code, duration_ms=round(duration_ms, 2),
            timed_out=timed_out,
            truncated_stdout=truncated_stdout,
            truncated_stderr=truncated_stderr,
            error=None,
        )

    @mcp.tool()
    def find_venv(repl_id: str) -> dict:
        """Locate a virtual environment in the REPL's current directory.

        Checks, in order: .venv/, venv/, env/. Returns the path to the
        python binary if found, so it can be fed directly to
        run_python_in_venv.

        Args:
            repl_id: REPL whose cwd to search under.
        """
        repl = repl_instances.get(repl_id)
        if not repl:
            return {"ok": False, "error": f"REPL {repl_id} not found."}
        search_root = getattr(repl, "cwd", os.getcwd())

        for candidate in (".venv", "venv", "env"):
            venv_root = os.path.join(search_root, candidate)
            python_bin, err = _resolve_python(venv_root)
            if python_bin:
                # Extract python version for the caller's convenience.
                try:
                    ver = subprocess.run(
                        [python_bin, "--version"],
                        capture_output=True, text=True,
                        stdin=subprocess.DEVNULL,
                        timeout=5, check=False,
                    )
                    version = (ver.stdout or ver.stderr).strip()
                except Exception:
                    version = "unknown"
                return {
                    "ok": True,
                    "venv_root": venv_root,
                    "python_path": python_bin,
                    "version": version,
                    "searched_under": search_root,
                }
        return {
            "ok": False,
            "searched_under": search_root,
            "checked": [".venv", "venv", "env"],
            "error": "no venv found; create one with `uv venv` or `python -m venv .venv`",
        }
