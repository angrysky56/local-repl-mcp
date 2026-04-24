"""Shell command bridge for LocalREPL.

Exposes tools that run shell commands with structured return values
(stdout, stderr, exit_code, duration_ms, cwd, command), timeouts,
and per-REPL cwd tracking. Every invocation is logged to evolution.db
via evolution_memory so past commands become queryable operational
memory for the agent.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from .repl import PythonREPL


# Commands refused even when the caller opts in; defense-in-depth on top
# of whatever the host OS / sandbox blocks. Kept small on purpose.
_BLOCKED_PREFIXES: tuple[str, ...] = (
    "sudo ", "su ",
    "rm -rf /", "rm -rf ~", "rm -rf $HOME", "rm -rf --no-preserve-root",
    ":(){ :|:& };:",   # classic fork bomb
    "mkfs", "fdisk", "parted",
    "dd if=/dev/zero of=/dev/", "dd if=/dev/random of=/dev/",
    "shutdown", "reboot", "halt", "poweroff", "init 0", "init 6",
)

DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 600
STDOUT_PREVIEW_CHARS = 4000   # cap on what we return to the caller
STDERR_PREVIEW_CHARS = 2000
LOG_PREVIEW_CHARS = 500       # first N chars stored in evolution.db


class ShellResult(TypedDict):
    """Structured output for shell command execution."""
    command: str
    cwd: str
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    timed_out: bool
    truncated_stdout: bool
    truncated_stderr: bool


def _is_blocked(command: str) -> tuple[bool, str]:
    """Return (blocked, reason) for a command. Case-insensitive prefix match."""
    stripped = command.strip().lower()
    for prefix in _BLOCKED_PREFIXES:
        if stripped.startswith(prefix):
            return True, f"blocked prefix: {prefix!r}"
        # Also catch chained invocations: e.g. '... && sudo rm ...'
        if f"&& {prefix}" in stripped or f"; {prefix}" in stripped:
            return True, f"blocked prefix (chained): {prefix!r}"
    return False, ""


def _resolve_cwd(repl: "PythonREPL", cwd_override: str | None) -> str:
    """Pick the directory to run the command in. Raises if cwd doesn't exist."""
    if cwd_override:
        expanded = os.path.expanduser(cwd_override)
        if not os.path.isdir(expanded):
            raise FileNotFoundError(f"cwd does not exist: {expanded}")
        return expanded
    # Fall back to the REPL's tracked cwd, else the server process's cwd.
    return getattr(repl, "cwd", os.getcwd())


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Return (maybe-truncated text, was_truncated)."""
    if len(text) <= limit:
        return text, False
    overflow = len(text) - limit
    return text[:limit] + f"\n... [truncated, {overflow} more chars]", True


def _log_soft(**entry) -> None:
    """Best-effort logging to evolution.db. Never raises."""
    try:
        from . import evolution_memory
        evolution_memory.log_command(**entry)
    except Exception:
        # Memory is a nice-to-have; never break the tool over a logging failure.
        pass


def register_shell_tools(
    mcp: "FastMCP",
    repl_instances: dict[str, "PythonREPL"],
) -> None:
    """Register shell-bridge tools on the given MCP instance."""

    @mcp.tool()
    def run_shell(
        command: str,
        repl_id: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        cwd: str | None = None,
        use_shell: bool = False,
        env_extra: dict[str, str] | None = None,
        stdin_input: str | None = None,
    ) -> ShellResult:
        """Run a shell command inside a REPL context with structured output.

        Returns a dict so the caller can branch on exit_code and parse
        stdout without string-scraping. Every call is logged to evolution.db
        for later recall via `query_command_history`.

        Args:
            command: The shell command to execute.
            repl_id: REPL whose cwd the command runs in.
            timeout_seconds: Kill the process if it exceeds this (capped at 600).
            cwd: Override the REPL's cwd for this one call.
            use_shell: If True, pass to /bin/sh (enables pipes, redirects,
                glob expansion). Default False = tokenized exec, safer.
            env_extra: Additional env vars merged onto os.environ.
            stdin_input: Optional string piped to the command's stdin.
                Enables first-class stdin piping so CLI tools that read
                from stdin (jq, wc, grep, etc.) work without shell pipe
                syntax.  When None the child's stdin is /dev/null.
        """
        repl = repl_instances.get(repl_id)
        if not repl:
            return ShellResult(
                command=command, cwd="", stdout="",
                stderr=f"REPL {repl_id} not found; create one first.",
                exit_code=127, duration_ms=0.0, timed_out=False,
                truncated_stdout=False, truncated_stderr=False,
            )

        blocked, reason = _is_blocked(command)
        if blocked:
            return ShellResult(
                command=command, cwd="", stdout="",
                stderr=f"refused: {reason}",
                exit_code=126, duration_ms=0.0, timed_out=False,
                truncated_stdout=False, truncated_stderr=False,
            )

        try:
            run_cwd = _resolve_cwd(repl, cwd)
        except FileNotFoundError as exc:
            return ShellResult(
                command=command, cwd=str(cwd or ""), stdout="",
                stderr=str(exc), exit_code=2, duration_ms=0.0,
                timed_out=False, truncated_stdout=False, truncated_stderr=False,
            )

        clamped_timeout = max(1, min(timeout_seconds, MAX_TIMEOUT_SECONDS))
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)

        # Build stdin kwargs: pipe caller data or seal off stdin entirely
        # so the child never accidentally reads MCP protocol traffic.
        stdin_kwargs: dict = (
            {"input": stdin_input}          # implies stdin=PIPE
            if stdin_input is not None
            else {"stdin": subprocess.DEVNULL}
        )

        start = time.perf_counter()
        timed_out = False
        stdout = stderr = ""
        exit_code = 0

        try:
            if use_shell:
                proc = subprocess.run(
                    command, shell=True, cwd=run_cwd, env=env,
                    capture_output=True, text=True,
                    timeout=clamped_timeout, check=False,
                    **stdin_kwargs,
                )
            else:
                tokens = shlex.split(command)
                if not tokens:
                    return ShellResult(
                        command=command, cwd=run_cwd, stdout="",
                        stderr="empty command after tokenization",
                        exit_code=2, duration_ms=0.0, timed_out=False,
                        truncated_stdout=False, truncated_stderr=False,
                    )
                proc = subprocess.run(
                    tokens, cwd=run_cwd, env=env,
                    capture_output=True, text=True,
                    timeout=clamped_timeout, check=False,
                    **stdin_kwargs,
                )
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            exit_code = 124  # matches GNU `timeout` convention
            timed_out = True
        except FileNotFoundError as exc:
            stdout, stderr, exit_code = "", f"{exc}", 127
        except ValueError as exc:
            # shlex.split on malformed input
            stdout, stderr, exit_code = "", f"parse error: {exc}", 2

        duration_ms = (time.perf_counter() - start) * 1000.0
        stdout, truncated_stdout = _truncate(stdout, STDOUT_PREVIEW_CHARS)
        stderr, truncated_stderr = _truncate(stderr, STDERR_PREVIEW_CHARS)

        _log_soft(
            repl_id=repl_id, command=command, cwd=run_cwd,
            exit_code=exit_code, duration_ms=duration_ms,
            stdout_preview=stdout[:LOG_PREVIEW_CHARS],
            stderr_preview=stderr[:LOG_PREVIEW_CHARS],
            timed_out=timed_out,
        )

        return ShellResult(
            command=command, cwd=run_cwd, stdout=stdout, stderr=stderr,
            exit_code=exit_code, duration_ms=round(duration_ms, 2),
            timed_out=timed_out,
            truncated_stdout=truncated_stdout,
            truncated_stderr=truncated_stderr,
        )

    @mcp.tool()
    def set_repl_cwd(repl_id: str, path: str) -> dict:
        """Change a REPL's tracked working directory for future shell calls."""
        repl = repl_instances.get(repl_id)
        if not repl:
            return {"ok": False, "error": f"REPL {repl_id} not found."}
        expanded = os.path.expanduser(path)
        if not os.path.isdir(expanded):
            return {"ok": False, "error": f"not a directory: {expanded}"}
        repl.cwd = expanded
        return {"ok": True, "repl_id": repl_id, "cwd": expanded}

    @mcp.tool()
    def get_repl_cwd(repl_id: str) -> dict:
        """Return the REPL's current working directory."""
        repl = repl_instances.get(repl_id)
        if not repl:
            return {"ok": False, "error": f"REPL {repl_id} not found."}
        return {"ok": True, "repl_id": repl_id,
                "cwd": getattr(repl, "cwd", os.getcwd())}
