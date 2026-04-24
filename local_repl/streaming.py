"""Long-running process management for LocalREPL.

Unlike shell_bridge.run_shell (which is blocking and one-shot), this
module spawns processes that keep running in the background. Output is
drained by daemon threads into a bounded deque so `tail_shell_output`
can return fresh lines without blocking. Stdin is writable via
`send_to_shell`, making this suitable for watchers, REPLs, servers,
and anything that would normally hang a one-shot subprocess call.

Typical use: `spawn_shell('watchexec -e py pytest')`, poll with
`tail_shell_output` every few seconds, kill when done.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from .repl import PythonREPL


# Cap per-process to prevent runaway memory use on chatty processes.
# At ~200 bytes/line avg, 5000 lines ≈ 1 MB per managed process.
MAX_BUFFERED_LINES = 5000
DEFAULT_SPAWN_TIMEOUT = 30  # for wait_shell


class SpawnResult(TypedDict):
    ok: bool
    proc_id: str
    pid: int | None
    command: str
    cwd: str
    started_at: str
    error: str | None


class TailResult(TypedDict):
    proc_id: str
    running: bool
    exit_code: int | None
    lines: list[dict]       # each {'ts': iso, 'stream': 'stdout'|'stderr', 'text': str}
    total_buffered: int
    truncated: bool         # True if buffer hit MAX_BUFFERED_LINES


class ProcInfo(TypedDict):
    proc_id: str
    pid: int
    command: str
    cwd: str
    started_at: str
    running: bool
    exit_code: int | None
    buffered_lines: int


@dataclass
class ManagedProcess:
    """Bookkeeping for one spawned process."""
    proc_id: str
    popen: subprocess.Popen
    command: str
    cwd: str
    started_at: str
    buffer: deque = field(default_factory=lambda: deque(maxlen=MAX_BUFFERED_LINES))
    buffer_lock: threading.Lock = field(default_factory=threading.Lock)
    threads: list[threading.Thread] = field(default_factory=list)
    hit_cap: bool = False  # flipped true once buffer rolled over


# Module-level registry of active processes. Keyed by proc_id (UUID) so
# PID reuse across the OS can't cause cross-wiring.
_PROCESSES: dict[str, ManagedProcess] = {}
_REGISTRY_LOCK = threading.Lock()


def _drain_stream(mp: ManagedProcess, stream, stream_name: str) -> None:
    """Background-thread loop: read lines from a stream into the buffer."""
    try:
        for raw in iter(stream.readline, ""):
            if not raw:
                break
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "stream": stream_name,
                "text": raw.rstrip("\n"),
            }
            with mp.buffer_lock:
                if len(mp.buffer) == mp.buffer.maxlen:
                    mp.hit_cap = True
                mp.buffer.append(entry)
    except (ValueError, OSError):
        # Stream closed underneath us — normal on process exit.
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _resolve_cwd(repl, cwd_override: str | None) -> str:
    """Mirror shell_bridge's logic without circular import."""
    if cwd_override:
        expanded = os.path.expanduser(cwd_override)
        if not os.path.isdir(expanded):
            raise FileNotFoundError(f"cwd does not exist: {expanded}")
        return expanded
    return getattr(repl, "cwd", os.getcwd())


def register_streaming_tools(
    mcp: "FastMCP",
    repl_instances: dict[str, "PythonREPL"],
) -> None:
    """Register long-running-process tools on the given MCP instance."""

    @mcp.tool()
    def spawn_shell(
        command: str,
        repl_id: str,
        cwd: str | None = None,
        use_shell: bool = False,
        env_extra: dict[str, str] | None = None,
    ) -> SpawnResult:
        """Spawn a long-running process. Returns a proc_id for follow-up calls.

        Use this for watchers (watchexec, tail -f), servers, REPLs, or any
        command you want to poll output from over time. For one-shot
        commands use run_shell instead — it's simpler and auto-logs.

        Returns proc_id which is passed to tail_shell_output, send_to_shell,
        kill_shell, and wait_shell.
        """
        repl = repl_instances.get(repl_id)
        if not repl:
            return SpawnResult(
                ok=False, proc_id="", pid=None, command=command, cwd="",
                started_at="", error=f"REPL {repl_id} not found",
            )

        try:
            run_cwd = _resolve_cwd(repl, cwd)
        except FileNotFoundError as exc:
            return SpawnResult(
                ok=False, proc_id="", pid=None, command=command,
                cwd=str(cwd or ""), started_at="", error=str(exc),
            )

        # Check the blocklist via shell_bridge so rules stay consistent.
        from .shell_bridge import _is_blocked
        blocked, reason = _is_blocked(command)
        if blocked:
            return SpawnResult(
                ok=False, proc_id="", pid=None, command=command,
                cwd=run_cwd, started_at="", error=f"refused: {reason}",
            )

        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)

        try:
            if use_shell:
                popen = subprocess.Popen(
                    command, shell=True, cwd=run_cwd, env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, bufsize=1,
                )
            else:
                tokens = shlex.split(command)
                popen = subprocess.Popen(
                    tokens, cwd=run_cwd, env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, bufsize=1,
                )
        except (FileNotFoundError, ValueError) as exc:
            return SpawnResult(
                ok=False, proc_id="", pid=None, command=command,
                cwd=run_cwd, started_at="", error=str(exc),
            )

        proc_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        mp = ManagedProcess(
            proc_id=proc_id, popen=popen, command=command,
            cwd=run_cwd, started_at=started_at,
        )

        # Spin up drain threads. Daemon=True so they don't prevent shutdown.
        t_out = threading.Thread(
            target=_drain_stream, args=(mp, popen.stdout, "stdout"),
            daemon=True, name=f"drain-stdout-{proc_id[:8]}",
        )
        t_err = threading.Thread(
            target=_drain_stream, args=(mp, popen.stderr, "stderr"),
            daemon=True, name=f"drain-stderr-{proc_id[:8]}",
        )
        t_out.start()
        t_err.start()
        mp.threads = [t_out, t_err]

        with _REGISTRY_LOCK:
            _PROCESSES[proc_id] = mp

        return SpawnResult(
            ok=True, proc_id=proc_id, pid=popen.pid, command=command,
            cwd=run_cwd, started_at=started_at, error=None,
        )


    @mcp.tool()
    def tail_shell_output(
        proc_id: str,
        n_lines: int = 50,
        stream: str = "both",
        drain: bool = False,
    ) -> TailResult:
        """Return the most recent lines from a spawned process's output.

        Args:
            proc_id: From spawn_shell.
            n_lines: How many recent lines to return (capped at 500).
            stream: 'stdout', 'stderr', or 'both'.
            drain: If True, remove returned lines from the buffer so the
                next call sees only NEW output. Useful for polling loops.
        """
        with _REGISTRY_LOCK:
            mp = _PROCESSES.get(proc_id)
        if not mp:
            return TailResult(
                proc_id=proc_id, running=False, exit_code=None,
                lines=[], total_buffered=0, truncated=False,
            )

        exit_code = mp.popen.poll()
        running = exit_code is None
        capped_n = max(1, min(int(n_lines), 500))
        wanted = {"stdout", "stderr"} if stream == "both" else {stream}

        with mp.buffer_lock:
            total = len(mp.buffer)
            matched = [e for e in mp.buffer if e["stream"] in wanted]
            lines = matched[-capped_n:]
            truncated = mp.hit_cap
            if drain:
                # Remove the entries we're returning so polling sees fresh
                # output next time. Only remove from the set we're returning.
                ids_to_drop = {id(e) for e in lines}
                remaining = deque(
                    (e for e in mp.buffer if id(e) not in ids_to_drop),
                    maxlen=mp.buffer.maxlen,
                )
                mp.buffer = remaining

        return TailResult(
            proc_id=proc_id, running=running, exit_code=exit_code,
            lines=lines, total_buffered=total, truncated=truncated,
        )

    @mcp.tool()
    def send_to_shell(proc_id: str, input_text: str,
                      append_newline: bool = True) -> dict:
        """Write to a spawned process's stdin. Useful for interactive REPLs."""
        with _REGISTRY_LOCK:
            mp = _PROCESSES.get(proc_id)
        if not mp:
            return {"ok": False, "error": f"proc_id {proc_id} not found"}
        if mp.popen.poll() is not None:
            return {"ok": False, "error": "process has already exited"}

        try:
            payload = input_text + ("\n" if append_newline else "")
            mp.popen.stdin.write(payload)
            mp.popen.stdin.flush()
            return {"ok": True, "bytes_written": len(payload)}
        except (BrokenPipeError, OSError) as exc:
            return {"ok": False, "error": f"write failed: {exc}"}

    @mcp.tool()
    def kill_shell(proc_id: str, signal_name: str = "SIGTERM",
                   remove_from_registry: bool = True) -> dict:
        """Send a signal to a spawned process.

        Args:
            signal_name: 'SIGTERM' (graceful, default), 'SIGKILL' (force),
                or 'SIGINT' (Ctrl-C equivalent).
            remove_from_registry: Drop from _PROCESSES after signalling.
        """
        with _REGISTRY_LOCK:
            mp = _PROCESSES.get(proc_id)
        if not mp:
            return {"ok": False, "error": f"proc_id {proc_id} not found"}

        sig_map = {
            "SIGTERM": signal.SIGTERM,
            "SIGKILL": signal.SIGKILL,
            "SIGINT": signal.SIGINT,
            "SIGHUP": signal.SIGHUP,
        }
        sig = sig_map.get(signal_name.upper())
        if sig is None:
            return {"ok": False, "error": f"unknown signal: {signal_name}"}

        already_exited = mp.popen.poll() is not None
        if not already_exited:
            try:
                mp.popen.send_signal(sig)
            except ProcessLookupError:
                already_exited = True

        # Brief grace period for graceful signals.
        if sig in (signal.SIGTERM, signal.SIGINT) and not already_exited:
            try:
                mp.popen.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass  # Caller can retry with SIGKILL.

        final_code = mp.popen.poll()
        if remove_from_registry and final_code is not None:
            with _REGISTRY_LOCK:
                _PROCESSES.pop(proc_id, None)

        return {
            "ok": True, "proc_id": proc_id, "signal_sent": signal_name,
            "exit_code": final_code, "still_running": final_code is None,
        }

    @mcp.tool()
    def wait_shell(proc_id: str, timeout_seconds: int = DEFAULT_SPAWN_TIMEOUT) -> dict:
        """Block until a spawned process exits (or timeout). Returns exit info."""
        with _REGISTRY_LOCK:
            mp = _PROCESSES.get(proc_id)
        if not mp:
            return {"ok": False, "error": f"proc_id {proc_id} not found"}

        clamped = max(1, min(int(timeout_seconds), 600))
        start = time.perf_counter()
        try:
            code = mp.popen.wait(timeout=clamped)
            return {
                "ok": True, "proc_id": proc_id, "exit_code": code,
                "waited_ms": round((time.perf_counter() - start) * 1000, 2),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": True, "proc_id": proc_id, "exit_code": None,
                "waited_ms": round((time.perf_counter() - start) * 1000, 2),
                "timed_out": True,
            }

    @mcp.tool()
    def list_shells() -> list[ProcInfo]:
        """Return info on all spawned processes (live and recently-exited)."""
        with _REGISTRY_LOCK:
            items = list(_PROCESSES.items())
        out: list[ProcInfo] = []
        for proc_id, mp in items:
            code = mp.popen.poll()
            with mp.buffer_lock:
                buf_count = len(mp.buffer)
            out.append(ProcInfo(
                proc_id=proc_id, pid=mp.popen.pid, command=mp.command,
                cwd=mp.cwd, started_at=mp.started_at,
                running=code is None, exit_code=code,
                buffered_lines=buf_count,
            ))
        return out

    @mcp.tool()
    def reap_exited() -> dict:
        """Remove exited processes from the registry. Call occasionally to
        keep list_shells tidy, especially after running many short spawns.
        """
        removed: list[str] = []
        with _REGISTRY_LOCK:
            for proc_id, mp in list(_PROCESSES.items()):
                if mp.popen.poll() is not None:
                    _PROCESSES.pop(proc_id, None)
                    removed.append(proc_id)
        return {"ok": True, "reaped_count": len(removed), "reaped": removed}
