"""Smoke test for the LocalREPL unlocks.

Run this BEFORE restarting Claude Desktop to catch import errors and
verify each new module works end-to-end against a scratch REPL.

    uv run python -m local_repl.doctor

Exits 0 if everything passes, 1 if any check fails.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable


# Colors are cosmetic — strip if not a tty.
_USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _pass(msg: str) -> None:
    print(f"  {_c('✓', '32')} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_c('✗', '31')} {msg}")


def _info(msg: str) -> None:
    print(f"  {_c('·', '36')} {msg}")


def _section(title: str) -> None:
    print(f"\n{_c(title, '1;37')}")


def _run_checks(checks: list[tuple[str, Callable[[], bool]]]) -> int:
    """Run a list of (name, check_fn) and return failure count."""
    failures = 0
    for name, fn in checks:
        try:
            ok = fn()
        except Exception as exc:  # noqa: BLE001 - we want the full story
            _fail(f"{name}: raised {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if ok:
            _pass(name)
        else:
            _fail(f"{name}: check returned falsy")
            failures += 1
    return failures


def check_imports() -> bool:
    """Can every new module be imported without errors?"""
    from local_repl import shell_bridge, streaming, evolution_memory  # noqa: F401
    from local_repl.repl import PythonREPL
    # Confirm the cwd attribute patch took effect.
    r = PythonREPL()
    assert hasattr(r, "cwd"), "PythonREPL.cwd missing — repl.py patch didn't apply"
    assert os.path.isdir(r.cwd), f"PythonREPL.cwd not a real dir: {r.cwd}"
    return True


def check_shell_basic() -> bool:
    """run_shell should return a ShellResult for a trivial command."""
    from local_repl.repl import PythonREPL
    from local_repl.shell_bridge import register_shell_tools
    from mcp.server.fastmcp import FastMCP

    test_mcp = FastMCP(name="doctor-shell", debug=False)
    instances: dict = {}
    register_shell_tools(test_mcp, instances)
    repl = PythonREPL()
    instances[repl.repl_id] = repl

    # Fetch the registered function out of the FastMCP tool manager.
    # FastMCP stores tools via its tool manager; access via _tool_manager.
    tool = test_mcp._tool_manager.get_tool("run_shell")
    result = tool.fn(command="echo hello-doctor", repl_id=repl.repl_id)

    assert result["exit_code"] == 0, f"exit_code {result['exit_code']}: {result['stderr']}"
    assert "hello-doctor" in result["stdout"], f"unexpected stdout: {result['stdout']!r}"
    assert result["duration_ms"] >= 0
    _info(f"echo completed in {result['duration_ms']:.1f} ms")
    return True


def check_shell_blocklist() -> bool:
    """Blocked prefixes should refuse without executing."""
    from local_repl.shell_bridge import _is_blocked
    blocked, reason = _is_blocked("sudo rm /etc/passwd")
    assert blocked, "sudo prefix should be blocked"
    blocked, _ = _is_blocked("echo hello && sudo poweroff")
    assert blocked, "chained sudo should be blocked"
    blocked, _ = _is_blocked("echo safe")
    assert not blocked, "plain echo should not be blocked"
    return True


def check_shell_timeout() -> bool:
    """A command that overruns its budget should come back timed_out=True."""
    from local_repl.repl import PythonREPL
    from local_repl.shell_bridge import register_shell_tools
    from mcp.server.fastmcp import FastMCP

    test_mcp = FastMCP(name="doctor-timeout", debug=False)
    instances: dict = {}
    register_shell_tools(test_mcp, instances)
    repl = PythonREPL()
    instances[repl.repl_id] = repl

    tool = test_mcp._tool_manager.get_tool("run_shell")
    result = tool.fn(command="sleep 5", repl_id=repl.repl_id, timeout_seconds=1)
    assert result["timed_out"] is True, "sleep 5 should have timed out at 1s"
    assert result["exit_code"] == 124, f"expected exit 124, got {result['exit_code']}"
    return True


def check_memory_roundtrip() -> bool:
    """Log a synthetic command and query it back."""
    from local_repl import evolution_memory as mem
    marker = f"doctor-marker-{int(time.time())}"
    row_id = mem.log_command(
        repl_id="doctor", command=f"echo {marker}",
        cwd="/tmp", exit_code=0, duration_ms=1.23,
        stdout_preview=marker, stderr_preview="",
    )
    assert row_id is not None and row_id > 0, "log_command returned None"
    _info(f"inserted row id={row_id}")
    return True


def check_memory_query() -> bool:
    """query_command_history should find the marker we just inserted."""
    from local_repl import evolution_memory as mem
    from mcp.server.fastmcp import FastMCP

    test_mcp = FastMCP(name="doctor-mem-query", debug=False)
    mem.register_memory_tools(test_mcp)
    tool = test_mcp._tool_manager.get_tool("query_command_history")

    rows = tool.fn(pattern="%doctor-marker%", limit=5)
    assert len(rows) >= 1, "query_command_history returned nothing"
    _info(f"found {len(rows)} doctor-marker row(s)")
    return True


def check_streaming_spawn_and_tail() -> bool:
    """spawn a short sleep, tail, then wait for exit."""
    from local_repl.repl import PythonREPL
    from local_repl.streaming import register_streaming_tools
    from mcp.server.fastmcp import FastMCP

    test_mcp = FastMCP(name="doctor-stream", debug=False)
    instances: dict = {}
    register_streaming_tools(test_mcp, instances)
    repl = PythonREPL()
    instances[repl.repl_id] = repl

    spawn = test_mcp._tool_manager.get_tool("spawn_shell").fn
    tail = test_mcp._tool_manager.get_tool("tail_shell_output").fn
    wait = test_mcp._tool_manager.get_tool("wait_shell").fn
    kill = test_mcp._tool_manager.get_tool("kill_shell").fn

    spawned = spawn(
        command="sh -c 'for i in 1 2 3; do echo line-$i; sleep 0.1; done'",
        repl_id=repl.repl_id, use_shell=True,
    )
    assert spawned["ok"], f"spawn failed: {spawned['error']}"
    proc_id = spawned["proc_id"]
    _info(f"spawned proc_id={proc_id[:8]}… pid={spawned['pid']}")

    # Give the drain thread a moment to read the output.
    time.sleep(0.8)

    result = tail(proc_id=proc_id, n_lines=10)
    joined = "\n".join(e["text"] for e in result["lines"])
    assert "line-1" in joined, f"missing line-1 in tail: {joined!r}"
    assert "line-3" in joined, f"missing line-3 in tail: {joined!r}"
    _info(f"tailed {len(result['lines'])} lines, running={result['running']}")

    waited = wait(proc_id=proc_id, timeout_seconds=3)
    assert not waited["timed_out"], "wait timed out — process should have exited"

    # Cleanup (kill is harmless on an already-exited process).
    kill(proc_id=proc_id)
    return True


def check_venv_exec_self() -> bool:
    """run_python_in_venv with our own python (as a stand-in for any venv)."""
    from local_repl.repl import PythonREPL
    from local_repl.venv_exec import register_venv_tools
    from mcp.server.fastmcp import FastMCP
    import sys

    test_mcp = FastMCP(name="doctor-venv", debug=False)
    instances: dict = {}
    register_venv_tools(test_mcp, instances)
    repl = PythonREPL()
    instances[repl.repl_id] = repl

    run = test_mcp._tool_manager.get_tool("run_python_in_venv").fn

    # Use sys.executable — any valid python works for this smoke test.
    result = run(
        code="import sys; print('hello from', sys.version_info[:2])",
        venv_path=sys.executable,
        repl_id=repl.repl_id,
    )
    assert result["ok"], f"exit {result['exit_code']}: {result['stderr']}"
    assert "hello from" in result["stdout"], f"unexpected stdout: {result['stdout']!r}"
    _info(f"executed in {result['duration_ms']:.1f} ms via {result['python_path']}")
    return True


def check_venv_exec_bad_path() -> bool:
    """Non-existent venv path should return structured error, not raise."""
    from local_repl.repl import PythonREPL
    from local_repl.venv_exec import register_venv_tools
    from mcp.server.fastmcp import FastMCP

    test_mcp = FastMCP(name="doctor-venv-bad", debug=False)
    instances: dict = {}
    register_venv_tools(test_mcp, instances)
    repl = PythonREPL()
    instances[repl.repl_id] = repl

    run = test_mcp._tool_manager.get_tool("run_python_in_venv").fn
    result = run(
        code="print('ok')",
        venv_path="/definitely/does/not/exist",
        repl_id=repl.repl_id,
    )
    assert result["ok"] is False
    assert result["error"] and "does not exist" in result["error"]
    return True


def main() -> int:
    _section("LocalREPL doctor — checking unlocks")

    all_checks: list[tuple[str, Callable[[], bool]]] = [
        ("imports clean + cwd attr present", check_imports),
        ("shell_bridge: basic echo", check_shell_basic),
        ("shell_bridge: blocklist (sudo, chained sudo)", check_shell_blocklist),
        ("shell_bridge: timeout → exit 124", check_shell_timeout),
        ("evolution_memory: log_command writes row", check_memory_roundtrip),
        ("evolution_memory: query_command_history finds it", check_memory_query),
        ("streaming: spawn + tail + wait", check_streaming_spawn_and_tail),
        ("venv_exec: run via sys.executable", check_venv_exec_self),
        ("venv_exec: bad path → structured error", check_venv_exec_bad_path),
    ]

    failures = _run_checks(all_checks)
    total = len(all_checks)

    print()
    if failures == 0:
        print(_c(f"All {total} checks passed.", "1;32"))
        print(_c("Safe to restart Claude Desktop.", "32"))
        return 0
    print(_c(f"{failures}/{total} checks failed.", "1;31"))
    print(_c("Fix the failures above before reloading.", "31"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
