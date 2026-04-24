"""Operational memory backed by evolution.db (SQLite).

Every shell command run via shell_bridge.run_shell is logged here with
exit code, duration, cwd, and truncated stdout/stderr previews. This
gives the agent a queryable history of what worked and what failed —
the same idea as Atuin for human shells, adapted to the MCP context.

Schema is created on first use. Safe to call `log_command` before any
tool is registered; bootstrap is lazy.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


# Co-located with the agent_data folder so db files cluster together.
_DB_PATH = os.path.join(os.path.dirname(__file__), "evolution.db")
_LOCK = threading.Lock()  # sqlite3 is serialized per-connection; we share one


_SCHEMA = """
CREATE TABLE IF NOT EXISTS command_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    repl_id         TEXT,
    command         TEXT    NOT NULL,
    cwd             TEXT,
    exit_code       INTEGER,
    duration_ms     REAL,
    timed_out       INTEGER DEFAULT 0,
    stdout_preview  TEXT,
    stderr_preview  TEXT,
    tags            TEXT    DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_cmd_timestamp ON command_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_cmd_exit_code ON command_log(exit_code);
CREATE INDEX IF NOT EXISTS idx_cmd_repl_id   ON command_log(repl_id);
"""


class CommandLogEntry(TypedDict):
    """One row from the command_log table."""
    id: int
    timestamp: str
    repl_id: str | None
    command: str
    cwd: str | None
    exit_code: int | None
    duration_ms: float | None
    timed_out: bool
    stdout_preview: str | None
    stderr_preview: str | None
    tags: list[str]


class CommandStats(TypedDict):
    """Aggregate stats across all logged commands."""
    total: int
    successes: int
    failures: int
    timeouts: int
    success_rate: float
    top_commands: list[dict[str, Any]]
    top_failures: list[dict[str, Any]]
    slowest: list[dict[str, Any]]
    db_path: str


def _connect() -> sqlite3.Connection:
    """Open the sqlite db, creating it + schema if needed."""
    conn = sqlite3.connect(_DB_PATH, timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _row_to_entry(row: sqlite3.Row) -> CommandLogEntry:
    """Convert a sqlite Row to a CommandLogEntry dict."""
    try:
        tags = json.loads(row["tags"]) if row["tags"] else []
    except (json.JSONDecodeError, TypeError):
        tags = []
    return CommandLogEntry(
        id=row["id"],
        timestamp=row["timestamp"],
        repl_id=row["repl_id"],
        command=row["command"],
        cwd=row["cwd"],
        exit_code=row["exit_code"],
        duration_ms=row["duration_ms"],
        timed_out=bool(row["timed_out"]),
        stdout_preview=row["stdout_preview"],
        stderr_preview=row["stderr_preview"],
        tags=tags,
    )


def _parse_since(since: str | None) -> str | None:
    """Accept '1h', '30m', '2d', or ISO timestamp. Return ISO or None."""
    if not since:
        return None
    since = since.strip()
    # Relative spec: <number><unit> where unit in m/h/d
    if since and since[-1] in "mhd" and since[:-1].isdigit():
        qty = int(since[:-1])
        delta = {"m": timedelta(minutes=qty),
                 "h": timedelta(hours=qty),
                 "d": timedelta(days=qty)}[since[-1]]
        return (datetime.now(timezone.utc) - delta).isoformat()
    return since  # assume caller provided ISO


def log_command(
    repl_id: str | None,
    command: str,
    cwd: str | None,
    exit_code: int,
    duration_ms: float,
    stdout_preview: str = "",
    stderr_preview: str = "",
    timed_out: bool = False,
    tags: list[str] | None = None,
) -> int | None:
    """Append a command-execution record to the log.

    Returns the new row id, or None on failure. This function never raises;
    callers treat logging as best-effort so a broken db can't break tools.
    """
    try:
        with _LOCK:
            conn = _connect()
            try:
                cur = conn.execute(
                    """INSERT INTO command_log
                       (timestamp, repl_id, command, cwd, exit_code,
                        duration_ms, timed_out, stdout_preview,
                        stderr_preview, tags)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        repl_id, command, cwd, exit_code,
                        float(duration_ms), 1 if timed_out else 0,
                        stdout_preview, stderr_preview,
                        json.dumps(tags or []),
                    ),
                )
                return cur.lastrowid
            finally:
                conn.close()
    except Exception:
        return None


def register_memory_tools(mcp: "FastMCP") -> None:
    """Register evolution.db query tools on the given MCP instance."""

    @mcp.tool()
    def query_command_history(
        pattern: str | None = None,
        repl_id: str | None = None,
        only_failures: bool = False,
        only_timeouts: bool = False,
        since: str | None = None,
        limit: int = 20,
    ) -> list[CommandLogEntry]:
        """Search the command execution log.

        Args:
            pattern: SQL LIKE pattern against the command text (case-insensitive).
                Use % as wildcard, e.g. '%ripgrep%' or 'git commit%'.
            repl_id: Filter to commands run in a specific REPL.
            only_failures: Return only non-zero exit codes.
            only_timeouts: Return only commands that hit the timeout.
            since: ISO timestamp, or relative like '1h', '30m', '2d'.
            limit: Max rows to return (capped at 200).

        Returns:
            List of CommandLogEntry dicts, newest first.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if pattern:
            clauses.append("LOWER(command) LIKE LOWER(?)")
            params.append(pattern)
        if repl_id:
            clauses.append("repl_id = ?")
            params.append(repl_id)
        if only_failures:
            clauses.append("exit_code != 0")
        if only_timeouts:
            clauses.append("timed_out = 1")
        since_iso = _parse_since(since)
        if since_iso:
            clauses.append("timestamp >= ?")
            params.append(since_iso)

        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        capped_limit = max(1, min(int(limit), 200))
        sql = (
            f"SELECT * FROM command_log{where_sql} "
            f"ORDER BY id DESC LIMIT ?"
        )
        params.append(capped_limit)

        with _LOCK:
            conn = _connect()
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        return [_row_to_entry(r) for r in rows]

    @mcp.tool()
    def command_stats(
        since: str | None = None,
        top_n: int = 5,
    ) -> CommandStats:
        """Aggregate metrics across the command log.

        Args:
            since: Window to analyze (ISO or '1h'/'1d' etc). None = all time.
            top_n: How many entries to return in each top-list.
        """
        since_iso = _parse_since(since)
        where_sql = " WHERE timestamp >= ?" if since_iso else ""
        params: tuple = (since_iso,) if since_iso else ()
        capped_n = max(1, min(int(top_n), 25))

        with _LOCK:
            conn = _connect()
            try:
                total = conn.execute(
                    f"SELECT COUNT(*) AS n FROM command_log{where_sql}",
                    params,
                ).fetchone()["n"]
                successes = conn.execute(
                    f"SELECT COUNT(*) AS n FROM command_log"
                    f"{where_sql}{' AND' if where_sql else ' WHERE'} exit_code = 0",
                    params,
                ).fetchone()["n"]
                timeouts = conn.execute(
                    f"SELECT COUNT(*) AS n FROM command_log"
                    f"{where_sql}{' AND' if where_sql else ' WHERE'} timed_out = 1",
                    params,
                ).fetchone()["n"]
                top_cmds = conn.execute(
                    f"SELECT command, COUNT(*) AS n FROM command_log"
                    f"{where_sql} GROUP BY command "
                    f"ORDER BY n DESC LIMIT ?",
                    (*params, capped_n),
                ).fetchall()
                top_fails = conn.execute(
                    f"SELECT command, exit_code, COUNT(*) AS n "
                    f"FROM command_log"
                    f"{where_sql}{' AND' if where_sql else ' WHERE'} exit_code != 0 "
                    f"GROUP BY command, exit_code "
                    f"ORDER BY n DESC LIMIT ?",
                    (*params, capped_n),
                ).fetchall()
                slowest = conn.execute(
                    f"SELECT id, command, duration_ms, exit_code "
                    f"FROM command_log{where_sql} "
                    f"ORDER BY duration_ms DESC LIMIT ?",
                    (*params, capped_n),
                ).fetchall()
            finally:
                conn.close()

        failures = total - successes
        return CommandStats(
            total=total,
            successes=successes,
            failures=failures,
            timeouts=timeouts,
            success_rate=round(successes / total, 3) if total else 0.0,
            top_commands=[dict(r) for r in top_cmds],
            top_failures=[dict(r) for r in top_fails],
            slowest=[dict(r) for r in slowest],
            db_path=_DB_PATH,
        )

    @mcp.tool()
    def tag_command(command_id: int, tags: list[str]) -> dict:
        """Attach tags to a logged command for later recall.

        Useful for marking commands as e.g. 'flaky', 'slow', 'solved',
        'security-review'. Replaces any existing tags on that row.
        """
        with _LOCK:
            conn = _connect()
            try:
                cur = conn.execute(
                    "UPDATE command_log SET tags = ? WHERE id = ?",
                    (json.dumps(tags), command_id),
                )
                if cur.rowcount == 0:
                    return {"ok": False, "error": f"no row with id {command_id}"}
            finally:
                conn.close()
        return {"ok": True, "id": command_id, "tags": tags}

    @mcp.tool()
    def forget_command(command_id: int) -> dict:
        """Delete a command log row. Use when a logged command captured
        a secret (token in argv, etc.) that shouldn't persist on disk.
        """
        with _LOCK:
            conn = _connect()
            try:
                cur = conn.execute(
                    "DELETE FROM command_log WHERE id = ?",
                    (command_id,),
                )
                deleted = cur.rowcount
            finally:
                conn.close()
        if deleted == 0:
            return {"ok": False, "error": f"no row with id {command_id}"}
        return {"ok": True, "deleted_id": command_id}

    @mcp.tool()
    def vacuum_memory(keep_last_n: int = 5000) -> dict:
        """Keep only the most recent N rows and reclaim disk space.
        Call occasionally to prevent unbounded growth.
        """
        keep = max(100, int(keep_last_n))
        with _LOCK:
            conn = _connect()
            try:
                total = conn.execute(
                    "SELECT COUNT(*) AS n FROM command_log"
                ).fetchone()["n"]
                if total > keep:
                    conn.execute(
                        "DELETE FROM command_log WHERE id NOT IN "
                        "(SELECT id FROM command_log ORDER BY id DESC LIMIT ?)",
                        (keep,),
                    )
                conn.execute("VACUUM")
                deleted = max(0, total - keep)
            finally:
                conn.close()
        return {"ok": True, "rows_deleted": deleted, "rows_kept": min(total, keep)}
