# Environment gotchas

These are all real failure modes that bit someone (usually me) during development. None of them produce clear error messages — they manifest as hangs, silent zero-matches, or "command not found" despite the tool being installed. Read this before debugging anything weird.

## Shell invocation

### `use_shell=True` runs `/bin/sh`, not bash

On Debian/Ubuntu `/bin/sh` is `dash` — a POSIX-only shell. Features that break:

| Bash-only | POSIX-compatible alternative |
|---|---|
| `{a,b,c}` brace expansion | `for x in a b c; do ...; done` |
| `[[ ... ]]` | `[ ... ]` (with quoting!) |
| `<(command)` process substitution | Temp file: `command > /tmp/x && ...` |
| `source` | `.` (dot) |
| `function foo()` | `foo() { ... }` |

If you need bash features, invoke bash explicitly:

```python
run_shell(command="bash -c '...'", repl_id=..., use_shell=True)
```

Or just don't use shell features — `use_shell=False` with tokenized args is often cleaner anyway.

### Prefer `use_shell=False` for single commands

The default `use_shell=False` uses `shlex.split` + `exec`. No shell metacharacter interpretation — safer and usually faster. Only use `use_shell=True` when you actually need pipes, redirects, glob expansion, or command chaining.

## PATH inheritance

### MCP servers inherit a minimal PATH

When an MCP server launches via launchd (macOS) or systemd (Linux), it doesn't read your `.bashrc` / `.zshrc` / nvm init scripts. So `$PATH` inside the server's subprocesses is typically just `/usr/local/bin:/usr/bin:/bin` plus maybe `~/.local/bin` and `~/.cargo/bin`.

**Symptom:** Tool works fine in your terminal, but `run_shell("ast-grep --version")` returns `exit_code: 127`.

**Fix:** Symlink the binary into `~/.local/bin` (which is usually on PATH):

```bash
ln -sf "$(which THE_TOOL)" ~/.local/bin/THE_TOOL
```

Or pass PATH explicitly via `env_extra`:

```python
run_shell(
  command="ast-grep --version",
  repl_id=...,
  env_extra={"PATH": "/home/user/.nvm/versions/node/v22.21.0/bin:/usr/bin:/bin"},
)
```

The symlink approach is usually cleaner — set once, works everywhere.

## Ripgrep's stdin heuristic

### Symptom: `rg` times out on what should be a trivial search

Ripgrep has a heuristic: if stdin appears readable, it reads from stdin *instead* of walking the directory tree. When an MCP server passes its own stdin down to child processes, rg sits waiting forever for input that will never come.

**The LocalREPL `shell_bridge` already handles this** by passing `stdin=subprocess.DEVNULL` to every subprocess.run call. If you're writing your own subprocess runner, do the same.

Debug signature in rg's output: `--debug` shows `is_readable_stdin=true, stdin_consumed=false, mode=Files` right before the hang.

Workaround outside LocalREPL: explicit `rg ... < /dev/null`.

## MCP JSON-string coercion on stdin_input

### Symptom: Pydantic `string_type` error on valid-looking string

The MCP tool-call transport auto-parses JSON-looking strings into dicts/lists before passing them to the tool. This is helpful for object arguments but breaks `stdin_input: str` because the validator rejects a dict.

```python
# ✗ Fails validation — the transport turns this into a dict
run_shell(command="jq .a", stdin_input='{"a": 1}', ...)

# ✓ Workaround 1: shell-pipe via echo
run_shell(command="echo '{\"a\":1}' | jq .a", use_shell=True, ...)

# ✓ Workaround 2: prefix with a non-JSON character to force string
run_shell(command="jq '. | .[1:] | fromjson | .a'", stdin_input=' {"a": 1}', ...)

# ✓ Workaround 3: plain text without braces works fine
run_shell(command="wc -l", stdin_input="line one\nline two\n", ...)
```

The echo-pipe workaround is usually the cleanest.

## Silent zero-matches

### ast-grep pattern

Already covered in `ast-grep-patterns.md`. Summary: `def $NAME($$$): $$$` matches nothing; use `def $NAME`.

### ripgrep scoping

`rg PATTERN` with no path argument searches `./` — which can include massive ignored-but-followed symlinks (`.venv`, `target`, cache dirs). Ripgrep respects `.gitignore` but not always `.venv` unless it's in a gitignore'd location.

**Always scope** broad searches: `rg PATTERN --type py src/`, not `rg PATTERN --type py`.

When a search takes >5 seconds, cancel and scope. Don't wait.

## Edit-drift recovery

### Symptom: A file looks wildly shorter/different than expected after a series of edits

This happens when edit tools accumulate pending changes that flush in a batch. If the batch included fuzzy-match failures that silently skipped sections, you end up with partial content.

**Recovery in one command** (assumes the file is tracked in git):

```bash
git show HEAD:path/to/file > path/to/file
```

This restores the last-committed version exactly, bypassing any merge dialogs or staged-state weirdness.

**Prevention:** Commit before any multi-edit session. A clean `git status` before and after each batch of changes makes recovery trivial.

This isn't paranoia — it's already happened once in this repo.

## Cargo installs that fail on stable Rust

### Symptom: `error[E0658]: use of unstable library feature`

Some crates accidentally use nightly features without gating them. Example: `watchexec-cli 2.5.1` uses `fmt::from_fn` which is pending stabilization in Rust 1.96.

**Workaround priority:**
1. Check the project's GitHub releases for prebuilt binaries (`.deb`, `.tar.xz`). Most well-maintained Rust CLIs ship them.
2. Pin to an older version: `cargo install toolname --version X.Y.Z`
3. Only as a last resort, install nightly via rustup and use `cargo +nightly install`.

For watchexec specifically: the `.deb` at https://github.com/watchexec/watchexec/releases works fine.

## File not found despite `which` showing it

### Symptom: `/bin/sh: 1: TOOL: not found` even though `which TOOL` from your shell shows a valid path

This is the same as the PATH inheritance issue above, but worth naming separately because the error message is misleading. The tool isn't "missing" — it's just not on the subprocess's PATH.

Always fix by symlinking into `~/.local/bin`, never by hardcoding full paths in scripts. Hardcoded paths don't survive nvm version bumps, Python venv changes, or user-home moves.
