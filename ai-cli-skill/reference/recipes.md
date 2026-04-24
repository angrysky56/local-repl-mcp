# ai-cli recipes

Copy-paste multi-tool pipelines. Each assumes a REPL with `set_repl_cwd` already pointed at the project root.

## Codebase first-touch (new repo, unknown size)

Run this before doing anything else in an unfamiliar codebase. One call returns: size by language, total files, size on disk, git status summary.

```bash
echo "=== Languages & lines ==="; \
tokei --output json --exclude '.venv' --exclude 'target' --exclude 'node_modules' \
  | jq 'to_entries | map({lang: .key, code: .value.code, files: (.value.reports | length)}) | sort_by(-.code) | .[0:5]'; \
echo; \
echo "=== Tracked files ==="; git ls-files | wc -l; \
echo; \
echo "=== Git status ==="; git status -s | head -20; \
echo; \
echo "=== Disk usage ==="; du -sh . --exclude='.venv' --exclude='target' --exclude='node_modules' 2>/dev/null
```

Expected output: top 5 languages by line count, a count of tracked files, a short dirty-file list, and a one-line disk size. Everything an agent needs to orient itself in \~100ms.

## Structural count of code patterns

Count decorators, classes, function calls, imports — anything ast-grep can match as a tree node.

```bash
# Count @mcp.tool() decorators across the modules
ast-grep --pattern '@mcp.tool()' --lang python src/ --json=compact \
  | jq 'length as $n | "\($n) @mcp.tool() registrations"'

# Count function definitions per file
for f in $(fd -e py . src/); do
  n=$(ast-grep --pattern 'def $NAME' --lang python "$f" --json=compact | jq length)
  printf '%-40s %s\n' "$f" "$n"
done | sort -k2 -rn | head -10
```

## Safe refactor preview (dry-run before apply)

ast-grep's `--rewrite` supports dry-run. Always preview first.

```bash
# Preview: what would this change?
ast-grep --pattern 'print($MSG)' --rewrite 'logger.debug($MSG)' \
  --lang python src/ --json=compact \
  | jq '.[].text' | head -20

# After reviewing — actually apply
ast-grep --pattern 'print($MSG)' --rewrite 'logger.debug($MSG)' \
  --lang python src/ --update-all
```

## Diff reading with delta

When you need to inspect a diff, pipe git output through delta. Delta respects side-by-side mode and syntax highlighting.

```bash
git diff HEAD --stat    # summary first
git diff HEAD | delta --side-by-side --line-numbers | head -80
```

## Reactive test loop (watchexec)

Re-run tests whenever Python files change. `spawn_shell` + `tail_shell_output(drain=True)` gives a polling-friendly interface.

```python
# In one call
proc = spawn_shell(
    command="watchexec -e py -c -- pytest -x tests/",
    repl_id=repl_id,
    use_shell=True,
)
proc_id = proc["proc_id"]

# Later — poll for new output without blocking
tail = tail_shell_output(proc_id=proc_id, n_lines=50, drain=True)
# `drain=True` means each call returns ONLY new lines — perfect for polling
```

`-e py` watches Python files only. `-c` clears the terminal between runs. `-x` on pytest stops at first failure.

Stop with `kill_shell(proc_id)` when done.

## Benchmarking (hyperfine)

Statistical benchmarking with warmup runs and JSON export.

```bash
hyperfine --warmup 3 --runs 10 --export-json /tmp/bench.json \
  'rg PATTERN src/' \
  'grep -r PATTERN src/' \
  'ag PATTERN src/'

jq '.results | map({cmd: .command, mean_ms: (.mean * 1000 | floor), stddev_ms: (.stddev * 1000 | floor)})' /tmp/bench.json
```

Output is a ranked list with mean and stddev — meaningful comparison, not anecdotal timing.

## JSON/API response drilling

When a command returns a large nested JSON blob, never dump the whole thing. Extract only the leaf fields the user asked about.

```bash
# Bad: dumps the entire tokei tree
tokei --output json

# Good: extract exactly what the user asked for
tokei --output json | jq '.Python.code'   # → 8969

# Filtering logs by level + recency
journalctl -o json --since '1 hour ago' | jq 'select(.PRIORITY <= "3") | {ts: .__REALTIME_TIMESTAMP, unit: ._SYSTEMD_UNIT, msg: .MESSAGE}' | head -20
```

The jq filter is the *point* of the query — it's what turns "a wall of data" into "the three fields that matter."
