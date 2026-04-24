# local-repl-mcp task recipes
# Run `just` or `just --list` to see available tasks.
# Any recipe can be run by name, e.g. `just doctor`.

# Default recipe when called with no args: list available tasks
default:
    @just --list --unsorted

# Run the smoke-test doctor script (verifies all unlocks still work)
doctor:
    uv run python -m local_repl.doctor

# Tokei code-statistics summary (top 5 languages by line count)
stats:
    @tokei --output json --exclude '.venv' --exclude 'target' \
        | jq 'to_entries | map({lang: .key, code: .value.code, files: (.value.reports | length)}) | sort_by(-.code) | .[0:5]'

# Count @mcp.tool() decorators across the new modules (structural match)
count-tools:
    @ast-grep --pattern '@mcp.tool()' --lang python \
        local_repl/shell_bridge.py \
        local_repl/evolution_memory.py \
        local_repl/streaming.py \
        --json=compact | jq 'length as $n | "\($n) @mcp.tool() registrations"'

# Show evolution.db size and recent row count
db-stats:
    @ls -lh local_repl/evolution.db 2>/dev/null || echo "evolution.db not yet created"
    @sqlite3 local_repl/evolution.db 'SELECT COUNT(*) AS rows, MAX(timestamp) AS latest FROM command_log' 2>/dev/null || true

# Syntax-check all new modules (no exec, just AST parse)
check:
    @for f in local_repl/shell_bridge.py local_repl/evolution_memory.py local_repl/streaming.py local_repl/doctor.py; do \
        python3 -c "import ast; ast.parse(open('$f').read())" && echo "✓ $f" || echo "✗ $f"; \
    done

# Clean up caches (keeps evolution.db intact)
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
    @echo "caches cleared (evolution.db preserved)"
