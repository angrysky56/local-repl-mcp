# ast-grep patterns — correct syntax

ast-grep's pattern language is specific. Wrong patterns return **zero matches with no error message** — the worst possible failure mode. This file is the reference for getting it right.

## Metavariable syntax

SyntaxBinds to`$NAME`A single node (identifier, expression, statement, etc.)`$$$`A *sequence* of nodes — specifically, argument lists, parameter lists, or body statements`$$$ARGS`Same as `$$$` but captured under name `ARGS$_`Any single node, not captured (wildcard)

**The critical distinction:** `$$$` does NOT mean "any body." It means "sequence of peer nodes." Function bodies are not a sequence — they're a `block` node. This is why `def $NAME($$$): $$$` matches nothing.

## Correct patterns by target

### Python

```
# Match ANY function definition (by name)
def $NAME

# Match a specific decorator
@mcp.tool()

# Match a call with any arguments
print($$$)
logger.$METHOD($$$)

# Match an import
import $MOD
from $MOD import $$$

# Match a class definition
class $NAME

# Match a type-annotated variable
$VAR: $TYPE = $VALUE
```

### TypeScript / JavaScript

```
# Arrow function assignment
const $NAME = ($$$) => $$$

# React hook call
use$HOOK($$$)

# Interface declaration
interface $NAME { $$$ }
```

### Rust

```
# Function with any signature
fn $NAME($$$) -> $RET { $$$ }

# Macro invocation
$MACRO!($$$)
```

## Anti-patterns (things that silently fail)

```
# ✗ matches NOTHING — $$$ can't bind to function body
def $NAME($$$): $$$

# ✗ matches NOTHING — $$$ inside braces is not a block
class $NAME: $$$

# ✓ correct — match by name alone, the rest is implied
def $NAME
class $NAME

# ✗ matches NOTHING — dot operator not allowed bare
$obj.$method

# ✓ correct — wrap as a call
$obj.$method($$$)
```

## Debugging zero matches

When a pattern returns no matches and you're sure there should be some:

1. Drop all metavariables and match a literal snippet first. If the literal doesn't match, the `--lang` is wrong or the file isn't being parsed.
2. Use `ast-grep scan --debug-query -p 'PATTERN'` to see how ast-grep is interpreting the pattern.
3. Look at the tree-sitter parse of a known-matching file: `ast-grep --lang python --pattern '$_' file.py --json=compact | jq '.[0]'` shows the node hierarchy.
4. Simplify: `def $NAME` (match any function) is safer than `def $NAME($X, $Y)` (match exactly two params) when you don't know the arg count.

## Rewriting code

`--rewrite` replaces matches in-place. Always dry-run first by omitting `--update-all`:

```bash
# Preview
ast-grep --pattern 'os.path.join($$$)' --rewrite 'Path($$$)' \
  --lang python src/ --json=compact | jq 'length'
# → 23 (shows how many would change)

# Apply
ast-grep --pattern 'os.path.join($$$)' --rewrite 'Path($$$)' \
  --lang python src/ --update-all
```

Note: semantic refactors (like os.path → pathlib) still need human review — ast-grep doesn't import `Path` for you.

## YAML rules for repeated patterns

For patterns used repeatedly, put them in an `sgconfig.yml` / rule file instead of inline:

```yaml
id: no-bare-print
language: python
rule:
  pattern: print($MSG)
fix: logger.debug($MSG)
message: Replace bare print() with structured logging
```

Then: `ast-grep scan --rule rule.yml src/` — batchable, CI-friendly.
