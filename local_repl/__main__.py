"""
LocalREPL - A local Python REPL server for Claude Desktop.

This module can be used directly via `python -m local_repl` or by importing the mcp 
server instance from `local_repl` package.
"""
from .server import mcp, main

if __name__ == "__main__":
    main()
