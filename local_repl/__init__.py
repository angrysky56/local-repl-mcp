"""
LocalREPL: A local Python REPL for Claude Desktop
"""
from local_repl.server import mcp

# Expose the MCP server instance
__all__ = ['mcp']

if __name__ == "__main__":
    # Direct execution just runs the server
    mcp.run()
