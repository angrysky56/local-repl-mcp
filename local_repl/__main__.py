"""
LocalREPL - A local Python REPL server for Claude Desktop.

This module can be used directly via `python -m local_repl` or by importing the mcp 
server instance from `local_repl` package.
"""
from typing import Dict
from mcp.server.fastmcp import FastMCP
from .repl import PythonREPL


# Dictionary to store REPL instances by ID
repl_instances: Dict[str, PythonREPL] = {}

# Create the MCP server with more robust settings for connection handling
mcp = FastMCP(
    name="LocalREPL",
    stateless_http=False,
    debug=False  # Disable debug mode to reduce potential connection issues
)


@mcp.tool()
def create_python_repl() -> str:
    """
    Create a new Python REPL environment.
    
    Returns:
        str: ID of the new REPL
    """
    repl = PythonREPL()
    repl_instances[repl.repl_id] = repl
    return repl.repl_id


@mcp.tool()
def run_python_in_repl(code: str, repl_id: str) -> str:
    """
    Execute Python code in a REPL.
    
    Args:
        code: Python code to execute
        repl_id: ID of the REPL to use
        
    Returns:
        str: Result of the execution including stdout, stderr, and the return value
    """
    repl = repl_instances.get(repl_id)
    if not repl:
        return f"Error: REPL with ID {repl_id} not found. Please create a new REPL first."
    
    # Execute the code
    stdout, stderr, result = repl.execute(code)
    
    # Format the response - keep it simple to minimize potential issues
    output = ""
    
    # Add stdout if present
    if stdout:
        output += f"--- Output ---\n{stdout}\n\n"
    
    # Add stderr if present
    if stderr:
        output += f"--- Error ---\n{stderr}\n\n"
    
    # Add result if it's not None
    if result is not None:
        output += f"--- Result ---\n{result!r}"
    
    # Return response or default message
    return output.strip() if output.strip() else "Code executed successfully with no output."


@mcp.tool()
def list_active_repls() -> str:
    """List all active REPL instances and their IDs."""
    if not repl_instances:
        return "No active REPL instances."
    
    active_repls = [f"- {repl_id}" for repl_id in repl_instances]
    return "Active REPL instances:\n" + "\n".join(active_repls)


@mcp.tool()
def get_repl_info(repl_id: str) -> str:
    """
    Get information about a specific REPL instance.
    
    Args:
        repl_id: ID of the REPL to get info for
        
    Returns:
        str: Information about the REPL
    """
    repl = repl_instances.get(repl_id)
    if not repl:
        return f"Error: REPL with ID {repl_id} not found."
    
    # Get information about defined variables
    variables = []
    for name, value in repl.globals.items():
        if not name.startswith("__"):
            try:
                type_name = type(value).__name__
                variables.append(f"{name}: {type_name}")
            except:
                variables.append(f"{name}: <unknown type>")
    
    if variables:
        return f"REPL ID: {repl_id}\nDefined variables:\n" + "\n".join(f"- {var}" for var in variables)
    else:
        return f"REPL ID: {repl_id}\nNo user-defined variables yet."


@mcp.tool()
def delete_repl(repl_id: str) -> str:
    """
    Delete a REPL instance.
    
    Args:
        repl_id: ID of the REPL to delete
        
    Returns:
        str: Confirmation message
    """
    if repl_id in repl_instances:
        del repl_instances[repl_id]
        return f"REPL {repl_id} deleted successfully."
    else:
        return f"REPL {repl_id} not found."


@mcp.prompt()
def python_repl_workflow() -> str:
    """
    A prompt template showing how to use the Python REPL.
    """
    return """
    To run Python code:
    
    1. First create a new REPL:
       ```
       repl_id = create_python_repl()
       ```
    
    2. Then run code in the REPL:
       ```
       result = run_python_in_repl(
         code="import numpy as np\\nprint(np.random.randn(5))",
         repl_id=repl_id
       )
       ```
    
    3. You can continue running code in the same REPL to build on previous results:
       ```
       more_results = run_python_in_repl(
         code="print('Previous numpy import is still available')",
         repl_id=repl_id
       )
       ```
    """


# Create module runnable
def main():
    mcp.run()


if __name__ == "__main__":
    main()
