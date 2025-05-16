"""
LocalREPL - A local Python REPL server for Claude Desktop.

This is a standalone file that implements a local Python REPL within an MCP server.
"""
from typing import Dict, Tuple, Optional, Any
from mcp.server.fastmcp import FastMCP
import io
import sys
import traceback
import uuid
from contextlib import redirect_stdout, redirect_stderr


class PythonREPL:
    """
    A stateful Python REPL implementation that maintains separate environment for each instance.
    """
    def __init__(self, repl_id: Optional[str] = None):
        self.repl_id = repl_id or str(uuid.uuid4())
        # Initialize a clean environment
        self.globals: Dict[str, Any] = {'__builtins__': __builtins__}
        self.locals: Dict[str, Any] = {}
    
    def execute(self, code: str) -> Tuple[str, str, Any]:
        """
        Execute Python code in the REPL and return stdout, stderr, and the result.
        
        Args:
            code: The Python code to execute
            
        Returns:
            Tuple of (stdout, stderr, result)
        """
        # Use StringIO for capturing output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = None
        
        # Make sure code is a string to avoid issues
        if not isinstance(code, str):
            return ("", "Error: Code must be a string", None)
            
        # Skip empty code
        if not code.strip():
            return ("", "", None)
        
        try:
            # Redirect stdout and stderr to our capture objects
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # First try to evaluate as an expression (for return value)
                try:
                    # Use eval for expressions that can return a value
                    result = eval(code, self.globals, self.locals)
                except SyntaxError:
                    # If it's not a valid expression, execute as a statement using exec
                    exec(code, self.globals, self.locals)
                # Update locals in globals to maintain state between calls
                self.globals.update(self.locals)
        except Exception:
            # Catch any exceptions and add to stderr
            stderr_capture.write(traceback.format_exc())
            
        # Return the captured output and result
        return (
            stdout_capture.getvalue(),
            stderr_capture.getvalue(),
            result
        )


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


if __name__ == "__main__":
    mcp.run()
