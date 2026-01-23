"""
LocalREPL - A local Python REPL server for Claude Desktop.

This is a standalone file that implements a local Python REPL within an MCP server.
"""
import os
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
        # Initialize a single namespace for environment
        # This is crucial for recursive functions to work properly
        self.namespace: Dict[str, Any] = {'__builtins__': __builtins__}

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
                import ast

                # Parse the code into an AST
                try:
                    tree = ast.parse(code)
                except SyntaxError:
                    # If it's a syntax error, run it to get the standard traceback
                    exec(code, self.namespace, self.namespace)
                    return stdout_capture.getvalue(), stderr_capture.getvalue(), None

                # Check if the last node is an expression
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    # It's an expression! We want to capture its value.

                    # Split into:
                    # 1. Everything EXCEPT the last node (run with exec)
                    # 2. The last node (run with eval)

                    last_node = tree.body[-1]
                    body_nodes = tree.body[:-1]

                    # Execute previous statements if any
                    if body_nodes:
                        module = ast.Module(body=body_nodes, type_ignores=[])
                        exec(compile(module, filename="<string>", mode="exec"), self.namespace, self.namespace)

                    # Evaluate the last expression
                    expr = ast.Expression(body=last_node.value)
                    result = eval(compile(expr, filename="<string>", mode="eval"), self.namespace, self.namespace)
                else:
                    # Last node is not an expression (e.g. assignment, function def)
                    # Just exec the whole thing
                    exec(code, self.namespace, self.namespace)

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
    Execute Python code in a REPL. (Create new or use an existing REPL)

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
    for name, value in repl.namespace.items():
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


# Import prompts from the prompts package
import sys
import os

# Add the current directory to the Python path if it's not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the prompts package
from prompts import prompts

# Print loaded prompts for debugging (to stderr to avoid corrupting JSONRPC on stdout)
print("\n=== Loaded Prompts ===", file=sys.stderr)
for prompt_name in prompts.keys():
    print(f"- {prompt_name}", file=sys.stderr)
print("======================\n", file=sys.stderr)

# Register prompts with MCP using the decorator
for prompt_name, prompt_func in prompts.items():
    # Apply the decorator to each prompt function
    mcp.prompt()(prompt_func)




@mcp.tool()
def initialize_modular_empowerment(repl_id: str) -> str:
    """
    Initialize the Modular Empowerment Framework in a specific REPL.

    Args:
        repl_id: ID of the REPL to initialize in

    Returns:
        str: Result of the initialization
    """
    # Create the initialization code
    init_code = f"""
import sys
import os
import random
import numpy as np
from typing import Dict, List, Any

try:
    # Try importing from the package structure
    try:
        from local_repl.modular_empowerment_framework.src.integration.integration import ModularEmpowermentIntegration
    except ImportError:
        # Fallback for direct execution
        from modular_empowerment_framework.src.integration.integration import ModularEmpowermentIntegration

    # Configure the MEF
    config = {{
        'individual_empowerment_weight': 0.4,
        'group_empowerment_weight': 0.4,
        'mdf_eo_balance': 0.6,
        'mdf_config': {{
            'quality_score_threshold': 0.5
        }},
        'eo_config': {{
            'cooperation_factor': 0.6
        }}
    }}

    # Initialize the integrated framework
    mef = ModularEmpowermentIntegration(config)
    print("✅ Modular Empowerment Framework initialized successfully!")
except ImportError as e:
    print(f"❌ Import error: {{e}}")
    print("Missing dependencies. Please install required packages:")
    print("  pip install numpy")
    raise
except Exception as e:
    print(f"❌ Error initializing framework: {{e}}")
    print("Make sure the framework is properly set up at the specified path.")
    raise
"""

    # Run the initialization code in the specified REPL
    result = run_python_in_repl(code=init_code, repl_id=repl_id)

    # Add success message if initialization was successful
    if "✅ Modular Empowerment Framework initialized successfully!" in result:
        return f"{result}\n\nFramework successfully initialized in REPL {repl_id}."
    else:
        return f"{result}\n\nFailed to initialize framework in REPL {repl_id}."

if __name__ == "__main__":
    mcp.run()
