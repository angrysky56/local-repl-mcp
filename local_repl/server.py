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
                # For multi-line code blocks or statements, use exec
                # For single expressions where we want a return value, use eval

                # Check if we're dealing with a single expression or a code block
                code_stripped = code.strip()

                # First try using exec for all code (safer for multi-line code)
                # Use a single namespace for both globals and locals to support recursive functions
                exec(code, self.namespace, self.namespace)

                # If it's potentially a simple expression, also try to evaluate it to get a return value
                # Only do this for single-line code without assignments or imports
                if ('\n' not in code_stripped and
                    '=' not in code_stripped and
                    not code_stripped.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while '))):
                    try:
                        result = eval(code_stripped, self.namespace, self.namespace)
                    except:
                        # Ignore errors in eval since we already executed with exec
                        pass
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

# Print loaded prompts for debugging
print("\n=== Loaded Prompts ===")
for prompt_name in prompts.keys():
    print(f"- {prompt_name}")
print("======================\n")

# Register prompts with MCP using the decorator
for prompt_name, prompt_func in prompts.items():
    # Apply the decorator to each prompt function
    mcp.prompt()(prompt_func)

@mcp.tool()
def setup_modular_empowerment(path: str = "") -> str:
    """
    Set up the Modular Empowerment Framework with a user-specified path.

    Args:
        path: Full path to the modular_empowerment_framework directory
              If empty, will look in the current directory

    Returns:
        str: Confirmation message with setup status
    """
    global mef_path_store

    if not path:
        # If no path provided, try the current directory
        path = os.path.join(os.getcwd(), "modular_empowerment_framework")
        if not os.path.exists(path):
            return (
                "❌ No path provided and couldn't find modular_empowerment_framework in the current directory.\n"
                "Please provide the full path to the modular_empowerment_framework directory."
            )

    # Check if the path exists
    if not os.path.exists(path):
        return f"❌ The specified path does not exist: {path}"

    # Check if it looks like a proper modular_empowerment_framework directory
    # We'll just check for a src directory as a basic validation
    src_dir = os.path.join(path, "src")
    if not os.path.exists(src_dir):
        return (
            f"⚠️  Warning: The path {path} exists but doesn't look like a modular_empowerment_framework directory.\n"
            "It should contain a 'src' directory with the framework code.\n"
            "You may need to create this structure or specify a different path."
        )

    # Store the path for future use
    mef_path_store = path

    return f"✅ Successfully configured modular_empowerment_framework at: {path}"


@mcp.tool()
def initialize_modular_empowerment(repl_id: str) -> str:
    """
    Initialize the Modular Empowerment Framework in a specific REPL.

    Args:
        repl_id: ID of the REPL to initialize in

    Returns:
        str: Result of the initialization
    """
    global mef_path_store

    # Check if the REPL exists
    if repl_id not in repl_instances:
        return f"❌ REPL with ID {repl_id} not found. Please create a new REPL first."

    # Check if the MEF path is set
    if not mef_path_store:
        return (
            "❌ The modular_empowerment_framework path is not set.\n"
            "Please run setup_modular_empowerment(path) first to configure the path."
        )

    # Create the initialization code
    init_code = f"""
import sys
import os
import random
import numpy as np
from typing import Dict, List, Any

# Add the framework directory to Python path
mef_path = "{mef_path_store}"
sys.path.append(mef_path)
print(f"Added framework path to Python path: {{mef_path}}")

try:
    # Import the integration module
    from src.integration.integration import ModularEmpowermentIntegration

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
