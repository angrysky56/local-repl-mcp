"""
The Python REPL implementation that maintains state between executions.
"""
import io
import sys
import traceback
import uuid
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Tuple, Optional, Any


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
