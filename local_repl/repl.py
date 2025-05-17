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
