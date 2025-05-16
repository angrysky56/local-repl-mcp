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
