"""
Wrapper function for running Python code in a REPL.
"""

def run_python_in_repl(repl_id, code):
    """
    Run Python code in the specified REPL.
    
    Args:
        repl_id (str): ID of the REPL to use
        code (str): Python code to execute
        
    Returns:
        str: Output of the execution
    """
    from run_python_in_repl import run_python_in_repl as _run_python_in_repl
    result = _run_python_in_repl(code=code, repl_id=repl_id)
    return result.get('output', '')
