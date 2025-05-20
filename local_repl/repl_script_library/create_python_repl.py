"""
Wrapper function for creating a new Python REPL.
"""

def create_python_repl():
    """
    Creates a new Python REPL and returns its ID.

    Returns:
        str: ID of the new REPL
    """
    from local_repl.repl_script_library.create_python_repl import create_python_repl as _create_python_repl
    return _create_python_repl()
