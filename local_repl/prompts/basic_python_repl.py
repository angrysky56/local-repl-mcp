"""
Python REPL prompt template.

This module provides the template for the Python REPL workflow prompt.
"""

def basic_python_repl_workflow() -> str:
    """
    A prompt template showing how to use the Python REPL.
    """
    return """
    This is a local Python REPL server. You can create, run, and manage Python REPL instances.

    Available commands:

    create_python_repl() - Creates a new Python REPL and returns its ID
    run_python_in_repl(code, repl_id) - Runs Python code in the specified REPL
    list_active_repls() - Lists all active REPL instances
    get_repl_info(repl_id) - Shows information about a specific REPL
    delete_repl(repl_id) - Deletes a REPL instance

    To run Python code:

    # First create a new REPL
    repl_id = create_python_repl()

    # Run some code
    result = run_python_in_repl(
      code="x = 42\\nprint(f'The answer is {x}')",
      repl_id=repl_id
    )

    # Run more code in the same REPL (with state preserved)
    more_results = run_python_in_repl(
      code="import math\\nprint(f'The square root of {x} is {math.sqrt(x)}')",
      repl_id=repl_id
    )

    # Check what variables are available in the environment
    environment_info = get_repl_info(repl_id)

    # When done, you can delete or keep the REPL
    delete_repl(repl_id)
    """
