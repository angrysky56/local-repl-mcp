"""
Wrapper function for deleting a REPL.
"""

def delete_repl(repl_id):
    """
    Delete a REPL instance.
    
    Args:
        repl_id (str): ID of the REPL to delete
        
    Returns:
        str: Confirmation message
    """
    from delete_repl import delete_repl as _delete_repl
    return _delete_repl(repl_id=repl_id)
