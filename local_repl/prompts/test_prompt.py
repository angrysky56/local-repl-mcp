# Add a testing prompt template
def test_prompt_workflow() -> str:
    """
    A testing prompt to verify the external prompts system is working.
    """
    return """
    This is a test prompt to verify that the external prompts system is working.
    
    If you can see this message, it means the prompt was successfully loaded
    from an external file!
    
    This is useful for testing the dynamic loading functionality.
    """
