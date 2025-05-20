"""
REPL Server Verification Prompt

This module provides a template for verifying the REPL server functionality.
"""

def basic_system_check() -> str:
    """
    A prompt template to verify that the REPL server is working correctly.
    """
    return """
    # REPL Server Verification

    This prompt verifies that the LocalREPL MCP server is running correctly and all components are functioning as expected.

    ## Basic Functionality Test

    Let's verify that the Python REPL is working properly:

    ```python
    # Step 1: Create a new REPL
    repl_id = create_python_repl()
    print(f"REPL created with ID: {repl_id}")

    # Step 2: Run a simple test
    test_result = run_python_in_repl(
      code='''
    import sys
    import platform

    print("Python version:", platform.python_version())
    print("Platform:", platform.platform())

    # Test some basic calculation
    result = sum(range(10))
    print(f"Sum of numbers 0-9: {result}")

    # Create a simple class
    class TestClass:
        def __init__(self, name):
            self.name = name

        def greet(self):
            return f"Hello from {self.name}!"

    # Test the class
    test = TestClass("REPL Verification")
    print(test.greet())
    ''',
      repl_id=repl_id
    )

    print(test_result)

    # Step 3: Check REPL state
    info = get_repl_info(repl_id)
    print(info)

    # Step 4: Clean up
    delete_repl(repl_id)
    print("Verification test completed successfully!")
    ```

    ## Available Prompts

    The following prompts should be available from the LocalREPL server:

    1. **Python REPL** (`python_repl_workflow`): Basic Python REPL usage guide
    2. **Test Prompt** (`test_prompt_workflow`): Simple test prompt
    3. **MEF Integration** (`mef_integration_workflow`): Modular Empowerment Framework integration guide
    4. **Modular Empowerment** (`modular_empowerment_workflow`): Advanced agent-based framework with persistence
    5. **Data Analysis** (`data_analysis_workflow`): Comprehensive data analysis and visualization workflow
    6. **Verification** (`verification_workflow`): This verification prompt

    ## Configuration Verification

    If all the prompts listed above are available, it means your MCP configuration is correct. You can access all these prompts by typing `/LocalREPL` in Claude Desktop.

    ## Troubleshooting

    If you encounter any issues:

    1. Check that the path in your `mcp_config.json` is correct
    2. Ensure the virtual environment is activated
    3. Verify that all prompt files have the proper return type annotations
    4. Check the server logs for any error messages

    For any persistent issues, please check the troubleshooting section in the README.
    """
