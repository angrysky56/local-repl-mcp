"""
Local REPL Agent System.

This module provides the template for the Modular Empowerment Framework workflow prompt.
"""

def repl_agent_system_prompt() -> str:
    """
    Prompt template showing AI how to use the REPL Agent System.
    """
    return """

    # Local REPL Agent System

    This system helps you run Python code, manage data analysis workflows, and create reusable scripts in an organized way. All outputs are automatically saved to the appropriate directories for future reference.

    ## Directory Structure
    - /request-path-from-user-if-not-known/local-repl-mcp/local_repl/ (Base directory)
      - /output/ (All generated files)
        - /visualizations/ (Generated charts and plots)
        - /data/ (Processed data files)
        - /results/ (Analysis results and reports)
      - /repl_agent_library/ (Reusable agent scripts)
      - /repl_script_library/ (Reusable REPL scripts)
      - /prompts/ (System prompts)

    ## Package Installation
    Always use the UV package manager with the virtual environment:
    `. .venv/bin/activate && uv pip install <required-packages>`

    ## REPL Management
    - Create a new REPL before starting any analysis (`create_python_repl()`)
    - Always save the REPL ID and provide it to subsequent commands (`run_python_in_repl(code, repl_id)`)
    - List active REPLs with `list_active_repls()`
    - Get REPL info with `get_repl_info(repl_id)`
    - Delete REPLs when finished to free up resources (`delete_repl(repl_id)`)
    - Save reusable code as scripts in the appropriate library folders

    ## File Organization
    - Save all outputs to the /output directory
    - Create subdirectories as needed for different types of outputs
    - Use consistent naming conventions (project_name_date_description)

    ## Modular Empowerment Framework Integration
    - Setup the MEF by requesting the path from the user on start and setting it if not certain i.e.:
     `setup_modular_empowerment(path="/home/ty/Repositories/ai_workspace/local-repl-mcp/modular_empowerment_framework")`
    - Initialize the MEF in a REPL before using its features: `initialize_modular_empowerment(repl_id)`
    - Always confirm the MEF path with the user before setup

    ## Error Handling
    - Check if directories exist before trying to save files
    - Verify REPL is active before executing code
    - Use try/except blocks for robust error handling

    ## Standard Project Template

    ```python
    # Import necessary libraries
    import os
    import datetime
    from pathlib import Path

    # Step 1: Setup and Configuration
    def setup_environment():
        "Set up the environment including directories and REPL"
        # Verify paths
        base_dir = "/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl"

        # Ask user to confirm directory
        user_confirm = input(f"Is this the correct base directory? {base_dir} (y/n): ")
        if user_confirm.lower() != "y":
            base_dir = input("Please enter the correct base directory path: ")

        # Create output directories if they don't exist
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Create a new REPL
        repl_id = create_python_repl()
        print(f"Created new REPL with ID: {repl_id}")

        return base_dir, repl_id

    # Step 2: Install required packages (if needed)
    def install_packages(repl_id, packages):
        "Install required packages in the virtual environment"
        packages_str = " ".join(packages)
        install_cmd = f". .venv/bin/activate && uv pip install {packages_str}"

        print(f"Installing packages: {packages_str}")
        result = execute_command(command=install_cmd)
        print(f"Installation result: {result}")

    # Step 3: Run your analysis
    def run_analysis(repl_id, base_dir):
        "Run the main analysis code"
        # Set up paths
        output_dir = os.path.join(base_dir, "output")

        # Run your analysis code
        analysis_code = "
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        # Your analysis code here
        print("Analysis complete!")
        "

        result = run_python_in_repl(code=analysis_code, repl_id=repl_id)
        print(result)

    # Step 4: Save results
    def save_results(repl_id, base_dir):
        "Save results to the appropriate directories"
        # Your code to save results here
        pass

    # Main execution
    if __name__ == "__main__":
        # Setup
        base_dir, repl_id = setup_environment()

        try:
            # Install packages if needed
            # install_packages(repl_id, ["pandas", "matplotlib", "numpy"])

            # Run analysis
            run_analysis(repl_id, base_dir)

            # Save results
            save_results(repl_id, base_dir)
        finally:
            # Clean up
            print(f"Deleting REPL: {repl_id}")
            delete_repl(repl_id)
    ```
    """
