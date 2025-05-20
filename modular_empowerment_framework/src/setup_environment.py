#!/usr/bin/env python3
"""
REPL Environment Setup Script

This script initializes the necessary directory structure for organizing
REPL outputs, agents, and scripts. It also verifies the environment paths
and ensures everything is properly configured before starting.
"""

import os
import sys

# Define the required directories
REQUIRED_DIRS = [
    "output",
    "output/data",
    "output/results",
    "output/visualizations",
    "repl_agent_library",
    "repl_script_library"
]

def setup_environment(base_path=None):
    """
    Set up the REPL environment directory structure.

    Args:
        base_path (str, optional): The base path to use. If None, will prompt user.

    Returns:
        bool: True if setup was successful, False otherwise
    """
    # Get the base path
    if base_path is None:
        # Default path
        default_path = "/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl"

        # Ask user to confirm the path
        print(f"Is the following local REPL directory correct?\n{default_path}")
        response = input("Enter 'y' to confirm, or provide the correct path: ").strip()

        if response.lower() == 'y':
            base_path = default_path
        else:
            base_path = response

    # Verify the path exists
    if not os.path.exists(base_path):
        print(f"Error: The path {base_path} does not exist.")
        return False

    # Create the required directories
    for directory in REQUIRED_DIRS:
        dir_path = os.path.join(base_path, directory)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
            except Exception as e:
                print(f"Error creating directory {dir_path}: {e}")
                return False
        else:
            print(f"Directory already exists: {dir_path}")

    print("\nEnvironment setup complete! Directory structure:")
    for directory in REQUIRED_DIRS:
        print(f"- {os.path.join(base_path, directory)}")

    print("\nYou can now use these directories for:")
    print("- output: Save all generated files, visualizations, and results")
    print("- repl_agent_library: Store agent scripts for automation")
    print("- repl_script_library: Store reusable Python scripts")

    return True

def check_venv_activation():
    """Check if virtual environment is activated and provide guidance."""
    if 'VIRTUAL_ENV' in os.environ:
        print("Virtual environment is activated. You can install packages with:")
        print("uv .venv/bin/activate && uv pip install <package-name>")
    else:
        print("\nVirtual environment is NOT activated. To activate, run:")
        print(". .venv/bin/activate")
        print("\nAfter activation, you can install packages with:")
        print("uv pip install <package-name>")

if __name__ == "__main__":
    print("=" * 80)
    print("REPL Environment Setup")
    print("=" * 80)

    success = setup_environment()

    if success:
        check_venv_activation()
        print("\nSetup completed successfully!")
    else:
        print("\nSetup failed. Please check the errors and try again.")
        sys.exit(1)
