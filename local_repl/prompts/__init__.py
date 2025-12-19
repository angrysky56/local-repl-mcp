"""
Prompts package for the MCP server.

This package contains all the prompt templates that can be used by the MCP server.
Prompts are dynamically loaded from the files in this directory.
"""

import importlib
import inspect
import os
import sys
import logging
from typing import Dict, Callable, Any

# Dictionary to store all loaded prompts
prompts: Dict[str, Callable[[], str]] = {}

def load_prompts():
    """
    Dynamically load all prompt functions from files in the prompts directory.

    A prompt function is any function that:
    1. Is defined in a file in the prompts directory
    2. Returns a string (the prompt template)

    This includes both functions ending with _workflow and other prompt functions.
    """
    # Get the directory where this file is located
    prompt_dir = os.path.dirname(os.path.abspath(__file__))

    # Get all Python files in the directory
    prompt_files = [f[:-3] for f in os.listdir(prompt_dir)
                   if f.endswith('.py') and f != '__init__.py']

    # Import each file as a module
    for file_name in prompt_files:
        try:
            module_path = f"prompts.{file_name}"
            module = importlib.import_module(module_path)

            # Get all functions that return strings
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj):
                    # Check if the function has a return annotation of str
                    # or if it ends with _workflow (for backward compatibility)
                    has_str_return = (hasattr(obj, "__annotations__") and
                                     obj.__annotations__.get("return") == str)

                    is_workflow = name.endswith('_workflow')

                    if has_str_return or is_workflow:
                        # Add the function to our prompts dictionary
                        prompts[name] = obj
                        print(f"Loaded prompt: {name} from {file_name}.py", file=sys.stderr)
        except Exception as e:
            print(f"Error loading prompts from {file_name}.py: {e}", file=sys.stderr)
            logging.exception(f"Error loading prompts from {file_name}.py")

    return prompts

# Load prompts when the package is imported
load_prompts()
