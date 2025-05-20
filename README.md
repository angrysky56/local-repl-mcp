# Local REPL MCP Server

A local Python REPL (Read-Eval-Print Loop) server for Claude Desktop, implemented using the Model Context Protocol (MCP).

## Overview

This MCP server provides a powerful Python environment that allows you to:

- Run Python code directly within Claude
- Maintain persistent state throughout your conversation
- Create and manage multiple independent Python environments
- Execute data analysis, visualizations, and computational tasks

## Features

- **Multiple REPL Instances**: Create and manage separate Python environments
- **State Persistence**: Variables and imports persist between code executions
- **Comprehensive Prompt Templates**: Ready-to-use workflows for common tasks
- **Enhanced Error Handling**: Clear error messages with traceback information
- **Modular Empowerment Framework**: Optional integration with advanced agent capabilities

## Installation

### Prerequisites

- Claude Desktop App (latest version)
- Python 3.10+ with uv package manager

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/angrysky56/local-repl-mcp.git
   cd local-repl-mcp
   ```

2. Create a virtual environment using uv:
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   uv add mcp
   ```

4. Create a copy of the example MCP configuration file:
   ```bash
   cp example_mcp_config.json ~/.config/claude-app/mcp_config.json
   ```

5. Edit the config file to point to your local repository path:
   ```json
   {
     "mcpServers": {
       "LocalREPL": {
         "command": "uv",
         "args": [
           "--directory",
           "/path/to/local-repl-mcp/local_repl",
           "run",
           "server.py"
         ]
       }
     }
   }
   ```

   Note: Replace "/path/to/local-repl-mcp" with your actual repository path.

6. Verify the server is working:
   ```bash
   cd /path/to/local-repl-mcp
   . .venv/bin/activate
   python -m local_repl.server
   ```

   You should see the server start without errors, and the loaded prompts will be displayed.

## Usage

Once installed, you can use the server directly in Claude Desktop:

1. Start Claude Desktop
2. Type `/LocalREPL` to access available prompts
3. Select a prompt to begin working with the REPL

### Basic REPL Commands

```python
# Create a new REPL
repl_id = create_python_repl()

# Run Python code in the REPL
run_python_in_repl(
  code="x = 42\nprint(f'The answer is {x}')",
  repl_id=repl_id
)

# Run more code in the same REPL (state is preserved)
run_python_in_repl(
  code="import math\nprint(f'The square root of {x} is {math.sqrt(x)}')",
  repl_id=repl_id
)

# List all active REPLs
list_active_repls()

# Get information about a specific REPL
get_repl_info(repl_id)

# Delete a REPL when finished
delete_repl(repl_id)
```

## Available Prompts

The server includes several prompt templates for common workflows:

- **Python REPL**: Basic Python REPL usage
- **Data Analysis**: Complete data analysis workflow
- **Test Prompt**: Simple test for verifying prompt functionality
- **Modular Empowerment**: Advanced agent-based system with persistence
- **REPL Integration Example**: Example of integrating with external systems
- **MEF Integration Example**: Modular Empowerment Framework integration

## Folder Structure

```
local-repl-mcp/
├── example_mcp_config.json   # Example configuration for Claude Desktop
├── local_repl/               # MCP server implementation
│   ├── prompts/              # Prompt templates
│   │   ├── __init__.py       # Prompt loader
│   │   ├── python_repl.py    # Basic REPL prompt
│   │   ├── data_analysis.py  # Data analysis prompt
│   │   └── ...               # Other prompt templates
│   ├── server.py             # Main MCP server implementation
│   └── ...                   # Other server components
├── modular_empowerment_framework/ # Optional MEF integration
└── README.md                 # This file
```

## Troubleshooting

If you encounter issues with the server:

1. Check that the path in your `mcp_config.json` is correct
2. Ensure the server's virtual environment is properly activated
3. Check Claude Desktop logs for any error messages
4. Restart Claude Desktop after making configuration changes

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
