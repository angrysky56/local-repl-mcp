"""
Modular Empowerment Framework (MEF) Integration Example

This script demonstrates how to integrate the Modular Empowerment Framework
with the Python REPL environment for enhanced capabilities.

The MEF provides advanced tools for:
- Agent-based automation
- Self-improvement mechanisms
- Complex problem solving
- Memory and reflection capabilities

Prerequisites:
- The MEF must be installed in your environment
- You need to know the path to the MEF directory
"""

def empowered_agent_basic_setup() -> str:
    """
    A prompt template for Modular Empowerment Framework integration.
    """
    return """
    # Modular Empowerment Framework Integration

    This guide demonstrates how to integrate the Modular Empowerment Framework (MEF)
    with the Python REPL environment for enhanced capabilities.

    ## Step 1: Create a new REPL instance

    ```python
    # Create a new Python REPL
    repl_id = create_python_repl()
    print(f"REPL created with ID: {repl_id}")
    ```

    ## Step 2: Set up the Modular Empowerment Framework

    ```python
    # Set up the framework (modify the path if needed)
    setup_result = setup_modular_empowerment(path="/home/ty/Repositories/ai_workspace/local-repl-mcp/modular_empowerment_framework")
    print(setup_result)
    ```

    ## Step 3: Initialize the MEF in the REPL

    ```python
    # Initialize the MEF in this REPL
    init_result = initialize_modular_empowerment(repl_id=repl_id)
    print(init_result)
    ```

    ## Step 4: Test the MEF capabilities

    ```python
    # Test MEF functionality
    test_code = \"\"\"
    # Import the MEF core modules (these will be available after initialization)
    from src.agent import Agent
    from src.task import Task

    # Create a simple agent
    agent = Agent(name="TestAgent")
    print(f"Created agent: {agent.name}")

    # Verify the MEF is working
    print("MEF is successfully initialized and ready to use!")
    \"\"\"

    run_python_in_repl(code=test_code, repl_id=repl_id)
    ```

    ## Step 5: Run a sample workflow

    ```python
    # Example workflow using MEF capabilities
    workflow_code = \"\"\"
    from src.agent import Agent
    from src.task import Task
    from src.analyzer import DataAnalyzer

    # Create an analysis agent
    analyzer = Agent(name="DataAnalyzer", capabilities=["data_analysis", "visualization"])

    # Create a sample task
    analysis_task = Task(
        name="AnalyzeDataset",
        description="Analyze and visualize a sample dataset",
        priority=0.8
    )

    # Simulate assigning the task to the agent
    print(f"Assigned task '{analysis_task.name}' to agent '{analyzer.name}'")

    # Create a sample dataset
    import numpy as np
    import pandas as pd

    # Generate random data
    np.random.seed(42)
    data = pd.DataFrame({
        'x': np.random.normal(0, 1, 100),
        'y': np.random.normal(5, 2, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })

    print(f"Created sample dataset with {len(data)} records")
    print(f"Features: {data.columns.tolist()}")

    # Display basic statistics
    print("\\nBasic statistics:")
    print(data.describe())
    \"\"\"

    run_python_in_repl(code=workflow_code, repl_id=repl_id)
    ```

    ## Step 6: Clean up

    ```python
    # Clean up when finished
    delete_repl(repl_id)
    print("MEF integration example completed successfully!")
    ```

    Note: If the MEF is not properly installed or the path is incorrect, the integration
    steps will guide you through the troubleshooting process.
    """
