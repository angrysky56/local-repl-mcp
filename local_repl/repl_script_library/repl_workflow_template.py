"""
REPL Workflow Example

This script demonstrates a complete workflow using the Python REPL environment.
It shows how to:
1. Set up a REPL instance
2. Install and import dependencies
3. Load and process data
4. Create visualizations
5. Save output to the correct folders
6. Clean up resources

Usage:
1. Review this template
2. Modify for your specific analysis needs
3. Run the relevant sections in sequence
"""

def main():
    # Step 1: Confirm local REPL directory
    print("Checking REPL environment...")
    # Replace with your base directory or use the default
    base_dir = "/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl"
    
    response = input(f"Is this REPL directory correct? {base_dir} (y/n): ")
    if response.lower() != 'y':
        base_dir = input("Please enter the correct path: ")
        print(f"Using directory: {base_dir}")
    
    # Step 2: Create output directories if they don't exist
    import os
    output_dir = os.path.join(base_dir, "output")
    agent_lib_dir = os.path.join(base_dir, "repl_agent_library")
    script_lib_dir = os.path.join(base_dir, "repl_script_library")
    
    for directory in [output_dir, agent_lib_dir, script_lib_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Step 3: Create a new REPL instance
    print("Creating a new Python REPL...")
    # Uncomment when running in Claude
    # repl_id = create_python_repl()
    # print(f"REPL created with ID: {repl_id}")
    
    # Step 4: Install required packages (if needed)
    print("Installing required packages...")
    # Uncomment when running in Claude
    # execute_command(command=". .venv/bin/activate && uv pip install pandas matplotlib seaborn")
    
    # Step 5: Run initialization code
    print("Initializing libraries...")
    init_code = """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # Configure matplotlib for better visuals
    plt.style.use('ggplot')
    
    # Print versions to confirm everything loaded correctly
    print(f"pandas version: {pd.__version__}")
    print(f"numpy version: {np.__version__}")
    print(f"matplotlib version: {plt.__version__}")
    
    # Define the output directory path
    output_dir = "{output_dir}"
    """
    
    # Uncomment when running in Claude
    # run_python_in_repl(code=init_code.format(output_dir=output_dir), repl_id=repl_id)
    
    # Step 6: Load and analyze data
    print("Loading and analyzing data...")
    analysis_code = """
    # Create some sample data
    data = pd.DataFrame({
        'x': np.random.normal(0, 1, 1000),
        'y': np.random.normal(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Display basic statistics
    print("Data summary:")
    print(data.describe())
    
    # Perform some analysis
    group_stats = data.groupby('category').agg({
        'x': ['mean', 'std'],
        'y': ['mean', 'std']
    })
    
    print("\\nGroup statistics:")
    print(group_stats)
    """
    
    # Uncomment when running in Claude
    # run_python_in_repl(code=analysis_code, repl_id=repl_id)
    
    # Step 7: Create visualizations
    print("Creating visualizations...")
    visualization_code = """
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    
    for category, group in data.groupby('category'):
        plt.scatter(group['x'], group['y'], label=category, alpha=0.7)
    
    plt.title('Scatter Plot by Category')
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.legend()
    plt.grid(True)
    
    # Save the figure to the output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"scatter_plot_{timestamp}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    """
    
    # Uncomment when running in Claude
    # run_python_in_repl(code=visualization_code, repl_id=repl_id)
    
    # Step 8: Clean up resources
    print("Cleaning up resources...")
    # Uncomment when running in Claude
    # delete_repl(repl_id)
    # print(f"Deleted REPL with ID: {repl_id}")
    
    print("Workflow completed successfully!")

if __name__ == "__main__":
    main()
