"""
A sample workflow prompt for data analysis.

This module provides a template for a basic data analysis workflow.
"""

def data_analysis_workflow() -> str:
    """
    A prompt template for basic data analysis with Python.
    """
    return """
    # Data Analysis Workflow
    
    This workflow helps you analyze data using Python. It provides a step-by-step 
    guide on how to load, clean, analyze, and visualize your data.
    
    ## Available commands:
    
    - create_python_repl() - Creates a new Python REPL
    - run_python_in_repl(code, repl_id) - Runs Python code in the REPL
    
    ## Step 1: Setup your environment and load data
    
    ```python
    # Create a new REPL
    repl_id = create_python_repl()
    print(f"Created REPL: {repl_id}")
    
    # Import basic libraries
    run_python_in_repl(
        code='''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Print versions to confirm everything is loaded
    print(f"pandas version: {pd.__version__}")
    print(f"numpy version: {np.__version__}")
    ''',
        repl_id=repl_id
    )
    
    # Load some data
    # Use window.fs.readFile to load data uploaded by the user
    run_python_in_repl(
        code='''
    # Example for loading a CSV file
    async def load_csv(file_path):
        try:
            data = await window.fs.readFile(file_path, {'encoding': 'utf8'})
            return pd.read_csv(pd.StringIO(data))
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
            
    # To load a file, use the function like this:
    # df = await load_csv('your_file.csv')
    ''',
        repl_id=repl_id
    )
    ```
    
    ## Step 2: Data exploration and cleaning
    
    ```python
    # Explore the data
    run_python_in_repl(
        code='''
    # For demonstration, let's create some sample data
    df = pd.DataFrame({
        'A': np.random.rand(10),
        'B': np.random.rand(10),
        'C': np.random.choice(['X', 'Y', 'Z'], 10)
    })
    
    # Basic exploration
    print("Data shape:", df.shape)
    print("\\nData types:")
    print(df.dtypes)
    print("\\nFirst 5 rows:")
    print(df.head())
    print("\\nSummary statistics:")
    print(df.describe())
    ''',
        repl_id=repl_id
    )
    
    # Clean the data
    run_python_in_repl(
        code='''
    # Check for missing values
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Handle missing values (if any)
    df_clean = df.fillna(df.mean())
    
    # Check for outliers
    print("\\nPotential outliers (values outside 3 standard deviations):")
    for col in df.select_dtypes(include=[np.number]).columns:
        mean = df[col].mean()
        std = df[col].std()
        outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
        if not outliers.empty:
            print(f"Column {col} has {len(outliers)} outliers")
    ''',
        repl_id=repl_id
    )
    ```
    
    ## Step 3: Data analysis and visualization
    
    ```python
    # Basic analysis
    run_python_in_repl(
        code='''
    # Group by categorical variables
    print("Group statistics:")
    print(df.groupby('C').mean())
    
    # Correlation analysis
    print("\\nCorrelation matrix:")
    print(df.corr())
    ''',
        repl_id=repl_id
    )
    
    # Visualization
    run_python_in_repl(
        code='''
    # Create a simple plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['A'], df['B'], c=['red' if x == 'X' else 'blue' if x == 'Y' else 'green' for x in df['C']])
    plt.xlabel('A')
    plt.ylabel('B')
    plt.title('Scatter plot of A vs B colored by C')
    plt.grid(True)
    plt.show()
    ''',
        repl_id=repl_id
    )
    ```
    
    ## Step 4: Statistical analysis
    
    ```python
    # Statistical tests
    run_python_in_repl(
        code='''
    from scipy import stats
    
    # t-test example
    x = df[df['C'] == 'X']['A']
    y = df[df['C'] == 'Y']['A']
    
    if len(x) > 0 and len(y) > 0:
        t_stat, p_val = stats.ttest_ind(x, y)
        print(f"t-test between groups X and Y for variable A:")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
        print(f"Statistically significant difference: {p_val < 0.05}")
    ''',
        repl_id=repl_id
    )
    ```
    
    ## Step 5: Save and export results
    
    ```python
    # Save results
    run_python_in_repl(
        code='''
    # Example of saving results to a string (can be used for download)
    output_csv = df.to_csv(index=False)
    print(f"CSV output (first 200 chars):\\n{output_csv[:200]}...")
    
    # Example of generating a summary report
    report = f\"\"\"
    Data Analysis Report
    -------------------
    Dataset dimensions: {df.shape[0]} rows x {df.shape[1]} columns
    
    Variables:
    {df.dtypes.to_string()}
    
    Summary statistics:
    {df.describe().to_string()}
    
    Group statistics:
    {df.groupby('C').mean().to_string()}
    \"\"\"
    
    print("\\nSummary Report:\\n")
    print(report)
    ''',
        repl_id=repl_id
    )
    ```
    
    ## Cleaning up
    
    ```python
    # Delete the REPL when done
    delete_repl(repl_id)
    print(f"Deleted REPL: {repl_id}")
    ```
    
    This workflow provides a basic template for data analysis. You can modify it
    based on your specific requirements and datasets.
    """
