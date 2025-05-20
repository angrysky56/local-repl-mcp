"""
Data Analysis workflow prompt template.

This module provides a template for data analysis workflow using the Python REPL.
"""

def data_analysis_workflow() -> str:
    """
    A prompt template for data analysis workflow.
    """
    return """
    # Data Analysis Workflow with Python REPL

    This workflow guides you through a structured data analysis process using the Python REPL.
    Follow these steps to explore, analyze, and visualize data effectively.

    ## Step 1: Setup Your Environment

    First, create a new REPL and initialize it with essential data science libraries:

    ```python
    # Create a new REPL
    repl_id = create_python_repl()

    # Initialize with data science libraries
    init_code = \"\"\"
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime

    # Configure visualization settings
    plt.style.use('ggplot')
    sns.set(style="whitegrid")

    # Create output directory if it doesn't exist
    import os
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Helper function to save figures
    def save_figure(fig, name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{name}_{timestamp}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {filename}")
        return filename
    \"\"\"

    run_python_in_repl(code=init_code, repl_id=repl_id)
    ```

    ## Step 2: Load and Inspect Data

    Load your data and perform initial inspection:

    ```python
    data_loading_code = \"\"\"
    # Load data (replace with your actual data source)
    # Example with CSV:
    # df = pd.read_csv('data.csv')
    
    # For demonstration, let's create sample data
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'is_valid': np.random.choice([True, False], 100, p=[0.9, 0.1])
    })
    
    # Print basic information
    print("Dataset shape:", df.shape)
    print("\\nDataset info:")
    df.info()
    
    print("\\nSummary statistics:")
    print(df.describe())
    
    print("\\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    print("\\nMissing values by column:")
    print(df.isna().sum())
    \"\"\"

    run_python_in_repl(code=data_loading_code, repl_id=repl_id)
    ```

    ## Step 3: Data Cleaning and Preprocessing

    Clean and preprocess your data:

    ```python
    preprocessing_code = \"\"\"
    # Handle missing values
    df_clean = df.copy()
    
    # Fill numeric missing values with mean
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # Fill categorical missing values with mode
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # Create new features
    df_clean['year_month'] = df_clean['date'].dt.strftime('%Y-%m')
    df_clean['is_weekend'] = df_clean['date'].dt.dayofweek >= 5
    
    # Standardize numerical features if needed
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_clean['value_scaled'] = scaler.fit_transform(df_clean[['value']])
    
    print("Preprocessed data sample:")
    print(df_clean.head())
    \"\"\"

    # You would need to install scikit-learn for the above code
    install_cmd = ". .venv/bin/activate && uv pip install scikit-learn"
    run_python_in_repl(code=f"import subprocess; subprocess.run('{install_cmd}', shell=True)", repl_id=repl_id)
    run_python_in_repl(code=preprocessing_code, repl_id=repl_id)
    ```

    ## Step 4: Exploratory Data Analysis

    Explore patterns and relationships in your data:

    ```python
    eda_code = \"\"\"
    # 1. Temporal analysis
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_clean, x='date', y='value', hue='category')
    plt.title('Values Over Time by Category')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    time_series_fig = plt.gcf()
    save_figure(time_series_fig, 'time_series_by_category')

    # 2. Distribution analysis
    plt.figure(figsize=(12, 6))
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data=df_clean, x='value', kde=True)
    plt.title('Value Distribution')
    # Box plot by category
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_clean, x='category', y='value')
    plt.title('Value Distribution by Category')
    plt.tight_layout()
    distribution_fig = plt.gcf()
    save_figure(distribution_fig, 'value_distributions')

    # 3. Correlation analysis
    numeric_df = df_clean.select_dtypes(include=['number'])
    plt.figure(figsize=(10, 8))
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    corr_fig = plt.gcf()
    save_figure(corr_fig, 'correlation_matrix')

    # 4. Feature relationships
    plt.figure(figsize=(12, 10))
    sns.pairplot(df_clean, hue='category', vars=['value', 'value_scaled', 'is_weekend'])
    pair_fig = plt.gcf()
    save_figure(pair_fig, 'feature_relationships')

    # 5. Time patterns
    # Aggregate by year-month
    monthly_data = df_clean.groupby('year_month').agg({
        'value': ['mean', 'std', 'min', 'max'],
        'is_valid': 'sum'
    })
    monthly_data.columns = ['_'.join(col).strip() for col in monthly_data.columns.values]
    monthly_data = monthly_data.reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_data, x='year_month', y='value_mean')
    plt.fill_between(
        monthly_data['year_month'],
        monthly_data['value_mean'] - monthly_data['value_std'],
        monthly_data['value_mean'] + monthly_data['value_std'],
        alpha=0.2
    )
    plt.title('Monthly Average Values with Standard Deviation')
    plt.xlabel('Month')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    monthly_fig = plt.gcf()
    save_figure(monthly_fig, 'monthly_trends')

    # Print summary statistics by category
    category_stats = df_clean.groupby('category').agg({
        'value': ['count', 'mean', 'std', 'min', 'max'],
        'is_valid': 'sum'
    })
    print("\\nSummary by Category:")
    print(category_stats)
    \"\"\"

    run_python_in_repl(code=eda_code, repl_id=repl_id)
    ```

    ## Step 5: Statistical Analysis

    Perform statistical tests and modeling:

    ```python
    statistical_code = \"\"\"
    # 1. ANOVA test to compare categories
    from scipy import stats

    anova_results = stats.f_oneway(
        *[df_clean[df_clean['category'] == cat]['value'] for cat in df_clean['category'].unique()]
    )
    print("ANOVA test (comparing values across categories):")
    print(f"F-statistic: {anova_results.statistic:.4f}")
    print(f"p-value: {anova_results.pvalue:.4f}")
    print(f"Significant difference: {anova_results.pvalue < 0.05}")

    # 2. Chi-square test for categorical variables
    contingency_table = pd.crosstab(df_clean['category'], df_clean['is_weekend'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print("\\nChi-square test (category vs weekend):")
    print(f"Chi2 value: {chi2:.4f}")
    print(f"p-value: {p:.4f}")
    print(f"Significant relationship: {p < 0.05}")

    # 3. Simple linear regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    
    # Prepare features (using dummy variables for categorical features)
    X = pd.get_dummies(df_clean[['value_scaled', 'is_weekend', 'category']], drop_first=True)
    y = df_clean['value']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print("\\nLinear Regression Results:")
    print(f"R² (train): {train_score:.4f}")
    print(f"R² (test): {test_score:.4f}")
    
    # Show coefficients
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    coefficients = coefficients.sort_values('Coefficient', ascending=False)
    print("\\nFeature Coefficients:")
    print(coefficients)
    
    # 4. Time series decomposition for trend and seasonality
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Prepare time series data
    ts_data = df_clean.set_index('date')['value']
    
    # Decompose time series
    decomposition = seasonal_decompose(ts_data, model='additive', period=7)
    
    # Plot decomposition
    plt.figure(figsize=(12, 10))
    plt.subplot(411)
    plt.plot(ts_data, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    ts_fig = plt.gcf()
    save_figure(ts_fig, 'time_series_decomposition')
    \"\"\"

    # Install statistical packages
    install_cmd = ". .venv/bin/activate && uv pip install scipy statsmodels"
    run_python_in_repl(code=f"import subprocess; subprocess.run('{install_cmd}', shell=True)", repl_id=repl_id)
    run_python_in_repl(code=statistical_code, repl_id=repl_id)
    ```

    ## Step 6: Advanced Visualization with Interactive Components

    Create interactive visualizations for better data exploration:

    ```python
    interactive_viz_code = \"\"\"
    # Save data for artifact visualization
    df_clean.to_csv('./output/analysis_data.csv', index=False)
    
    # Create a summary for display
    summary_data = {
        'dataset_shape': df_clean.shape,
        'categories': df_clean['category'].unique().tolist(),
        'date_range': [df_clean['date'].min().strftime('%Y-%m-%d'), 
                       df_clean['date'].max().strftime('%Y-%m-%d')],
        'value_stats': {
            'min': float(df_clean['value'].min()),
            'max': float(df_clean['value'].max()),
            'mean': float(df_clean['value'].mean()),
            'median': float(df_clean['value'].median())
        },
        'category_counts': df_clean['category'].value_counts().to_dict()
    }
    
    import json
    with open('./output/analysis_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print("Data and summary saved for React visualization")
    print("Summary:", summary_data)
    \"\"\"

    run_python_in_repl(code=interactive_viz_code, repl_id=repl_id)
    ```

    ## Step 7: Reporting and Documentation

    Generate a summary report of your findings:

    ```python
    report_code = \"\"\"
    # Create a summary report
    report = f\"\"\"
    # Data Analysis Report
    
    ## Dataset Overview
    - **Rows:** {df_clean.shape[0]}
    - **Columns:** {df_clean.shape[1]}
    - **Date Range:** {df_clean['date'].min().strftime('%Y-%m-%d')} to {df_clean['date'].max().strftime('%Y-%m-%d')}
    - **Categories:** {', '.join(df_clean['category'].unique())}
    
    ## Key Statistics
    - **Mean Value:** {df_clean['value'].mean():.2f}
    - **Median Value:** {df_clean['value'].median():.2f}
    - **Standard Deviation:** {df_clean['value'].std():.2f}
    - **Min Value:** {df_clean['value'].min():.2f}
    - **Max Value:** {df_clean['value'].max():.2f}
    
    ## Category Distribution
    {df_clean['category'].value_counts().to_frame().to_markdown()}
    
    ## Key Findings
    1. The data shows a {"significant" if anova_results.pvalue < 0.05 else "non-significant"} difference between categories (ANOVA p-value: {anova_results.pvalue:.4f})
    2. Category and weekend status have a {"significant" if p < 0.05 else "non-significant"} relationship (Chi-square p-value: {p:.4f})
    3. The linear regression model explains {test_score:.2%} of the variance in the test data
    4. {"Category " + str(coefficients.iloc[0]['Feature'].split('_')[-1]) if 'category' in coefficients.iloc[0]['Feature'] else coefficients.iloc[0]['Feature']} has the strongest positive effect on values
    
    ## Visualizations
    Several visualizations were generated during this analysis:
    - Time series by category
    - Value distributions
    - Correlation matrix
    - Feature relationships
    - Monthly trends
    - Time series decomposition
    
    ## Recommendations
    Based on the analysis:
    1. Focus on {"category " + str(coefficients.iloc[0]['Feature'].split('_')[-1]) if 'category' in coefficients.iloc[0]['Feature'] else coefficients.iloc[0]['Feature']} for maximum value
    2. Consider the seasonal patterns when planning future actions
    3. Monitor the trend component for long-term changes
    \"\"\"
    
    # Save the report to a file
    report_filename = './output/data_analysis_report.md'
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {report_filename}")
    \"\"\"

    run_python_in_repl(code=report_code, repl_id=repl_id)
    ```

    ## Step 8: Clean Up

    Clean up resources when you're done:

    ```python
    # Clean up the REPL when done
    delete_repl(repl_id)
    print("Analysis workflow completed!")
    ```

    ## Custom Modifications

    You can customize this workflow:

    1. **Data Sources**: Replace the sample data with your own data sources
    2. **Visualizations**: Add or modify visualizations based on your specific needs
    3. **Statistical Tests**: Add more sophisticated statistical tests relevant to your domain
    4. **Machine Learning**: Extend with more advanced ML models if needed
    5. **Reporting**: Customize the report format and content

    Remember that all outputs are saved in the `./output` directory for future reference.
    """
