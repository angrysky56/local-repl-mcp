"""
REPL Data Analysis Agent

This script defines a reusable data analysis agent that can be used with the REPL.
It provides an interface for loading, processing, and visualizing data with
built-in error handling and progress reporting.

Usage:
1. Import this agent in your REPL scripts
2. Initialize with your specific parameters
3. Use the provided methods for data operations

Example:
```python
from repl_agent_library.data_analysis_agent import DataAnalysisAgent

# Initialize the agent
agent = DataAnalysisAgent(output_dir="/path/to/output")

# Use the agent
agent.load_data("my_data.csv")
agent.analyze()
agent.visualize("histogram")
agent.save_results()
```
"""

import os
import traceback
from datetime import datetime


class DataAnalysisAgent:
    """A reusable agent for data analysis tasks in the REPL environment."""
    
    def __init__(self, output_dir=None, name="DataAnalysisAgent"):
        """
        Initialize the data analysis agent.
        
        Args:
            output_dir (str): Directory to save outputs. If None, uses default.
            name (str): Name of the agent instance
        """
        self.name = name
        
        # Set default output directory if not provided
        if output_dir is None:
            base_dir = "/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl"
            self.output_dir = os.path.join(base_dir, "output")
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.data = None
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"{self.name} initialized. Output directory: {self.output_dir}")
    
    def load_data(self, data_source, **kwargs):
        """
        Load data from a source (file, URL, etc.).
        
        Args:
            data_source (str): Path to data file or URL
            **kwargs: Additional arguments for data loading
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import pandas as pd
            
            print(f"Loading data from {data_source}...")
            
            # Handle different file types
            if data_source.endswith('.csv'):
                self.data = pd.read_csv(data_source, **kwargs)
            elif data_source.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(data_source, **kwargs)
            elif data_source.endswith('.json'):
                self.data = pd.read_json(data_source, **kwargs)
            else:
                # Handle URLs or other sources
                if data_source.startswith(('http://', 'https://')):
                    self.data = pd.read_csv(data_source, **kwargs)
                else:
                    raise ValueError(f"Unsupported data source format: {data_source}")
            
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return False
    
    def analyze(self, methods=None):
        """
        Perform data analysis with specified methods.
        
        Args:
            methods (list): List of analysis methods to apply
            
        Returns:
            dict: Analysis results
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return {}
        
        try:
            print("Analyzing data...")
            
            # Default analysis if no methods provided
            if methods is None:
                methods = ['summary', 'correlation', 'missing']
            
            results = {}
            
            # Perform requested analyses
            if 'summary' in methods:
                results['summary'] = self.data.describe()
                print("Summary statistics calculated.")
            
            if 'correlation' in methods:
                numeric_data = self.data.select_dtypes(include=['number'])
                if not numeric_data.empty:
                    results['correlation'] = numeric_data.corr()
                    print("Correlation analysis completed.")
                else:
                    print("No numeric columns for correlation analysis.")
            
            if 'missing' in methods:
                results['missing'] = self.data.isnull().sum()
                print("Missing value analysis completed.")
            
            # Store results
            self.results.update(results)
            return results
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            traceback.print_exc()
            return {}
    
    def visualize(self, plot_type, **kwargs):
        """
        Create visualizations of the data.
        
        Args:
            plot_type (str): Type of plot to generate
            **kwargs: Additional arguments for plotting
            
        Returns:
            str: Path to saved visualization or None
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print(f"Creating {plot_type} visualization...")
            
            # Configure plot
            plt.figure(figsize=kwargs.get('figsize', (10, 6)))
            
            # Create visualization based on type
            if plot_type == 'histogram':
                column = kwargs.get('column', self.data.columns[0])
                self.data[column].hist()
                plt.title(f'Histogram of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                
            elif plot_type == 'scatter':
                x_col = kwargs.get('x', self.data.columns[0])
                y_col = kwargs.get('y', self.data.columns[1])
                plt.scatter(self.data[x_col], self.data[y_col])
                plt.title(f'Scatter Plot: {x_col} vs {y_col}')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                
            elif plot_type == 'boxplot':
                column = kwargs.get('column', self.data.select_dtypes(include=['number']).columns[0])
                self.data.boxplot(column=column)
                plt.title(f'Boxplot of {column}')
                
            elif plot_type == 'heatmap':
                numeric_data = self.data.select_dtypes(include=['number'])
                if not numeric_data.empty:
                    sns.heatmap(numeric_data.corr(), annot=kwargs.get('annot', True))
                    plt.title('Correlation Heatmap')
                else:
                    print("No numeric columns for heatmap.")
                    return None
                
            else:
                print(f"Unsupported plot type: {plot_type}")
                return None
            
            # Save the visualization
            filename = f"{self.name}_{plot_type}_{self.timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            traceback.print_exc()
            return None
    
    def save_results(self, format='csv'):
        """
        Save analysis results to files.
        
        Args:
            format (str): Format to save results in (csv, json, etc.)
            
        Returns:
            list: Paths to saved result files
        """
        if not self.results:
            print("No results to save. Please run analysis first.")
            return []
        
        saved_files = []
        
        try:
            print(f"Saving results in {format} format...")
            
            for result_name, result_data in self.results.items():
                filename = f"{self.name}_{result_name}_{self.timestamp}.{format}"
                filepath = os.path.join(self.output_dir, filename)
                
                if format == 'csv':
                    result_data.to_csv(filepath)
                elif format == 'json':
                    result_data.to_json(filepath)
                elif format == 'excel':
                    result_data.to_excel(filepath)
                else:
                    print(f"Unsupported format: {format}")
                    continue
                
                saved_files.append(filepath)
                print(f"Saved {result_name} results to: {filepath}")
            
            return saved_files
            
        except Exception as e:
            print(f"Error saving results: {e}")
            traceback.print_exc()
            return saved_files
    
    def generate_report(self):
        """
        Generate a comprehensive report of the analysis.
        
        Returns:
            str: Path to the generated report
        """
        if not self.results:
            print("No results for report. Please run analysis first.")
            return None
        
        try:
            print("Generating analysis report...")
            
            # Create report content
            report_content = [
                f"# Data Analysis Report\n",
                f"Generated by {self.name} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                f"## Dataset Overview\n"
            ]
            
            if self.data is not None:
                report_content.extend([
                    f"* Number of records: {len(self.data)}\n",
                    f"* Number of features: {len(self.data.columns)}\n",
                    f"* Features: {', '.join(self.data.columns)}\n\n"
                ])
            
            # Add results sections
            if 'summary' in self.results:
                report_content.extend([
                    f"## Summary Statistics\n\n",
                    f"```\n{self.results['summary']}\n```\n\n"
                ])
            
            if 'correlation' in self.results:
                report_content.extend([
                    f"## Correlation Analysis\n\n",
                    f"```\n{self.results['correlation']}\n```\n\n"
                ])
            
            if 'missing' in self.results:
                report_content.extend([
                    f"## Missing Values\n\n",
                    f"```\n{self.results['missing']}\n```\n\n"
                ])
            
            # Save report to file
            filename = f"{self.name}_report_{self.timestamp}.md"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(''.join(report_content))
            
            print(f"Report generated and saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error generating report: {e}")
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Example usage
    print("This is a reusable Data Analysis Agent for REPL environment.")
    print("Import this module in your analysis script to use the agent.")
    
    # Simple demonstration if run directly
    agent = DataAnalysisAgent()
    print(f"Agent '{agent.name}' is ready to use.")
    print("Example methods: load_data(), analyze(), visualize(), save_results(), generate_report()")
