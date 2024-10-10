import pandas as pd
import matplotlib.pyplot as plt

# Load the three CSV files
default_cwmin_path = 'Default_CwMin.csv'
ml_cwmin_path = 'ML_CwMin.csv'
fed_ml_path = 'Fed_CwMin.csv'

# Read the CSV files into dataframes
default_cwmin_df = pd.read_csv(default_cwmin_path)
ml_cwmin_df = pd.read_csv(ml_cwmin_path)
fed_ml_df = pd.read_csv(fed_ml_path)

# Define the columns to compare
columns_to_compare = ['Throughput (Mbps)', 'Fairness', 'Lost Packets']

# Function to plot the comparison bar graphs
def plot_comparison_bar(default_df, ml_df, fed_df, columns):
    for column in columns:
        plt.figure(figsize=(14, 7))
        bar_width = 0.25
        index = default_df['Router ID']

        plt.bar(index - bar_width, default_df[column], bar_width, label='Default CW')
        plt.bar(index, ml_df[column], bar_width, label='Local ML  Optimized CW')
        plt.bar(index + bar_width, fed_df[column], bar_width, label='FL Optimized CW')

        plt.xlabel('Router ID')
        plt.ylabel(column)
        plt.title(f'{column} Comparison by Router ID')
        plt.xticks(index)
        plt.legend()
        plt.grid(True, axis='y')
        plt.show()

# Create bar plots for each column
plot_comparison_bar(default_cwmin_df, ml_cwmin_df, fed_ml_df, columns_to_compare)
