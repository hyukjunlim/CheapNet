import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Data from the table
data = {
    "Model": ["IGN", "EGNN", "GIGN", "CheapNet"],
    "Memory (b=4)": [97, 113, 60, 132],
    "Memory (b=8)": [168, 203, 111, 247],
    "Memory (b=16)": [309, 380, 211, 470],
    "Memory (b=32)": [590, 734, 413, 914],
    "Memory (b=64)": [1150, 1442, 813, 1796],
    "Memory (b=128)": [2270, 2859, 1610, 3561],
    "Memory (b=256)": [4466, 5629, 3172, 7013]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Batch sizes as an array for linear regression calculation
batch_sizes = np.array([4, 8, 16, 32, 64, 128, 256])

# Plotting the memory usage for different models and batch sizes
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for each model
colors = ["blue", "green", "red", "purple"]

# Loop through each model, plot data points, and perform linear regression
for idx, model in enumerate(df["Model"]):
    # Extract memory usage data for the current model
    memory_usage = np.array(df.iloc[idx, 1:]).astype(float)  # Ensure the data is converted to float
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(batch_sizes, memory_usage)
    
    # Generate fitted line using the regression slope and intercept
    fitted_line = slope * batch_sizes + intercept
    
    # Plot the original data points
    ax.plot(batch_sizes, memory_usage, label=f"{model} (slope={slope:.2f}, R^2={r_value**2:.2f})", marker='o', color=colors[idx])
    
    # Plot the linear regression line
    ax.plot(batch_sizes, fitted_line, color=colors[idx], alpha=0.5)

# Set labels and title
ax.set_xlabel('Batch Size')
ax.set_ylabel('Memory Usage (MB)')
ax.set_title('Memory Footprint Experiment Across Batch Sizes')

# Add a legend
ax.legend()

# Display the plot
plt.tight_layout()

# Save the figure in both PNG and PDF formats
plt.savefig('figure/memory_footprint.png', format='png', bbox_inches='tight')
plt.savefig('figure/memory_footprint.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()
