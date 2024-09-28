import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Data from the table in the image
data = {
    "Model": ["GAABind", "DEAttentionDTA", "CheapNet"],
    "Memory (b=1)": [21911, 1967, 1106],
    "Memory (b=2)": [None, 2175, 1127],
    "Memory (b=4)": [None, 2595, 1205],
    "Memory (b=8)": [None, 3421, 1329],
    "Memory (b=16)": [None, 5041, 1601],
    "Memory (b=32)": [None, 8329, 2139],
    "Memory (b=64)": [None, 14887, 3363],
    "Memory (b=128)": [None, None, 5635],
    "Memory (b=256)": [None, None, 10373]
}

df = pd.DataFrame(data)

# Batch sizes as an array for linear regression calculation
batch_sizes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

# Plotting the memory usage for different models and batch sizes
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for each model
colors = ["blue", "green", "purple"]

# Loop through each model, plot data points, and perform linear regression
for idx, model in enumerate(df["Model"]):
    memory_usage = np.array(df.iloc[idx, 1:]).astype(float) / 1024  # Convert to GB
    ax.plot(batch_sizes, memory_usage, label=f"{model}", marker='o', color=colors[idx])

oom_value = 24  # OOM value for DEAttentionDTA in GB
ax.hlines(oom_value, 6.8, 8, colors='green', linestyles='dashed')
ax.text(7, oom_value, 'OOM', ha='right', va='bottom', color='green')

oom_value = 24  # OOM value for DEAttentionDTA in GB
ax.hlines(oom_value, 0.1, 1, colors='blue', linestyles='dashed')
ax.text(1, oom_value, 'OOM', ha='right', va='bottom', color='blue')

# connect for DEAttentionDTA, (6, 14887/1024) to (6.8, 24)
x = [6, 6.8]
y = [14887 / 1024, 24]
ax.plot(x, y, color='green')

x = [0, 0.1]
y = [21911 / 1024, 24]
ax.plot(x, y, color='blue')

# Set labels and title
ax.set_xlabel('Batch Size')
ax.set_ylabel('Memory Usage (GB)')
ax.set_title('Memory Footprint Experiment of Attention-based Models')
ax.set_ylim(0, 25)
ax.legend()

# Display the plot
plt.tight_layout()

# Save the figure in both PNG and PDF formats
plt.savefig('figure/memory_footprint.png', format='png', bbox_inches='tight')
plt.savefig('figure/memory_footprint.pdf', format='pdf', bbox_inches='tight')

