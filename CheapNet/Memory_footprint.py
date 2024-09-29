import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
plt.rcParams.update({'font.size': 14})

# Data from the table in the image
data = {
    "Model": ["GAABind", "DEAttentionDTA", "CheapNet"],
    "Memory (b=2)": [21553, 1975, 1059],
    "Memory (b=4)": [None, 2397, 1091],
    "Memory (b=32)": [None, 7321, 2383],
    "Memory (b=64)": [None, 12977, 3515],
    "Memory (b=128)": [None, None, 5763],
}

df = pd.DataFrame(data)

# Batch sizes as an array for linear regression calculation
batch_sizes = np.array([0, 1, 2.5, 3.5, 4.5])

# Plotting the memory usage for different models and batch sizes
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for each model
colors = ["blue", "green", "purple"]

# Loop through each model, plot data points, and perform linear regression
for idx, model in enumerate(df["Model"]):
    memory_usage = np.array(df.iloc[idx, 1:3]).astype(float) / 1024  # Convert to GB
    ax.plot(batch_sizes[:2], memory_usage, label=f"{model}", marker='o', color=colors[idx])
    memory_usage = np.array(df.iloc[idx, 3:]).astype(float) / 1024  # Convert to GB
    ax.plot(batch_sizes[2:], memory_usage, marker='o', color=colors[idx])

oom_value = 24  # OOM value for DEAttentionDTA in GB
ax.hlines(oom_value, 4.3, 4.8, colors='green', linestyles='dashed')
ax.text(4.8, oom_value, 'OOM', ha='right', va='bottom', color='green')
ax.hlines(oom_value, 0.5, 1, colors='blue', linestyles='dashed')
ax.text(1, oom_value, 'OOM', ha='right', va='bottom', color='blue')

# connect for DEAttentionDTA, (6, 14887/1024) to (6.8, 24)
x = [3.5, 4.3]
y = [12977 / 1024, 24]
ax.plot(x, y, color='green')

x = [0, 0.5]
y = [21553 / 1024, 24]
ax.plot(x, y, color='blue')

x = [1, 2.5]
y = [2397 / 1024, 7321 / 1024]
ax.plot(x, y, color='green', linestyle='dashed')

x = [1, 2.5]
y = [1091 / 1024, 2383 / 1024]
ax.plot(x, y, color='purple', linestyle='dashed')

# bold line, not dashed
ax.vlines(1.75, 0, 27, colors='black', linestyles='solid', linewidth=2)
# ax.text(1.65, 26.5, 'Small complex', ha='right', va='top', color='black',)
# ax.text(1.85, 26.5, 'Large complex', ha='left', va='top', color='black')

# enlarge font
# Set labels and title
ax.set_xlabel('Batch Size')
ax.set_ylabel('Memory Usage (GB)')
ax.set_title('Memory Footprint Experiment of Attention-based Models')
ax.set_xlim(-0.25, 5.5)
ax.set_ylim(0, 27)
ax.legend(loc=7)

plt.xticks([0, 1, 2.5, 3.5, 4.5], [2, 4, 32, 64, 128])
# make y axis step by 4
plt.yticks(np.arange(0, 25, 4))
# Display the plot
plt.tight_layout()

# Save the figure in both PNG and PDF formats
plt.savefig('figure/memory_footprint.png', format='png', bbox_inches='tight')
plt.savefig('figure/memory_footprint.pdf', format='pdf', bbox_inches='tight')

