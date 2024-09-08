import pandas as pd
from collections import defaultdict
import os
from glob import glob
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# List of modes
all_list = ['1', '2', '3', '4', '5'] + \
    ['12', '13', '14', '15', '23', '24', '25', '34', '35', '45'] + \
    ['123', '124', '125', '134', '135', '145', '234', '235', '245', '345'] + \
    ['1234', '1235', '1245', '1345', '2345'] + \
    ['12345']

data = []

# Data collection from log files
for mode in all_list:
    models = ['GIGN']
    model_root = f'./model/ours-adamw-gbad_{mode}'
    results_dict = defaultdict(list)
    for model_name in models:
        model_dir = glob(os.path.join(model_root, f'*_{model_name}_repeat*'))
        for md in model_dir:
            log_path = os.path.join(md, 'log', 'train', 'Train.log')

            with open(log_path, 'r') as f:
                logs = f.readlines()
                check = False
                for log in logs:
                    if 'best' in log or 'epoch-799' in log:
                        check = True
                    if 'test2013_pr' in log and check:
                        messages = log.split(', ')
                        record = {'mode': mode}
                        for msg in messages:
                            key, val = msg.split('-')[-2].rstrip().replace(',', ''), msg.split('-')[-1].rstrip().replace(',', '')
                            val = float(val)
                            record[key] = val
                        data.append(record)

df = pd.DataFrame(data)
print(df)

# Baseline values for overlay
baseline_values = {
    'test2013_rmse': 1.380,
    'test2013_pr': 0.821,
    'test2016_rmse': 1.190,
    'test2016_pr': 0.840,
    'test2019_rmse': 1.393,
    'test2019_pr': 0.641
}

# Function to plot enhanced bar charts for each metric in a subplot
def plot_enhanced_bar_chart(ax, metric, better_higher=True):
    subset = df[['mode', metric]].dropna()
    
    # Determine top-5 based on the better_higher flag
    if better_higher:
        top5 = subset.nlargest(5, metric)
    else:
        top5 = subset.nsmallest(5, metric)
    
    colors = ['red' if mode in top5['mode'].values else 'skyblue' for mode in subset['mode']]
    bars = ax.bar(subset['mode'], subset[metric], color=colors)
    
    # Adding value labels on bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    
    # Plot baseline value as a horizontal line
    baseline = baseline_values[metric]
    ax.axhline(baseline, color='green', linestyle='--', linewidth=2)
    
    # Annotate baseline value
    ax.text(len(subset['mode']) - 1, baseline, f'Baseline: {baseline}', color='green', va='bottom', ha='right', fontsize=10, backgroundcolor='white')
    
    # Set y-axis limits
    if 'rmse' in metric:
        ax.set_ylim(1, ax.get_ylim()[1])
    else:
        ax.set_ylim(0.5, ax.get_ylim()[1])

    ax.set_xlabel('Mode', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f'{metric} results based on various modes', fontsize=12)
    ax.tick_params(axis='x', rotation=90)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Plotting Performance Metrics
metrics = {
    'test2013_rmse': False,
    'test2013_pr': True,
    'test2016_rmse': False,
    'test2016_pr': True,
    'test2019_rmse': False,
    'test2019_pr': True
}

# Create a 3x2 subplot
fig, axs = plt.subplots(3, 2, figsize=(40, 30))

# Flatten the array of axes for easy iteration
axs = axs.flatten()

# Plot each metric in a subplot
for i, (metric, better_higher) in enumerate(metrics.items()):
    plot_enhanced_bar_chart(axs[i], metric, better_higher)

plt.tight_layout()
plt.savefig('metrics_subplot.png')

# Analyzing the Trends
performance_summary = df.describe()
print(performance_summary)

for metric in metrics:
    if metric in performance_summary.columns:
        metric_summary = performance_summary[metric]
        print(f"{metric}: Mean={metric_summary['mean']}, Std={metric_summary['std']}, Min={metric_summary['min']}, Max={metric_summary['max']}")
