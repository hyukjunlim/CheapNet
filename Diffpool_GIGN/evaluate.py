import os
import pandas as pd
from collections import defaultdict
from glob import glob

path = 'model'
# path = 'save/g-d-c/q2q2'

# Get all subdirectories in the path directory
model_directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
model_directories = sorted(model_directories)

# Processing each directory
for model_root in model_directories:
    print(f'==================== {model_root} ====================')
    models = ['GIGN']
    model_performance_dict = defaultdict(list)
    model_root_path = os.path.join(path, model_root)
    results_dict = defaultdict(list)
    for model_name in models:
        model_dirs = sorted(glob(os.path.join(model_root_path, f'repeat*')))
        best_dict = defaultdict(list)
        for md in model_dirs:
            log_path = os.path.join(md, 'log', 'train', 'Train.log')
            with open(log_path, 'r') as f:
                logs = f.readlines()
                check = False
                log = logs[-1]
                if 'test2013_pr' in log:
                    messages = log.split(', ')
                    for msg in messages:
                        parts = msg.split('-')
                        key = parts[-2].rstrip().replace(',', '')
                        try:
                            val = float(parts[-1].rstrip().replace(',', ''))
                            results_dict[key].append(val)
                        except ValueError:
                            continue

    model_df = pd.DataFrame(results_dict)
    print(model_df)
    if 'test2013_pr' not in model_df.columns:
        continue
    
    performance_df = model_df.describe()
    for indicator in ['test2013_rmse', 'test2013_pr', 'test2016_rmse', 'test2016_pr', 'test2019_rmse', 'test2019_pr']:
        if indicator in performance_df.columns:
            res = performance_df[indicator]
            print("%s: %.4f (%.4f)" % (indicator, res.iloc[1], res.iloc[2]))
    print()
