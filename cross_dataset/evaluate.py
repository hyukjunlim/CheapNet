import os
import pandas as pd
from collections import defaultdict
from glob import glob

path = 'Cross_best_models'

print(f'==================== {path} ====================')
models = ['CheapNet']
model_performance_dict = defaultdict(list)
results_dict = defaultdict(list)

for model_name in models:
    model_dirs = sorted(glob(os.path.join(path, f'repeat*')))
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
            print("%s: %.3f (%.3f)" % (indicator, res.iloc[1], res.iloc[2]))
    print()
