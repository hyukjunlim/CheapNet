import os
import sys
sys.path.append(os.path.abspath('/home/users/CheapNet/atom3d'))
import atom3d.util.results as res
import atom3d.util.metrics as met

seqid = 60

name = f'LBA_{seqid}_best_model'
rloader = res.ResultsGNN(name, reps=[0,1,2])
results = rloader.get_all_predictions()

# Calculate and print results
summary = met.evaluate_average(results, metric = met.rmse, verbose = False)
print('Test RMSE: %6.3f \pm %6.3f'%summary[2])
summary = met.evaluate_average(results, metric = met.pearson, verbose = False)
print('Test Pearson: %6.3f \pm %6.3f'%summary[2])
summary = met.evaluate_average(results, metric = met.spearman, verbose = False)
print('Test Spearman: %6.3f \pm %6.3f'%summary[2])