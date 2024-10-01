import os
import sys
sys.path.append(os.path.abspath('/home/users/CheapNet/atom3d'))
import atom3d.util.results as res
import atom3d.util.metrics as met

name = 'LEP_best_models'

# Load training results
rloader = res.ResultsGNN(name, task='lep', reps=[0,1,2])
results = rloader.get_all_predictions()

# Calculate and print results
summary_roc = met.evaluate_average(results, metric = met.auroc, verbose = False)
print('Test AUROC: %6.3f \pm %6.3f'%summary_roc[2])

summary_prc = met.evaluate_average(results, metric = met.auprc, verbose = False)
print('Test AUPRC: %6.3f \pm %6.3f'%summary_prc[2])
