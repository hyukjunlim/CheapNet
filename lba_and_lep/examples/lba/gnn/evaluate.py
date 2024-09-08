import os
import sys
sys.path.append(os.path.abspath('/data/project/dlagurwns03/GIGN/codes/lba_and_lep'))
import numpy as np
import torch
import atom3d.util.results as res
import atom3d.util.metrics as met
seqid = 60

# Define the training run 
for rep in range(1):
    for g in [1]:
        name = f'save/lba_test_withH_{seqid}_q2q2_{rep}_{g}/'
        print(name)

        # Load training results
        rloader = res.ResultsGNN(name, reps=[0,1,2])
        results = rloader.get_all_predictions()

        # Calculate and print results
        summary = met.evaluate_average(results, metric = met.rmse, verbose = False)
        print('Test RMSE: %6.3f \pm %6.3f'%summary[2])
        summary = met.evaluate_average(results, metric = met.pearson, verbose = False)
        print('Test Pearson: %6.3f \pm %6.3f'%summary[2])
        summary = met.evaluate_average(results, metric = met.spearman, verbose = False)
        print('Test Spearman: %6.3f \pm %6.3f'%summary[2])

# # Define the training run 
# for reps in [0]:
#     l = []
#     ls = []
#     for rep in range(3):
#         for g in [0]:
#             name = f'logs/lba_test_withH_{seqid}_q2q2_{rep}_{reps}'
#             # print(name)

#             # Load training results
#             rloader = res.ResultsGNN(name, task='lba', reps=[0])
#             results = rloader.get_all_predictions()

#             # Calculate and print results
#             summary_r = met.evaluate_average(results, metric = met.rmse, verbose = False)
#             # print('Test RMSE: %6.3f \pm %6.3f'%summary_r[2])
#             summary_s = met.evaluate_average(results, metric = met.spearman, verbose = False)
#             # print('Test Spearman: %6.3f \pm %6.3f'%summary_s[2])
#             summary_p = met.evaluate_average(results, metric = met.pearson, verbose = False)
#             # print('Test Pearson: %6.3f \pm %6.3f'%summary_p[2])

#             l.append((summary_r[2][0], summary_p[2][0], summary_s[2][0], name))
#             ls.append(summary_r[2][0])

#     # sort ls and get the best 5
#     ls = np.array(ls)
#     idx = np.argsort(ls)
#     print('Best 5')
#     for i in range(2):
#         print(l[idx[i]])
    


