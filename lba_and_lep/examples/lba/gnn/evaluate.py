import os
import sys
sys.path.append(os.path.abspath('/data/project/dlagurwns03/GIGN/codes/lba_and_lep'))
import numpy as np
import torch
import atom3d.util.results as res
import atom3d.util.metrics as met
seqid = 30

# Define the training run 
for rep in range(1):
    for g in [1]:
        name = f'save/crucial/{seqid}/lba_test_withH_{seqid}_q2q2_{rep}_{g}/'
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
# for reps in [0, 1]:
#     l = []
#     ls = []
#     for rep in range(2):
#         for g in [0, 1]:
#             name = f'logs/lba_test_withH_{seqid}_q2q2_{rep}_{reps}'
#             # print(name)

#             # Load training results
#             rloader = res.ResultsGNN(name, task='lba', reps=[reps])
#             try:
#                 results = rloader.get_all_predictions()
#             except:
#                 continue

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
#     print(ls)
#     ls = np.array(ls)
#     idx = np.argsort(ls)
#     print('Best 5')
#     for i in range(1):
#         print(l[idx[i]])
    


