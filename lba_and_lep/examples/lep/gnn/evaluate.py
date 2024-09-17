import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath('/data/project/dlagurwns03/GIGN/codes/lba_and_lep'))
import atom3d.util.results as res
import atom3d.util.metrics as met

# for rep in range(1):
#     for g in [0]:
#         name = f'save/crucial//lep_test_q2q2_{rep}_{g}'
#         print(name)

#         # Load training results
#         rloader = res.ResultsGNN(name, task='lep', reps=[0,1,2])
#         results = rloader.get_all_predictions()

#         # Calculate and print results
#         summary_roc = met.evaluate_average(results, metric = met.auroc, verbose = False)
#         print('Test AUROC: %6.3f \pm %6.3f'%summary_roc[2])

#         summary_prc = met.evaluate_average(results, metric = met.auprc, verbose = False)
#         print('Test AUPRC: %6.3f \pm %6.3f'%summary_prc[2])

total = []
totals = []
for reps in [1]:
    l = []
    ls = []
    lname = []
    for dir in os.listdir('logs'):
        name = f'logs/{dir}'
        if not name.endswith(f'{reps}'):
            continue

        # Load training results
        rloader = res.ResultsGNN(name, task='lep', reps=[reps])
        try:
            results = rloader.get_all_predictions()
        except:
            continue

        # Calculate and print results
        summary_roc = met.evaluate_average(results, metric = met.auroc, verbose = False)
        # print('Test AUROC: %6.3f \pm %6.3f'%summary_roc[2])

        summary_prc = met.evaluate_average(results, metric = met.auprc, verbose = False)
        # print('Test AUPRC: %6.3f \pm %6.3f'%summary_prc[2])

        l.append((summary_roc[2][0], summary_prc[2][0]))
        ls.append((summary_roc[2][0] + summary_prc[2][0])/2)
        lname.append(name[19:])

    # max index
    # print(np.argmax(ls))
    # print(l[np.argmax(ls)])
    total.append(lname[np.argmax(ls)])
    totals.append(l[np.argmax(ls)])
print(f'Best model: {total}')
print(f'Result: {totals}')
print(f'Mean: {np.mean(totals, axis=0)}')

