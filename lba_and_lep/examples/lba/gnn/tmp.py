# read .npy file and print the shape of the array
# file name is lba_test.py
import numpy as np

data = np.load('lba_test.npy', allow_pickle=True)
# change ndarray to dict
data = data.item()
print(data.keys())
