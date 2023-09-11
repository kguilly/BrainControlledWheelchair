import numpy as np
import os 
import pyedflib


data_path = "/home/kaleb/Documents/eeg_dataset/files/S001/S001R10.edf" # grabbing one subject
# num = int(data_path[-5:-7])
num = int(data_path[-5] + data_path[-6])
other = int(data_path[-6] + data_path[-5])
print(num)
print(other)


##########################################################
print("\n\n###################################\n")
print("Reading from S001R10")
edf_data = pyedflib.EdfReader(data_path)
arr = edf_data.readSignal(1)

edf_data.close()
print("Shape: ", arr.shape)

## visualize the data
import matplotlib.pyplot as plt

time = np.arange(len(arr))
plt.figure(figsize=(10,4))
plt.plot(time, arr, lw=0.5)
plt.title('EEG data')
plt.grid(True)
plt.show()