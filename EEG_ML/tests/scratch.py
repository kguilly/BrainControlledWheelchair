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
# plt.show()

###############################################
### READING THE .EVENT FILES
print("\n\n###################################\n")
print("READING THE .EVENT FILES")
import mne 

file = '/home/kaleb/Documents/eeg_dataset/files/S001/S001R14.edf.event'
# annotations = mne.read_annotations(file)
# print(annotations)

# Initialize lists to store annotation information
onsets = []
durations = []
descriptions = []

# Read the file line by line and parse the annotations

with open(file, 'rb') as file:
    for line in file:
        print(line)
    
        # parts = line.strip().split()  # Split each line into parts
        # if len(parts) >= 3:
        #     onset = float(parts[0])  # Parse onset time
        #     duration = float(parts[1])  # Parse duration
        #     description = parts[2]  # Parse annotation description
        #     onsets.append(onset)
        #     durations.append(duration)
        #     descriptions.append(description)


# Create an Annotations object manually
# annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)

# Now you can work with the annotations object
# print(annotations)


########################################################
print("\n\n###################################\n")
print("READING THE ENTIRE FILE")
import pyedflib
import numpy as np

file = '/home/kaleb/Documents/eeg_dataset/files/S001/S001R14.edf'

edf_data = pyedflib.EdfReader(file)

# print all the attributes
print(edf_data.file_info)
print("file duration: ", edf_data.getFileDuration())
print("num samples: ", edf_data.getNSamples())

annotation = edf_data.readAnnotations()

eeg_data = []
for channel in range(64):
    arr = edf_data.readSignal(channel)
    eeg_data.append(arr)

print(" ")
eeg_data = np.stack(eeg_data)
annotation = np.stack(annotation)
print(eeg_data.shape)

print("annotations: ", annotation)