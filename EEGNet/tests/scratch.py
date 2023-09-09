data_path = "/home/kaleb/Documents/eeg_dataset/files/S001/S001R10.edf" # grabbing one subject
# num = int(data_path[-5:-7])
num = int(data_path[-5] + data_path[-6])
other = int(data_path[-6] + data_path[-5])
print(num)
print(other)