'''

This file was developed by Kaleb Guillot for the purposes of the 
senior design project: The Brain-Controlled Wheelchair

The dataset used: bci2000: 
https://physionet.org/content/eegmmidb/1.0.0/

to run this file from the command line: 
(env) Documents/GitHub/BCWheelchair_ML $ python -m tests.test_1


Questions that I have to answer: 
- How long of durations do I want to sample? 
- How do I want to separate training and testing? 


'''
import numpy as np
import os
import pyedflib

# EEGNet specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K


kernels, channels, samples = 1, 64, 1

################################################################
## Load the dataset and order it 
data_path = "/home/kaleb/Documents/eeg_dataset/files/S001/" # grabbing one subject
all_files = os.listdir(data_path)

edf_files = [file for file in all_files if file.endswith('.edf')] # grab each of the files ending in '.edf'


# init arrays to train/validate/test. Make split 50/25/25
X_train = []
X_validate = []
X_test = []

y_train = []
y_validate = []
y_test = []


for file in edf_files: # grab all the relevant data from the edf files
    path = os.path.join(data_path, file)
    edf_data = pyedflib.EdfReader(path)
    
    tmp_x_train = []
    tmp_x_val = []
    tmp_x_test = []

    for channel in range(channels):
        arr = edf_data.readSignal(channel)
        
        half = int(len(arr) / 2)
        quarter = int(half / 2)
        
        tmp_x_train = arr[0:half]
        tmp_x_val = arr[half:(half+quarter)]
        tmp_x_test = arr[(half+quarter):len(arr)]
        
    edf_data.close()

    # after reading from a file, get the label array
    label = int(path[-6] + path[-5])

    tmp_y_train = np.full_like(tmp_x_train, label)
    tmp_y_val = np.full_like(tmp_x_val, label)
    tmp_y_test = np.full_like(tmp_x_test, label)

    X_train.append(tmp_x_train)
    X_test.append(tmp_x_test)
    X_validate.append(tmp_x_val)

    y_train.append(tmp_y_train)
    y_test.append(tmp_y_test)
    y_validate.append(tmp_y_val)



    

print('done')


################################################################
## Process, filter, and epoch the data

y_train = y_train[0]
y_validate = y_validate[0]
y_test = y_test[0]

# convert labels to one-hot encoding
y_train = np_utils.to_categorical(y_train-1)
y_validate = np_utils.to_categorical(y_validate-1)
y_test = np_utils.to_categorical(y_test-1)

# convert data to NHWC (trials, channels, samples, kernels) format
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], kernels)
X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], X_validate.shape[2], kernels)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], kernels)

print('x_train shape: ', X_train.shape, '\ny_train shape: ', y_train.shape)

################################################################
## Call EEGNet