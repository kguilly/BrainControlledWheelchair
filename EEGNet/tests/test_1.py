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

# # init arrays to train/validate/test. Make split 50/25/25
# X_train = []
# X_validate = []
# X_test = []

# y_train = []
# y_validate = []
# y_test = []


# for file in edf_files: # grab all the relevant data from the edf files
#     path = os.path.join(data_path, file)
#     edf_data = pyedflib.EdfReader(path)
    
#     tmp_x_train = []
#     tmp_x_val = []
#     tmp_x_test = []

#     for channel in range(channels):
#         arr = edf_data.readSignal(channel)
        
#         half = int(len(arr) / 2)
#         quarter = int(half / 2)
        
#         tmp_x_train = arr[0:half]
#         tmp_x_val = arr[half:(half+quarter)]
#         tmp_x_test = arr[(half+quarter):len(arr)]
        
#     edf_data.close()

#     # after reading from a file, get the label array
#     label = int(path[-6] + path[-5])

#     tmp_y_train = np.full_like(tmp_x_train, label)
#     tmp_y_val = np.full_like(tmp_x_val, label)
#     tmp_y_test = np.full_like(tmp_x_test, label)

#     X_train.append(tmp_x_train)
#     X_test.append(tmp_x_test)
#     X_validate.append(tmp_x_val)

#     y_train.append(tmp_y_train)
#     y_test.append(tmp_y_test)
#     y_validate.append(tmp_y_val)


# init arrays to train/validate/test. Make split 50/25/25
X_train = []
X_validate = []
X_test = []

y_train = []
y_validate = []
y_test = []

num_labels = 0

for file in edf_files:
    eeg_arrays = []
    path = os.path.join(data_path, file)
    edf_data = pyedflib.EdfReader(path)

    for channel in range(channels):
        arr = edf_data.readSignal(channel)
        eeg_arrays.append(arr)

    edf_data.close()

    # print("shape: ", len(eeg_arrays), ' ', len(eeg_arrays[1]))

    # now stack it into 20 different trials, and make the corresponding Y label
    try: 
        combined_arr = np.stack(eeg_arrays)
        # reshape
        segmented_arr = combined_arr.reshape((20,64,1000))

        # grab the label for the data based on the file name
        label = int(path[-6] + path[-5])
        y_tmp = np.full((20, ), label)

        # take 10 for train, 5 for validate, and 5 for testing
        try: 
            X_train = np.concatenate((X_train, segmented_arr[:10, :, :]), axis=0)
            X_validate = np.concatenate((X_validate, segmented_arr[10:15, :, :]), axis=0)
            X_test = np.concatenate((X_test, segmented_arr[15:20, :, :]), axis=0)

            y_train = np.concatenate((y_train, y_tmp[:10]), axis=0)
            y_test = np.concatenate((y_test, y_tmp[10:15]), axis=0)
            y_test = np.concatenate((y_test, y_tmp[15:20]), axis=0)

        except: 
            X_train = segmented_arr[:10, :, :]
            X_validate = segmented_arr[10:15, :, :]
            X_test = segmented_arr[15:20, :, :]

            y_train = y_tmp[:10]
            y_test = y_tmp[10:15]
            y_test = y_tmp[15:20]

        # this file went through the entire try/catch block, so add it to the
        # num_labels (will be used when calling eegnet)
        num_labels+=1
    except:
        print("Something wrong with shape of arrays for file: ", path)    
    
print(X_train.shape)
print(y_train.shape)
    

# print('done')


################################################################
## Process, filter, and epoch the data

# for each of the elements of the Y array, if the label is greater than 
# the num_labels, adjust 
# TODO: problem with the labels
# difference = 14 - num_labels
# for elem in y_train:
#     if elem > num_labels:
#         elem-=difference

# for elem in y_validate:
#     if elem > num_labels:
#         elem-=difference

# for elem in y_train:
#     if elem > num_labels:
#         elem -= difference


# convert labels to one-hot encoding
# y_train = np_utils.to_categorical(y_train-1)
# y_validate = np_utils.to_categorical(y_validate-1)
# y_test = np_utils.to_categorical(y_test-1)
y_train = np.eye(num_labels)[y_train]
y_test = np.eye(num_labels)[y_test]
y_validate = np.eye(num_labels)[y_validate]

# convert data to NHWC (trials, channels, samples, kernels) format
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], kernels)
X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], X_validate.shape[2], kernels)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], kernels)

print('x_train shape: ', X_train.shape, '\ny_train shape: ', y_train.shape)

################################################################
## Call EEGNet

model = EEGNet(nb_classes=num_labels, chans=X_train.shape[1], Samples=X_train.shape[2],
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
                 dropoutType= 'Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

# count number of parameters in the model
numParams    = model.count_params()    

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                               save_best_only=True)

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during 
# optimization to balance it out. This data is approximately balanced so we 
# don't need to do this, but is shown here for illustration/completeness. 
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
class_weights = {0:1, 1:1, 2:1, 3:1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
# Riemannian geometry classification (below)
################################################################################
fittedModel = model.fit(X_train, y_train, batch_size = 16, epochs = 300, 
                        verbose = 2, validation_data=(X_validate, y_validate),
                        callbacks=[checkpointer], class_weight = class_weights)

# load optimal weights
model.load_weights('/tmp/checkpoint.h5')

###############################################################################
# can alternatively used the weights provided in the repo. If so it should get
# you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
# system.
###############################################################################

# WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5 
# model.load_weights(WEIGHTS_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)  
acc         = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))