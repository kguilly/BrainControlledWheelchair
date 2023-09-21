'''

This file was developed by Kaleb Guillot for the purposes of the 
senior design project: The Brain-Controlled Wheelchair

The dataset used:  
https://physionet.org/content/eegmmidb/1.0.0/

from the PhysioToolkit Software:
https://archive.physionet.org/physiotools/ 

to run this file from the command line: 
(env) Documents/GitHub/BCWheelchair_ML $ python -m tests.test_1



Questions that I have to answer: 
- How long of durations do I want to sample? 


Conclusions from reading the Documentation on the datset: 
- Use task 1 and 3 bc tasks 2 and 4 are imagined versions of 1 and 3
- Tasks 1 and 3 are in files: 
    Task 1: 
        R03, R07, R11
    Task 3:
        R05, R09, R13

- Need 5 different labels: 
    - Relaxed
    - Squeeze both fists
    - Squeeze both feet
    - Squeeze left hand
    - Squeeze right hand

- Need to read .event files in order to know when 
'''
import numpy as np
import os
import pyedflib

# EEGNet specific imports
# import EEGNet
from EEG_ML.EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

import read_edf_files
kernels, chans = 1, 64
label_mapping = {
        1: "Rest",
        2: "Squeeze Both Fists",
        3: "Squeeze Both Feet",
        4: "Squeeze Left Hand",
        5: "Squeeze Right Hand",
    }
num_labels = 5
X, Y = read_edf_files.reader() # use other function to read the edf files
################################################################
## Process, filter, and epoch the data
# init arrays to train/validate/test. Make split 50/25/25
half = int(len(X) / 2)
quarter = int(half / 2)
three_fourths = half + quarter

X_train = X[:half, :, :]
X_validate = X[half : three_fourths, :, :]
X_test = X[three_fourths:, :, :]

y_train = Y[:half]
y_validate = Y[half:three_fourths]
y_test = Y[three_fourths:]

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

model = EEGNet(nb_classes=num_labels, Chans=X_train.shape[1], Samples=X_train.shape[2],
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
class_weights = {0:1, 1:1, 2:1, 3:1, 4:1}

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