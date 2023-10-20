# This is test 0.2.2.4 of the Brain-Controlled Wheelchair Senior Design II
# Project. This test takes labeled data, similar to that of test 0.2.2.3 and formats it
# in order to train a deep learning model, EEGNet. This file also includes code on
# how to save the fitted model for use in later generating predictions.

import time
import os

import numpy as np
import pandas as pd

# from tensorflow import keras
# from keras import utils as np_utils
# from keras import ModelCheckpoint

from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint



from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

from EEG_ML.tests import read_edf_files as ref
from EEG_ML.EEGModels import EEGNet

eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

cur_dir = os.path.dirname(os.path.abspath(__file__))
folder_dir = os.path.join(cur_dir, "test_data/headset_data")

X = []
Y = []
for file in os.listdir(folder_dir):
    # load the data into a file
    full_dir = os.path.join(folder_dir, file)
    data = DataFilter.read_file(full_dir)
    eeg_data = data[eeg_channels, :]
    eeg_data_3d = np.reshape(eeg_data, (1, eeg_data.shape[0], eeg_data.shape[1]))
    # print(eeg_data_3d.shape)

    this_y = []
    # based on name of file, select the label
    if file == 'rest.csv':
        this_y.append(1)
    elif file == 'forward.csv':
        this_y.append(2)
    elif file == 'backward.csv':
        this_y.append(3)
    elif file == 'left.csv':
        this_y.append(4)
    elif file == 'right.csv':
        this_y.append(5)

    # separate the data by seconds
    this_x, this_y = ref.split_by_second(eeg_data_3d, this_y, 120)
    # print(f'this_x: {this_x.shape}, and len of this_y: {len(this_y)}')

    # append to the main X and Y
    try:
        X = np.vstack((X, this_x))
    except:
        X = this_x
    for label in this_y:
        Y.append(label)

# results in this_x: (5, 16, 120), and this_y: 5 for one file, we're good to go
print(f'X.shape: {X.shape}, and len of Y: {len(Y)}')

## Process, filter, and epoch the data
# init arrays to train/validate/test. Make split 50/25/25
half = int(len(X) / 2)
quarter = int(half / 2)
three_fourths = half + quarter

X_train = X[:half, :, :]
X_validate = X[half: three_fourths, :, :]
X_test = X[three_fourths:, :, :]

y_train = Y[:half]
y_validate = Y[half:three_fourths]
y_test = Y[three_fourths:]

# convert labels to one-hot encoding
## TODO: fix dis
y_train = np_utils.to_categorical(y_train - 1)
y_validate = np_utils.to_categorical(y_validate - 1)
y_test = np_utils.to_categorical(y_test - 1)

# convert data to NHWC (trials, channels, samples, kernels) format
kernels = 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], kernels)
X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], X_validate.shape[2], kernels)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], kernels)

print('x_train shape: ', X_train.shape, '\ny_train shape: ', y_train.shape)
################################################################
## Call EEGNet
num_labels = 5
model = EEGNet(nb_classes=num_labels, Chans=X_train.shape[1], Samples=X_train.shape[2],
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
               dropoutType='Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# count number of parameters in the model
numParams = model.count_params()

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                               save_best_only=True)

# the weights all to be 1
class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

# fittedModel =
model.fit(X_train, y_train, batch_size=16, epochs=300,
          verbose=2, validation_data=(X_validate, y_validate),
          callbacks=[checkpointer], class_weight=class_weights)

# load optimal weights
model.load_weights('/tmp/checkpoint.h5')

probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

# save the model to a file
from keras.models import load_model
model.save('model.h5') # save to the user's directory!

# load the model back into an obj
loaded_model = load_model('model.h5')







