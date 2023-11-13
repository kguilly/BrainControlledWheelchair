
# this file needs to:
'''
- pick the best hyperparameters
- train the model
- use the model on incoming data
- function to filter the data (depending on the results of tst 0.2.3.1.3 and 0.2.3.1.4)
'''

import time
import os

import numpy as np
import pandas as pd

from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

from EEG_ML.tests import read_edf_files as ref
from EEG_ML.EEGModels import EEGNet

# append the path to eeget
import sys
from contextlib import contextmanager
@contextmanager
def add_to_path(directory):
    sys.path.append(directory)
    try:
        yield
    finally:
        sys.path.remove(directory)
curr_file_path = os.path.dirname(os.path.abspath(__file__))
with add_to_path(curr_file_path):
    from EEGModels import EEGNet
    import read_edf_files as ref


# IMPORTANT: this var sets how many samples will be used to get predictions
num_samples = 180 # this is about a second and a half with the ultracortex mark iv
samples_to_jump_by = 18 # for the convolutional split training, this is the number of samples 
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
num_channels = len(eeg_channels)


def get_trained_model(X, Y, dropoutRate=0.5, kernels=2, kernLength=32, F1=8, D=2, F2=16, batch_size=16):
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
    y_train = np_utils.to_categorical([x - 1 for x in y_train])
    y_validate = np_utils.to_categorical([x - 1 for x in y_validate])
    y_test = np_utils.to_categorical([x - 1 for x in y_test])

    # convert data to NHWC (trials, channels, samples, kernels) format
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], kernels)
    X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], X_validate.shape[2], kernels)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], kernels)

    print('x_train shape: ', X_train.shape, '\ny_train shape: ', y_train.shape)
    ################################################################
    ## Call EEGNet
    model = EEGNet(nb_classes=5, Chans=X_train.shape[1], Samples=X_train.shape[2],
                   dropoutRate=dropoutRate, kernLength=kernLength, F1=F1, D=D, F2=F2,
                   dropoutType='Dropout')

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # set a valid path for your system to record model checkpoints
    chkpt_filepath = '/home/kaleb/tmp/checkpoint.h5'
    checkpointer = ModelCheckpoint(filepath=chkpt_filepath, verbose=1,
                                   save_best_only=True)
    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    model.fit(X_train, y_train, batch_size=batch_size, epochs=30,
              verbose=2, validation_data=(X_validate, y_validate),
              callbacks=[checkpointer], class_weight=class_weights)
    
    return model

def get_model_acc(trained_model, X_test, Y_test): 
    
    probs = trained_model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))
    
    return acc 

def train_the_model(profile_path): 
    global num_samples, eeg_channels, num_channels, samples_to_jump_by
    # call function to pick the best hyperparameters

    # find the headset data path
    data_dir = os.path.join(profile_path, 'headset_data')

    # read each file in the directory and load it in
    X = []
    Y = []
    for file in os.listdir(data_dir):
        # load the data into a file
        full_dir = os.path.join(data_dir, file)
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

        # separate the data in a convolutional manner
        this_x, this_y = ref.convolutional_split(eeg_data_3d, this_y, samples_to_jump_by, num_samples, num_channels)

        # append to the main X and Y
        try:  
            X = np.vstack((X, this_x))
        except:
            X = this_x
        for label in this_y:
            Y.append(label)

    # process, filter, and epoch the data 
    model = get_trained_model(X, Y)
    acc = get_model_acc(model)


def get_best_hyperparams():
    pass

def generate_prediction(): # function to generate prediction given the trained model
    pass
