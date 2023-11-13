
# this file needs to:
'''
- pick the best hyperparameters
- train the model
- use the model on incoming data
- function to filter the data (depending on the results of tst 0.2.3.1.3 and 0.2.3.1.4)
'''

import time
import os
import itertools

import numpy as np
import pandas as pd

from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

# from EEG_ML.tests import read_edf_files as ref
# from EEG_ML.EEGModels import EEGNet

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

# what each of the predictions from the model mean
label_decoding = {
    1: 'rest',
    2: 'forward',
    3: 'backward',
    4: 'left',
    5: 'right',
}

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

    # after gathering together all of the data, call the function to get the best hyperparams
    dropoutRate, kernels, kernLength, F1, D, F2, batch_size = get_best_hyperparams(X, Y)
    
    # process, filter, and epoch the data 
    model = get_trained_model(X, Y, dropoutRate=dropoutRate, kernels=kernels, kernLength=kernLength,
                              F1=F1, D=D, F2=F2, batch_size=batch_size) # call the function to train the model
    
    last_quarter_idx = int(len(Y) * 0.75)
    X_test = X[last_quarter_idx:, :, :]
    Y_test = Y[last_quarter_idx:]
    
    acc = get_model_acc(model, X_test, Y_test) # call the function to get the acc

    # save the trained model to a file and return the accuracy
    model_dir = os.path.join(profile_path, 'trained_model.h5')
    model.save(model_dir)
    return acc


def get_best_hyperparams(X, Y): # function will return a df that shows every combination of hyperparam and 
                            # its accuaracy score
    hyperparameter_map = {
        'dropoutRate' : [0.4, 0.5, 0.6],
        'kernels' : [1, 2, 3],
        'kernLength' : [16, 32, 64],
        'F1' : [4, 8, 16],
        'D' : [1, 2, 3],
        'F2' : [8, 16, 32],
        'batch_size' : [8, 16, 32]
    }
    
    df = pd.DataFrame(columns=['combination', 'acc'])
    combinations = list(itertools.product(*(hyperparameter_map[param] for param in hyperparameter_map)))
    for combination in combinations:
        try:
            model = get_trained_model(X, Y, combination[0], combination[1], combination[2],
                                combination[3], combination[4], combination[5], combination[6])
            last_quarter_idx = int(len(Y) * 0.75)

            X_test = X[last_quarter_idx:, :, :]
            Y_test = Y[last_quarter_idx:]

            acc = get_model_acc(model, X_test, Y_test)

            # now add this info to the dataframe
            df.loc[len(df)] = [combination, acc]

        except:
            continue

    # grab the highest value of accuracy in the df
    best_combo_row = df[df['acc'] == df['acc'].max()]
    best_combo = best_combo_row.iloc[0]['combination']

    # translate to the names of the params in best combo row
    dropoutRate = best_combo[0]
    kernels = best_combo[1]
    kernLength = best_combo[2]
    f1 = best_combo[3]
    d = best_combo[4]
    f2 = best_combo[5]
    batch_size = best_combo[6]

    return dropoutRate, kernels, kernLength, f1, d, f2, batch_size 

def generate_prediction(board, profile_path): # function to generate prediction given the trained model
    # THIS FUNCTION ASSUMES: 
        # a session has already been activated
        # the session has been recording for at least a second and a half alread

    # load the user's trained model
    model_path = os.path.join(profile_path, 'trained_model.h5')
    model = load_model(model_path)

    # generate a prediction
    preds = []
    while len(preds) <= 10: # MAY CAUSE MORE PROBLEMS, resolved with threading
        time.sleep(0.1)
        try:
            data = board.get_data(num_samples)
        except:
            continue

        eeg_data = data[eeg_channels, :]
        eeg_3d_data = eeg_data.reshape(1, eeg_data.shape[0], 120, 1)

        # pass through the model
        probs = model.predict(eeg_3d_data)
        
        # get the highest values prediction
        index = np.argmax(probs)
        prediction = label_decoding.get(index)

        # append that prediction to the arr
        preds.append(prediction)

    # return the value which appears the most
    most_common_output = np.argmax(np.bincount(preds))
    return most_common_output

    
