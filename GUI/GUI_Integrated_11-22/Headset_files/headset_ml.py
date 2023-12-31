
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
# num_samples = 180 # this is about a second and a half with the ultracortex mark iv
num_samples = 230 # this is about a second with 8 channels connected
samples_to_jump_by = 25 # for the convolutional split training, this is the number of samples
kernels = 1
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
num_channels = len(eeg_channels)

# what each of the predictions from the model mean
label_decoding = {
    1: 'rest',
    2: 'forward',
    3: 'backward',
    4: 'left',
    5: 'right',
}

def get_trained_model(X, Y, dropoutRate=0.5, kernels=1, kernLength=32, F1=8,
                      D=2, F2=16, batch_size=16, epochs=300):
    global curr_file_path

    chkpt_file_path = os.path.join(curr_file_path, 'extra', 'checkpoint.h5')

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
    checkpointer = ModelCheckpoint(filepath=chkpt_file_path, verbose=1,
                                   save_best_only=True)
    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=2, validation_data=(X_validate, y_validate),
              callbacks=[checkpointer], class_weight=class_weights)
    
    return model, X_test, y_test

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

        this_y = []
        # based on name of file, select the label
        if file.lower() == 'rest.csv':
            this_y.append(1)
        elif file.lower() == 'forward.csv':
            this_y.append(2)
        elif file.lower() == 'backward.csv':
            this_y.append(3)
        elif file.lower() == 'left.csv':
            this_y.append(4)
        elif file.lower() == 'right.csv':
            this_y.append(5)
        else:
            print(f'{file} not in classification')
            continue

        # load the data into a file
        full_dir = os.path.join(data_dir, file)
        data = DataFilter.read_file(full_dir)
        eeg_data = data[eeg_channels, :]
        eeg_data_3d = np.reshape(eeg_data, (1, eeg_data.shape[0], eeg_data.shape[1]))
        # print(eeg_data_3d.shape)

        # separate the data in a convolutional manner
        this_x, this_y = ref.convolutional_split(eeg_data_3d, this_y, samples_to_jump_by=samples_to_jump_by,
                                                 trial_len=num_samples, num_channels=num_channels)

        # append to the main X and Y
        try:  
            X = np.vstack((X, this_x))
        except:
            X = this_x
        for label in this_y:
            Y.append(label)

    # shuffle the data
    combined_data = list(zip(X, Y))
    np.random.shuffle(combined_data)
    X_shuf, Y_shuf = zip(*combined_data)
    X = np.array(X_shuf)
    Y = np.array(Y_shuf)

    # after gathering together all of the data, call the function to get the best hyperparams
    # dropoutRate, kernels, kernLength, F1, D, F2, batch_size = get_best_hyperparams(X, Y)
    #
    # # process, filter, and epoch the data
    # model, x_test, y_test = get_trained_model(X, Y, dropoutRate=dropoutRate, kernels=kernels, kernLength=kernLength,
    #                           F1=F1, D=D, F2=F2, batch_size=batch_size) # call the function to train the model

    # getting best hyperparameters takes too long
    model, x_test, y_test = get_trained_model(X, Y, epochs=300)

    acc = get_model_acc(model, x_test, y_test) # call the function to get the acc

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
            model, X_test, Y_test = get_trained_model(X, Y, combination[0], combination[1], combination[2],
                                combination[3], combination[4], combination[5], combination[6])

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

def generate_prediction(board, model): # function to generate prediction given the trained model
    global kernels
    # THIS FUNCTION ASSUMES:
        # a session has already been activated
        # the session has been recording for at least a second and a half alrea
    # generate a prediction
    try:
        time.sleep(1.5)
        data = board.get_board_data(int(num_samples * 1.5))
        eeg_data = data[eeg_channels, :]
        eeg_3d_data = np.reshape(eeg_data, (1, eeg_data.shape[0], eeg_data.shape[1]))

        # split the data in a convolutional manner
        X, Y = ref.convolutional_split(eeg_3d_data, [1], samples_to_jump_by=samples_to_jump_by,
                                       trial_len=num_samples, num_channels=num_channels)

        # do not use Y, its a dummy var
        X_4d = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
        probs = model.predict(X_4d)

        # return the value which appears the most
        most_common_output = np.argmax(probs)
        prediction = label_decoding.get(most_common_output + 1)
        print(prediction)
        return prediction
    except:
        return 'none'


    
