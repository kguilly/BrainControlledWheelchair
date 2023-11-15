'''
Signal Denoising and Filtering:
This test is similar ot 0.2.3.1.3 except instead of filtering the data, we
are doing methods of denoising
'''

import sys
from contextlib import contextmanager
import os
import seaborn as sns
import matplotlib.pyplot as plt
from brainflow.data_filter import DataFilter, AggOperations, WaveletTypes, NoiseEstimationLevelTypes, \
    WaveletExtensionTypes, ThresholdTypes, WaveletDenoisingTypes


@contextmanager
def add_to_path(directory):
    sys.path.append(directory)
    try:
        yield
    finally:
        sys.path.remove(directory)


curr_file_path = os.path.dirname(os.path.abspath(__file__))
dir_above = os.path.dirname(curr_file_path)
with add_to_path(dir_above):
    from EEGModels import EEGNet
    import read_edf_files as ref

import pandas as pd
import numpy as np

from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes
from sklearn.metrics import confusion_matrix

def get_confusion_matrix(model, x, y):
    # send through the model
    probs = model.predict(x)
    # change from one hot encoded to int
    y_int = y.argmax(axis=1)
    preds = probs.argmax(axis=-1)
    # return cm
    return confusion_matrix(y_int, preds)

def get_trained_model(X, Y, dropoutRate=0.5, kernels=1, kernLength=32, F1=8, D=2, F2=16, batch_size=16):
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

    return model, X_test, y_test


def get_model_acc(trained_model, X_test, Y_test):
    probs = trained_model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))

    return acc


eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
board_id = BoardIds.CYTON_DAISY_BOARD.value

# read the data into x and y arrs
data_dir = os.path.join(curr_file_path, "test_data", 'kaleb_bald', 'headset_data')
X = []
Y = []
for file in os.listdir(data_dir):
    # load the data from the file
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
    samples_to_jump_by = 10
    num_samples = 180
    num_channels = 16
    # this_x, this_y = ref.convolutional_split(eeg_data_3d, this_y, samples_to_jump_by, num_samples, num_channels)
    this_x, this_y = ref.split_by_second(eeg_data_3d, this_y, 120, 16)

    # append to the main X and Y
    try:
        X = np.vstack((X, this_x))
    except:
        X = this_x
    for label in this_y:
        Y.append(label)

# take the data, run the noisy and denoised data through the model 10 times and
# append to a dataframe
df = pd.DataFrame(columns=['Noisy Acc', 'Average', 'Median', 'Wavelet Based'])
for i in range(0, 10):
    # combine and shuffle the data
    combined_data = list(zip(X, Y))
    np.random.shuffle(combined_data)
    X_shuf, Y_shuf = zip(*combined_data)
    X_shuf = np.array(X_shuf)
    Y_shuf = np.array(Y_shuf)
    model, x, y = get_trained_model(X_shuf, Y_shuf)

    noisy_acc = get_model_acc(model, x, y)
    '''
    # if i == 0:
    #     cm = get_confusion_matrix(model, x, y)
    #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Rest", "Forward", "Backward", "Left", "Right"],
    #                 yticklabels=["Rest", "Forward", "Backward", "Left", "Right"])
    #     cm_name =  "0.2.3.1.4.noisy_confusion.png"
    #     sns.set(font_scale=1.2)
    #     plt.xlabel("Predicted")
    #     plt.ylabel("True")
    #     plt.title("Confusion Matrix")
    #     output_path = os.path.join(curr_file_path, 'test_data', cm_name)
    #     plt.savefig(output_path, dpi=300)
    '''
    # denoise the data
    X_mean_shuf = X_shuf
    for trial in range(0, X_shuf.shape[0]):
        for channel in range(0, X_shuf.shape[1]):
            DataFilter.perform_rolling_filter(X_mean_shuf[trial, channel], 3, AggOperations.MEAN.value)

    # get the filtered acc
    model, x, y = get_trained_model(X_mean_shuf, Y_shuf)
    mean_acc = get_model_acc(model, x, y)
    '''
    if i == 0:
        cm = get_confusion_matrix(model, x, y)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2"],
                    yticklabels=["Class 0", "Class 1", "Class 2"])
        cm_name =  "0.2.3.1.4.denoised_confusion.png"
        plt.xlabel("Predicted", fontweight='bold')
        plt.ylabel("True", fontweight='bold')
        plt.title("Confusion Matrix")
        output_path = os.path.join(curr_file_path, 'test_data', cm_name)
        plt.savefig(output_path, dpi=300)
    '''
    X_median_shuf = X_shuf
    for trial in range(0, X_shuf.shape[0]):
        for channel in range(0, X_shuf.shape[1]):
            DataFilter.perform_rolling_filter(X_median_shuf[trial, channel], 3, AggOperations.MEDIAN.value)

    # get the filtered acc
    model, x, y = get_trained_model(X_median_shuf, Y_shuf)
    median_acc = get_model_acc(model, x, y)

    X_wavelet_shuf = X_shuf
    for trial in range(0, X_shuf.shape[0]):
        for channel in range(0, X_shuf.shape[1]):
            DataFilter.perform_wavelet_denoising(X_wavelet_shuf[trial, channel], WaveletTypes.BIOR1_1, 3,
                                                 WaveletDenoisingTypes.SURESHRINK, ThresholdTypes.HARD,
                                                 WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)

    # get the filtered acc
    model, x, y = get_trained_model(X_wavelet_shuf, Y_shuf)
    wavelet_acc = get_model_acc(model, x, y)

    df.loc[len(df)] = [noisy_acc, mean_acc, median_acc, wavelet_acc]

# after all the shits have run, output to a csv
# get the averages
avg_unfilt = df['Noisy Acc'].mean()
avg_filt = df['Average'].mean()
median_acc = df['Median'].mean()
wavelet_med = df['Wavelet Based'].mean()
df.loc['avg score'] = [avg_unfilt, avg_filt, median_acc, wavelet_med]
output_path = os.path.join(curr_file_path, 'test_data', '0.2.3.1.4.csv')
df.to_csv(output_path)