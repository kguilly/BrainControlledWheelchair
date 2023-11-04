'''
This is one version of the machine learning accuracy test
for hte brain controlled wheelchair

This test compares three versions of modifying the training data
before feeding it into the model:
- keeping the format as read
- splitting the trials by the second
- splitting the trials in a convolutional manner

Every subject from the open source dataset was used, and their accuracy
scores for the model were kept in a pandas df. the scores for each method
were then averaged

The dataset used:
https://physionet.org/content/eegmmidb/1.0.0/

from the PhysioToolkit Software:
https://archive.physionet.org/physiotools/
'''
import pandas as pd
import numpy as np
import os.path
import read_edf_files as ref
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils as np_utils
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
with add_to_path('/home/kaleb/Documents/GitHub/BrainControlledWheelchair/EEG_ML'):
    from EEGModels import EEGNet

os.environ["CUDA_VISIBLE_DEVICES"] = ''
def get_model_acc(X, Y):
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

    kernels = 1
    # convert data to NHWC (trials, channels, samples, kernels) format
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], kernels)
    X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], X_validate.shape[2], kernels)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], kernels)

    print('x_train shape: ', X_train.shape, '\ny_train shape: ', y_train.shape)
    ################################################################
    ## Call EEGNet

    model = EEGNet(nb_classes=5, Chans=X_train.shape[1], Samples=X_train.shape[2],
                   dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
                   dropoutType='Dropout')

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # count number of parameters in the model
    numParams = model.count_params()
    # set a valid path for your system to record model checkpoints
    chkpt_filepath = '/home/kaleb/tmp/checkpoint.h5'
    checkpointer = ModelCheckpoint(filepath=chkpt_filepath, verbose=1,
                                   save_best_only=True)
    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    model.fit(X_train, y_train, batch_size=16, epochs=30,
                            verbose=2, validation_data=(X_validate, y_validate),
                            callbacks=[checkpointer], class_weight=class_weights)
    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == y_test.argmax(axis=-1))
    return acc

import csv
# df = pd.DataFrame(columns=['Keep Format', 'Split by Second', 'Convolutional Split'])
output_path = '/home/kaleb/Documents/GitHub/BrainControlledWheelchair/EEG_ML/tests/test_data/0.2.3.1.1.csv'
col_names = ['Keep Format', 'Split by Second', 'Convolutional Split']
with open(output_path, 'w', newline='') as file:
    csv_writer = csv.DictWriter(file, fieldnames=col_names)
    csv_writer.writeheader()

# for each subject
for i in range(1, 110):

    try:
        X, Y = ref.reader(passed_path='/home/kaleb/Documents/eeg_dataset/files/', patient_num=i)
        acc_normal = get_model_acc(X,Y)

        X_sec, Y_sec = ref.split_by_second(X, Y, 160, 64)
        acc_sec = get_model_acc(X_sec, Y_sec)

        X_conv, Y_conv = ref.convolutional_split(X, Y, 16, 240, 64)
        acc_conv = get_model_acc(X_conv, Y_conv)

        with open(output_path, 'a', newline='') as file:
            csv_writer = csv.DictWriter(file, fieldnames=col_names)
            data = {
                'Keep Format' : acc_normal,
                'Split by Second' : acc_sec,
                'Convolutional Split' : acc_conv
            }
            csv_writer.writerow(data)
    except:
        continue
df = pd.read_csv(output_path)
df.loc['Averaged Scores'] = [df['Keep Format'].mean(), df['Split by Second'].mean(),
                             df['Convolutional Split'].mean()]
df.to_csv(output_path, index=True)



