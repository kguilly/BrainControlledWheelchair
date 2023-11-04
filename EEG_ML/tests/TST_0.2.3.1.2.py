'''
This is one of the subsets of the machine learning accuracy
test.

This test aims to see if splitting the incoming data affects
the accuracy of the model. Every tenth of a second, the previous
second and a half's worth of samples will be used to generate a 
prediction, rather than waiting a second and a half for each 
prediction. 

Implementing a type of moving average filter on the incoming 
data will be a great help in getting more predictions in the 
same amount of time. 

This test will be run on every subject. The results will be saved in 
./test_data/0.2.3.1.2.csv. The columns will be moving average vs
no moving average. The rows are each subject. An average accuracy
score will be assigned to each column


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

def get_model_acc_conv(X, Y, conv:bool):
    half = int(len(X) / 2)
    quarter = int(half / 2)
    three_fourths = half + quarter

    X_train = X[:half, :, :]
    X_validate = X[half: three_fourths, :, :]
    X_test = X[three_fourths:, :, :]

    y_train = Y[:half]
    y_validate = Y[half:three_fourths]
    y_test = Y[three_fourths:]

    if conv: 
        X_test, y_test = ref.convolutional_split(X_test, y_test, 16, 240, 64)

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
    checkpointer = ModelCheckpoint(filepath='/home/kaleb/tmp/checkpoint.h5', verbose=1,
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
# df = pd.DataFrame(columns=['Keep Format', 'Split Training Data'])
output_path = '/home/kaleb/Documents/GitHub/BrainControlledWheelchair/EEG_ML/tests/test_data/0.2.3.1.2.csv'
col_names = ['Keep Format', 'Split Training Data']
with open(output_path, 'w', newline='') as file:
    csv_writer = csv.DictWriter(file, fieldnames=col_names)
    csv_writer.writeheader()

# for each subject
for i in range(1,110):
    try:
        X, Y = ref.reader(passed_path='/home/kaleb/Documents/eeg_dataset/files/', patient_num=i)
        X, Y = ref.split_by_second(X, Y, 240, 64)
        acc_normal = get_model_acc_conv(X, Y, False)
        acc_split = get_model_acc_conv(X, Y, True)

        with open(output_path, 'a', newline='') as file:
            csv_writer = csv.DictWriter(file, fieldnames=col_names)
            data = {
                'Keep Format' : acc_normal,
                'Split Training Data' : acc_split
            }
            csv_writer.writerow(data)
    except:
        continue

df = pd.read_csv(output_path)
df.loc['Averaged Scores'] = [df['Keep Format'].mean(), df['Split Training Data'].mean()]
df.to_csv(output_path, index=True)
