'''
This is one version of the machine learning accuracy test
for the brain controlled wheelchair

This test compares a different combination of hyper parameters
for each run.

This is a brute force method. Due to time constraints, a random three
subjects were chosen: 7, 13, 84

Every subject from the open source dataset was used, and their accuracy
scores for the model were kept in a pandas df. the scores for each method
were then averaged.

There were X total combinations. The entirety of the results
were stores in X.csv, with the columns being the combination,
and the rows being each user. Each combination has an average 
score for each user

The dataset used:
https://physionet.org/content/eegmmidb/1.0.0/

from the PhysioToolkit Software:
https://archive.physionet.org/physiotools/
'''
import csv

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

def get_model_acc(X, Y, dropoutRate, kernels, kernLength,
                  F1, D, F2, batch_size):
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
    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == y_test.argmax(axis=-1))
    return acc

hyperparameter_map = {
    'dropoutRate' : [0.4, 0.5, 0.6],
    'kernels' : [1, 2, 3],
    'kernLength' : [16, 32, 64],
    'F1' : [4, 8, 16],
    'D' : [1, 2, 3],
    'F2' : [8, 16, 32],
    'batch_size' : [8, 16, 32]
}

output_path = '/home/kaleb/Documents/GitHub/BrainControlledWheelchair/EEG_ML/tests/test_data/0.2.3.1.0.csv'
col_names = ['Subject', 'HyperParams', 'Acc Score']
with open(output_path, 'w', newline='') as file:
    csv_writer = csv.DictWriter(file, fieldnames=col_names)
    csv_writer.writeheader()

import itertools
combinations = list(itertools.product(*(hyperparameter_map[param] for param in hyperparameter_map)))
random_subjects = [7, 13, 84]

for subject in random_subjects:
    for combination in combinations:
        try:
            X, Y = ref.reader(passed_path='/home/kaleb/Documents/eeg_dataset/files/', patient_num=subject)
            acc = get_model_acc(X, Y, combination[0], combination[1], combination[2],
                                combination[3], combination[4], combination[5], combination[6])

            # write that acc out to file
            with open(output_path, 'a', newline='') as file:
                csv_writer = csv.DictWriter(file, fieldnames=col_names)
                data = {
                    'Subject' : subject,
                    'HyperParams' : combination,
                    'Acc Score' : acc
                }
                csv_writer.writerow(data)
        except:
            continue

# after all have finished, find the max vals for each subject and append to end
df = pd.read_csv(output_path)
max_acc_rows = df.loc[df.groupby('Subject')['Acc Score'].idxmax()]

new_rows = []
for index, row in max_acc_rows.iterrows():
    new_row = {
        'Subject' : f'Max for Subject {row["Subject"]}',
        'HyperParams' : row['HyperParams'],
        'Acc Score' : row['Acc Score']
    }
    new_rows.append(new_row)
new_row_df = pd.DataFrame(new_rows)
df = pd.concat([df, new_row_df], ignore_index=True)
df.to_csv(output_path, index=False)




