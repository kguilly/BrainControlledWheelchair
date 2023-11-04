'''
This is one version of the machine learning accuracy test
for the brain controlled wheelchair

This test compares a different combination of hyper parameters
for each run.

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

import pandas as pd
import numpy as np
import os.path
import read_edf_files as ref
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils as np_utils
from EEG_ML.EEGModels import EEGNet

hyperparameter_map = {
    'dropoutRate' : [0.4, 0.5, 0.6],
    'kernels' : [1, 2, 3],
    'kernLength' : [16, 32, 64],
    'F1' : [4, 8, 16],
    'D' : [1, 2, 3],
    'F2' : [8, 16, 32],
    'batch_size' : [8, 16, 32]
    'epochs' : [15, 30, 60]
}




