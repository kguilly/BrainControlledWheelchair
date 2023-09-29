import numpy as np
import os
import pyedflib

# EEGNet specific imports
from EEG_ML.EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import read_edf_files
import innvestigate

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

# X is of shape (trials, channels, values)

'''
frequency_of_values 
for each trial
    for each channel 
        these_values = X[trial][channel]
        frequency_of_values = fft(these_values) 
        append frequency to bigger array 
'''
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
print('x_train shape: ', X_train.shape, '\ny_train shape: ', y_train.shape)
################################################################
## Call EEGNet

model = EEGNet(nb_classes=num_labels, Chans=X_train.shape[1], Samples=X_train.shape[2],
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
                 dropoutType= 'Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])

'''
# call the analyzer
analyzer = innvestigate.create_analyzer('lrp.z', model)

# choose a sample
samp = X_test[0]

# analyze
analysis = analyzer.analyze(samp)
channel_relevance = np.sum(analysis, axis=(1,2))

# select the top 16 channels with the highest relevance
top_channels_indices = np.argsort(channel_relevance)[-16:]

print(top_channels_indices)
'''

analyzer = innvestigate.create_analyzer('input', model)
analysis = analyzer.analyze(X)

print(analysis.shape)

import matplotlib.pyplot as plt 
plt.imshow(analysis[1], cmap='hot', interpolation='nearest')
plt.axis('off')
plt.show()