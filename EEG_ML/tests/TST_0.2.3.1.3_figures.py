'''
This is the noise filtration test of the Brain Controlled Wheelchair
senior design project. This test will document and measure the ability of the
Brainflow library for filtering and denoising a signal.

To do this, we will take an intentionally noisy signal, plot it,
then use the brainflow library to filter out the noise, then plot the signal
again.

This test determines if further filtering methods will be used to 
feed into the machine learning model. The question of if this method
of filtering provides a more accurate classification score will be answered. 

We will be using a notch filter of 8-30Hz that Gerhort pointed me to
Link: @incollection{AFRAKHTEH202025,
title = {Chapter 2 - Applying an efficient evolutionary algorithm for EEG signal feature selection and classification in decision-based systems},
editor = {Amr Mohamed},
booktitle = {Energy Efficiency of Medical Devices and Healthcare Applications},
publisher = {Academic Press},
pages = {25-52},
year = {2020},
isbn = {978-0-12-819045-6},
doi = {https://doi.org/10.1016/B978-0-12-819045-6.00002-9},
url = {https://www.sciencedirect.com/science/article/pii/B9780128190456000029},
author = {Sajjad Afrakhteh and Mohammad Reza Mosavi},
}

The dataset that will be used is one that was recorded with the reconfigured
headset with Kaleb's bald ass head
'''
# append the path to eeget
import sys
from contextlib import contextmanager
import os
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

import os
import matplotlib
import pandas as pd
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes

# grab the signal
board_id = BoardIds.CYTON_DAISY_BOARD.value
eeg_channels = BoardShim.get_eeg_channels(board_id)
cur_dir = os.path.dirname(os.path.abspath(__file__))
folder_dir = os.path.join(cur_dir, "test_data", 'kaleb_bald', 'headset_data')
file_dir = os.path.join(folder_dir, 'forward.csv')
# read sig
read_data = DataFilter.read_file(file_dir)[:, 1000:1250]
data = read_data[eeg_channels, :]
df = pd.DataFrame(np.transpose(data))
# df = pd.DataFrame(data)

# plot it
plt.figure()
df.plot(subplots=True, figsize=(15,25), title='Before Processing')
save_dir = os.path.join(cur_dir, "test_data/")
plt.savefig(os.path.join(save_dir, "0.2.3.1.3_before_proc.png"), dpi=300)

# get the accuracy and send out to file



# denoise the data
for channel in eeg_channels:
    # third order butterworth bandpass filter with no ripple
    DataFilter.perform_bandpass(read_data[channel], BoardShim.get_sampling_rate(board_id), 
                                3, 60, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.remove_environmental_noise(read_data[channel], BoardShim.get_sampling_rate(board_id),
                                           NoiseTypes.SIXTY.value)
# plot denoised data
df = pd.DataFrame(np.transpose(read_data))
plt.figure()
df.plot(subplots=True, figsize=(15,25), title='After Processing', legend=False)
plt.savefig(os.path.join(save_dir, '0.2.3.1.3_after_proc.png'), dpi=300)

# get the accuracy and send out to file