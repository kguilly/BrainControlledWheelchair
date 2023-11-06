'''
This is the noise filtration test of the Brain Controlled Wheelchair
senior design project. This test will document and measure the ability of the
Brainflow library for filtering and denoising a signal.

To do this, we will take an intentionally noisy signal, plot it,
then use the brainflow library to filter out the noise, then plot the signal
again. A successful passing grade for this test will be a plotted signal
that is free from the spikes present in the noisy signal.
'''
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
    from tests import read_edf_files as ref

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
folder_dir = os.path.join(cur_dir, "test_data/kaleb/headset_data")
file_dir = os.path.join(folder_dir, 'forward.csv')
# read sig
read_data = DataFilter.read_file(file_dir)[:, 250:500]
data = read_data[eeg_channels, :]
df = pd.DataFrame(np.transpose(data))

# plot it
plt.figure()
df.plot(subplots=True, figsize=(10,20), title='Before Processing')
save_dir = os.path.join(cur_dir, "test_data/")
plt.savefig(os.path.join(save_dir, "0.2.3.1.3_before_proc.png"), dpi=300)

# get the accuracy and send out to file



# denoise the data
for channel in eeg_channels:
    # notch filter at 60Hz
    DataFilter.remove_environmental_noise(read_data[channel], BoardShim.get_sampling_rate(board_id),
                                          NoiseTypes.SIXTY.value)
# plot denoised data
df = pd.DataFrame(np.transpose(read_data))
plt.figure()
df.plot(subplots=True, figsize=(10,20), title='After Processing')
plt.savefig(os.path.join(save_dir, '0.2.3.1.3_after_proc.png'), dpi=300)

# get the accuracy and send out to file