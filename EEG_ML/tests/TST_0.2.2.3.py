# this is test 0.2.2.3 of the BrainControlled Wheelchair
# this test takes 0.2.2.2 a step further. It gathers the data after
# five seconds and rather than printing the data out, the
# data is saved to a file. The file is then loaded, and the tail end is printed
# Five more seconds of headset data is written to the end of the file.
# The file is again loaded into a dataframe, and the tail is printed out.
# The tail is printed twice to ensure that new data is in the file. The length of
# the dataframe is also printed out to each load of the file to ensure that
# old data is not overwritten.

import time
import os

import numpy as np
import pandas as pd

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0' # for Linux, check com ports for windows
board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)


# get 5 seconds worth of data from the board
board.prepare_session()
board.start_stream()
time.sleep(5)
data = board.get_board_data()
board.stop_stream()

# write out to file
cur_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(cur_dir, "test_data/0.2.2.3.csv")
DataFilter.write_file(data, save_dir, 'w')

# read from file
read_data = DataFilter.read_file(save_dir)
eeg_data = read_data[eeg_channels, :]
print(f'Final shape of data: {eeg_data.shape} ')
print('Tail of df: ', eeg_data.tail(10))
print('##############################################################')

# get 5 more seconds worth of data
board.start_stream()
time.sleep(5)
data = board.get_board_data()
board.stop_stream()
board.release_session()

# write out to same file
DataFilter.write_file(data, save_dir, 'a')

# read from file, make sure that length and tail is different
read_data = DataFilter.read_file(save_dir)
eeg_data = read_data[eeg_channels, :]
print(f'Final shape of data: {eeg_data.shape} ')
print('Tail of df: ', eeg_data.tail(10))
print('##############################################################')

