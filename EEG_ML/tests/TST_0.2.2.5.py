'''
This is test 0.2.2.5 of the BRAIN-CONTROLLED WHEELCHAIR senior design II
project. This test takes the trained model from 0.2.2.4 and incoming data
from the headset and generates predictions once a second from incoming data
'''
import os
import time

from tensorflow.keras import utils as np_utils
from keras.models import load_model

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

count = 0

# load the model into the program
cur_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(cur_dir, 'test_data/0.2.2.4_model.h5')
model = load_model(model_dir)

# load and connect to the board
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0' # for linux, check com ports for windows
board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
board.prepare_session()

while 1:
    # record a second worth of dat
    board.start_stream()
    time.sleep(1)
    data = board.get_board_data()
    board.stop_stream()

    # grab the data and format
    eeg_data = data[eeg_channels, :]
    # reformat to (1, 16, samples(120))

    # pass through model

    # print prediction

    # if 10 seconds have passed, end the program
    if count > 10:
        break
    count += 1

board.release_session()
