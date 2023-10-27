'''
THIS TEST DIFFERS FROM 0.2.2.5: 
    - I will be slicing the data up by tenths of a second
    - I will be taking the previous second and a half worth (180 samples)
        of data to make a prediction 


This is test 0.2.2.5 of the BRAIN-CONTROLLED WHEELCHAIR senior design II
project. This test takes the trained model from 0.2.2.4 and incoming data
from the headset and generates predictions once a second from incoming data
'''
import os
import time
import numpy as np
from tensorflow.keras import utils as np_utils
from keras.models import load_model

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

count = 0
label_decoding = {
        0: "Rest",
        1: "Squeeze Both Fists",
        2: "Squeeze Both Feet",
        3: "Squeeze Left Hand",
        4: "Squeeze Right Hand",
    }

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
    if eeg_data.shape[1] < 120:
        eeg_data = np.pad(eeg_data, ((0,0), (0,120-eeg_data.shape[1])),
                          mode='constant', constant_values=0)
    elif eeg_data.shape[1] > 120:
        eeg_data = eeg_data[:, 1:121]
    eeg_3d_data = eeg_data.reshape(1, eeg_data.shape[0], 120, 1)

    # pass through model
    probs = model.predict(eeg_3d_data)
    # print prediction
    index = np.argmax(probs)
    prediction = label_decoding.get(index)
    print(f'prob dist: {probs} .... pred: {prediction}')

    # if 10 seconds have passed, end the program
    if count > 15:
        break
    count += 1

board.release_session()
