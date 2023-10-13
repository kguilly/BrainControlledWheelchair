import time
import numpy as np
from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
import os

def handle_data(board_id, board_data):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')
    electrode_data = board_data[0]
    print(f"timestamp: {timestamp}")
    print("electrode data: ", electrode_data)

params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'


board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
board.prepare_session()

board.start_stream()

time.sleep(2)
data = board.get_board_data()
# data = np.array(data)
# print(data)
# data.shape = (32,234)
board.stop_stream()
board.release_session()
DataFilter.write_file(data, './test_data.csv', 'a')