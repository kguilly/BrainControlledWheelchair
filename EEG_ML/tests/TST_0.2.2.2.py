# this is test 0.2.2.2 of the BrainControlled Wheelchair
# this test takes 0.2.2.1 a step further and connects to the headset
# but also streams incoming data and prints that data out. This test
# ensures that data is actually being received from the headset in the
# manner that we expect it to be received in

import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0' # for Linux, check com ports for windows
board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)

board.prepare_session()

board.start_stream()
time.sleep(5)

data = board.get_board_data()
board.stop_stream()
board.release_session()
print(f'DATA Shape: {data.shape}\n\n#################################', data)

