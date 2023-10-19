# this is test 0.2.2.1 of the BrainControlled Wheelchair
# this test consists solely of connecting to the Ultracortex Mark IV and
# maintaining that connection for 5 seconds. This test will be validated by
# performing the test on multiple machines which span across multiple operating systems

import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0' # for Linux, check com ports for windows
board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)

board.prepare_session()

time.sleep(5)

board.release_session()

