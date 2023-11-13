import os
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

# TEST NOTE: FOR LINUX ALLOW PERMISSION TO THE PORT WITH COMMANDS:
# sudo usermod -aG dialout $USER
# sudo chmod a+rw /dev/ttyUSB0

def connect_to_headset(directory):
    try: 
        params = BrainFlowInputParams()
        params.serial_port = directory # for Linux, check com ports for windows
        board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)

        # board.prepare_session()
        
        return board, True
    except:
        return None, False

# this is the initial startup function that connects to the headset and does a health check on the 
# connectivity of the headset. 
def diagnostic_test(directory: str):
    # do some other things (eventually)
    board, connection_status = connect_to_headset(directory)

    return board, connection_status

# this is the start of the training
def start_session(board):
    try:
        board.prepare_session()
        board.start_session()
        return True
    except:
        return False
    
def end_session(board):
    board.stop_stream()
    board.release_session()

def gather_training_data(board, label: str, profile_path: str):
    data = board.get_board_data()
    
    # output the data to a labeled csv file. 
    save_dir = os.path.join(profile_path, 'headset_data', label + '.csv')
    DataFilter.write_file(data, save_dir, 'a')