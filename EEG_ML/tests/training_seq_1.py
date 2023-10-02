'''
https://brainflow.readthedocs.io/en/stable/Examples.html
This file will:
- connect to the headset
- prompt the user to imaging moving in a direction
- record data
- format and store data
- reopen the data and use to train the model
'''
import time
import pandas as pd
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter

class TrainingSeq():
    def __init__(self):
        self.info = 'file to connect to the headset and get the data'
        self.storage_path = './training_data/'
        self.file_count = 0 # tracks how many times we go through training seq

        ##########################
        ## headset init params
        self.board = None
        self.timeout = 50
        self.ip_port = None
        self.ip_protocol = None
        self.serial_port = None
        self.mac_address = None
        self.other_info = None
        self.serial_number = None
        self.ip_address = None
        self.file = None
        self.master_board = None
        self.board_id = None
        ##########################

    def main(self):
        print("connecting to headset...")
        self.headset_connect()

        # prompt the user:
        choice = ''
        while choice != 'X':
            print("\n\n###########################")
            print("Tell me what to do: ")
            print("\t- Go through training sequence: 1")
            print("\t- Train the model: 2")
            print("\t- Test the model: 3")
            print("\t- exit: X")
            choice = input("Pick one: ")

            if choice == '1':
                print("\n***********************")
                print("TRAINING SEQUENCE: GET READY")
                time.sleep(1)
                self.training_sequence()

            elif choice == '2':
                print("\n***********************")
                print("USING THAT DATA TO TRAIN DA MODEL")
                time.sleep(1)
                self.train_the_model()

            elif choice == '3':
                print("\n***********************")
                print("Using recorded data to test")
                time.sleep(1)
                self.test_the_model()

            else:
                print("Wrong one try again")

        # free the board after the session
        self.board.release_all_sessions()

    def headset_connect(self):

        params = BrainFlowInputParams()
        params.ip_port = self.ip_port
        params.serial_port = self.serial_port
        params.mac_address = self.mac_address
        params.other_info = self.other_info
        params.serial_number = self.serial_number
        params.ip_address = self.ip_address
        params.ip_protocol = self.ip_protocol
        params.timeout = self.timeout
        params.file = self.file
        params.master_board = self.master_board

        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()

    def training_sequence(self):
        # prompt the user, record the data, send out to a file

        ## FORWARD
        self.countdown(3, "Imagine moving forward in: ", "Go")
        self.board.start_stream()
        self.countdown(5, "", "Stop")
        dataf = self.board.get_board_data()
        self.board.stop_stream()
        self.send_data_to_file(dataf, "forward")
        time.sleep(2)


        ## BACKWARD
        self.countdown(3, "BACKWARDS", "Go")
        self.board.start_stream()
        self.countdown(5, "", "STOP")
        datab = self.board.get_board_data()
        self.board.stop_stream()
        self.send_data_to_file(datab, "backward")
        time.sleep(2)


        ## LEFT
        self.countdown(3, 'LEFT', "GO")
        self.board.start_stream()
        self.countdown(5, '', 'STOP')
        datal = self.board.get_board_data()
        self.board.stop_stream()
        self.send_data_to_file(datal, "left")
        time.sleep(2)

        ## RIGHT
        self.countdown(3, 'RIGHT', 'GO')
        self.board.start_stream()
        self.countdown(5, '', 'STOP')
        datar = self.board.get_board_data()
        self.board.stop_stream()
        self.send_data_to_file(datar, "right")
        time.sleep(1)

        self.file_count += 1

    def train_the_model(self):
        pass

    def test_the_model(self):
        # activate the session with the board 
        # for every 100 samples of data, read the data and send through the trained model
        choice = ''
        while choice.upper() != 'X':
            
            
            choice = input()
            
            

        
        
    
    def countdown(self, from_num, start_message, end_message):
        print(start_message)
        for i in range(from_num, 0, -1):
            print(i)
            time.sleep(1)
        print(end_message)


    def send_data_to_file(self, data, label): 
        filename = self.storage_path + label + '.csv'
        DataFilter.write_file(data, filename, 'a') # TODO: MAKE SURE APPENDING WORKS AS EXPECTED 


t = TrainingSeq()
t.main()