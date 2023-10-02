'''

This file will:
- connect to the headset
- prompt the user to imaging moving in a direction
- record data
- format and store data
- reopen the data and use to train the model
'''
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
class TrainingSeq():
    def __init__(self):
        self.info = 'file to connect to the headset and get the data'
        self.storage_path = '/path/'
        self.board = None

        ##########################
        ## headset init params
        self.timeout = None
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
        # self.headset_connect()

        # prompt the user:
            # training sequence
            # train the model
            # test the model

        choice = ''
        while choice != 'X':
            print("\n\n###########################")
            print("Tell me what to do: ")
            print("\t- Go through training sequence: 1")
            print("\t- Train the model: 2")
            print("\t- Test the model: 3")
            print("\t- exit: X")
            choice = input("Pick bitch: ")

            if choice == '1':
                print("\n***********************")
                print("TRAINING SEQUENCE BITCH: GET READY")
                time.sleep(1)
                self.training_sequence()

            elif choice == '2':
                print("\n***********************")
                print("USING THAT DATA TO TRAIN DA MODEL")
                time.sleep(1)
                self.train_the_model()

            elif choice == '3':
                print("\n***********************")
                print("GO HEAD LIL BOY TEST IT")
                time.sleep(1)
                self.test_the_model()
            else:
                print("Wrong one try again")



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
        self.countdown(3, "Imagine moving forward in: ", "Go")

        self.countdown(5, "", "Stop")
        time.sleep(2)

        self.countdown(3, "BACKWARDS", "Go")

        self.countdown(5, "", "STOP")
        time.sleep(2)

        self.countdown(3, 'LEFT', "GO")

        self.countdown(5, '', 'STOP')
        time.sleep(2)

        self.countdown(3, 'RIGHT', 'GO')

        self.countdown(5, '', 'STOP')
        time.sleep(1)

    def countdown(self, from_num, start_message, end_message):
        print(start_message)
        for i in range(from_num, 0, -1):
            print(i)
            time.sleep(1)
        print(end_message)

t = TrainingSeq()
t.main()