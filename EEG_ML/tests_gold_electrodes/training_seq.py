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
import os
import csv

from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter

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
curr_file_path = os.path.dirname(os.path.abspath(__file__))
dir_above = os.path.dirname(curr_file_path)
with add_to_path(dir_above):
    from EEGModels import EEGNet
    import read_edf_files as ref


class TrainingSeq():
    def __init__(self):
        self.info = 'file to connect to the headset and get the data'
        # self.user_path = '/home/kaleb/Documents/GitHub/BrainControlledWheelchair/EEG_ML/tests/' \
        #                  'test_data/kaleb/'
        self.user_path = os.path.join(curr_file_path, 'test_data', 'kaleb_test0')
        self.storage_path = os.path.join(self.user_path, 'headset_data')
        # self.storage_path = '/home/kaleb/Desktop/HEADSET_DATA/'

        ##########################
        ## headset init params
        self.board = None
        self.timeout = 50
        self.serial_port = '/dev/ttyUSB0'
        self.board_id = BoardIds.CYTON_BOARD
        ##########################

        label_mapping = {
            1: "rest",
            2: "forward", # "Squeeze Both Fists",
            3: "backward", # "Squeeze Both Feet",
            4: "left", # "Squeeze Left Hand",
            5: "right", # "Squeeze Right Hand",
        }

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
        params.serial_port = self.serial_port

        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()

    def training_sequence(self):
        # prompt the user, record the data, send out to a file

        ## FORWARD
        while True:
            self.countdown(3, "FORWARD", "Go")
            self.board.start_stream()
            self.countdown(5, "", "Stop")
            dataf = self.board.get_board_data()
            self.board.stop_stream()
            choice = input("Send to File? [y/n]: ")
            if str(choice).lower() == 'y':
                self.send_data_to_file(dataf, "forward")
                time.sleep(2)
                break
            else:
                print("OK, do it again\n\n")
                time.sleep(2)

        ## BACKWARD
        while True:
            self.countdown(3, "BACKWARD", "Go")
            self.board.start_stream()
            self.countdown(5, "", "Stop")
            dataf = self.board.get_board_data()
            self.board.stop_stream()
            choice = input("Send to File? [y/n]: ")
            if str(choice).lower() == 'y':
                self.send_data_to_file(dataf, "backward")
                time.sleep(2)
                break
            else:
                print("OK, do it again\n\n")
                time.sleep(2)

        ## LEFT
        while True:
            self.countdown(3, "LEFT", "Go")
            self.board.start_stream()
            self.countdown(5, "", "Stop")
            dataf = self.board.get_board_data()
            self.board.stop_stream()
            choice = input("Send to File? [y/n]: ")
            if str(choice).lower() == 'y':
                self.send_data_to_file(dataf, "left")
                time.sleep(2)
                break
            else:
                print("OK, do it again\n\n")
                time.sleep(2)

        
        ## RIGHT
        while True:
            self.countdown(3, "RIGHT", "Go")
            self.board.start_stream()
            self.countdown(5, "", "Stop")
            dataf = self.board.get_board_data()
            self.board.stop_stream()
            choice = input("Send to File? [y/n]: ")
            if str(choice).lower() == 'y':
                self.send_data_to_file(dataf, "right")
                time.sleep(2)
                break
            else:
                print("OK, do it again\n\n")
                time.sleep(2)


        ## REST
        while True:
            self.countdown(3, "REST", "Go")
            self.board.start_stream()
            self.countdown(10, "", "Stop")
            dataf = self.board.get_board_data()
            self.board.stop_stream()
            choice = input("Send to File? [y/n]: ")
            if str(choice).lower() == 'y':
                self.send_data_to_file(dataf, "rest")
                time.sleep(2)
                break
            else:
                print("OK, do it again\n\n")
                time.sleep(2)



    def train_the_model(self):
        # TODO: TEST THIS FUNCTIONN
        # load the data into a file
        X = []
        Y = []
        for file in os.listdir(self.storage_path):
            # load the file into numpy array
            # (trials, channels, samples)
            this_x = np.loadtxt(file)
            this_x = np.reshape(this_x, (1, this_x[0], this_x[1]))
            # TODO: only load the channels which are electrodes

            this_y = []
            # based on name of file, select the label
            if file == 'rest.csv':
                this_y.append(1)
            elif file == 'forward.csv':
                this_y.append(2)
            elif file == 'backward.csv':
                this_y.append(3)
            elif file == 'left.csv':
                this_y.append(4)
            elif file == 'right.csv':
                this_y.append(5)

            # separate by second
            ref.split_by_second(X, Y, 120)

            # append to the main x and y
            X = np.vstack((X, this_x))

            for label in this_y:
                Y.append(label)

        ## Process, filter, and epoch the data
        # init arrays to train/validate/test. Make split 50/25/25
        half = int(len(X) / 2)
        quarter = int(half / 2)
        three_fourths = half + quarter

        X_train = X[:half, :, :]
        X_validate = X[half : three_fourths, :, :]
        X_test = X[three_fourths:, :, :]

        y_train = Y[:half]
        y_validate = Y[half:three_fourths]
        y_test = Y[three_fourths:]

        # convert labels to one-hot encoding
        y_train = np_utils.to_categorical(y_train-1)
        y_validate = np_utils.to_categorical(y_validate-1)
        y_test = np_utils.to_categorical(y_test-1)

        # convert data to NHWC (trials, channels, samples, kernels) format
        kernels = 1
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], kernels)
        X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], X_validate.shape[2], kernels)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], kernels)

        print('x_train shape: ', X_train.shape, '\ny_train shape: ', y_train.shape)
        ################################################################
        ## Call EEGNet
        num_labels = 5
        model = EEGNet(nb_classes=num_labels, Chans=X_train.shape[1], Samples=X_train.shape[2],
                    dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
                        dropoutType= 'Dropout')
        
        # compile the model and set the optimizers
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                    metrics = ['accuracy'])

        # count number of parameters in the model
        numParams    = model.count_params()    

        # set a valid path for your system to record model checkpoints
        checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                    save_best_only=True)
        
        # the weights all to be 1
        class_weights = {0:1, 1:1, 2:1, 3:1, 4:1}   
        
        # fittedModel = 
        model.fit(X_train, y_train, batch_size = 16, epochs = 300, 
                        verbose = 2, validation_data=(X_validate, y_validate),
                        callbacks=[checkpointer], class_weight = class_weights)
        
        # load optimal weights
        model.load_weights('/tmp/checkpoint.h5')

        probs       = model.predict(X_test)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == y_test.argmax(axis=-1))
        print("Classification accuracy: %f " % (acc))

        # TODO: save the model to a file
        
        

    def test_the_model(self):
        # activate the session with the board 
        # for every 100 samples of data, read the data and send through the trained model
        
        # TODO: Load the model
        choice = ''
        print('####################\nTesting Results\n\tpress \'X\' to exit')
        while choice.upper() != 'X':
            
            
            choice = input()
            
            

        
        
    
    def countdown(self, from_num, start_message, end_message):
        print(start_message)
        for i in range(from_num, 0, -1):
            print(i)
            time.sleep(1)
        print(end_message)


    def send_data_to_file(self, data, label):
        filename = os.path.join(self.storage_path, label + '.csv')
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            
        # if not os.path.exists(filename): # if the file does not exist, create it 
        #     with open(filename, 'w', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow([])
        
        DataFilter.write_file(data, filename, 'a') 


t = TrainingSeq()
t.main()