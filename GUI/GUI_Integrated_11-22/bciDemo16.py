import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import ImageTk, Image
import pygame
import time
import keyboard
import os
import threading

# Kaleb's imports: 
import Headset_files.initialization as init
import Headset_files.headset_ml as ml
from keras.models import load_model
import wheelchair_files.wheelchair as wc

# PYTHON 3.9.2 REQUIRED TO OPERATE CORRECTLY!


############################################################################################
# Global variables

profSelected = ""
prof1Status = False
prof1Name = ""
prof2Status = False
prof2Name = ""
prof3Status = False
prof3Name = ""
trainingFlag = False
headsetMode = False
keyboardMode = True
switch_state = True # when switching from headset to keyboard and vice versa
levelStatus = True
safetyStatus = True
prediction = '' # the headset function will call and change this
prediction_semaphore = threading.Semaphore(value=1) # protect when the variable changes state
flip_warning = False
collision_warning = False


# TODO: implement user choice for config of com port / directory
# this is the directory that the heaadset is connected to, needs to be configured on each run
headset_directory = None    # for windows, a com port, for linux, usually '/dev/ttyUSB0'
pi_directory = None         # for windows, a com port, for linux, usually '/dev/ttyUSB0'

# TODO: ADD A TRAINING SEQUENCE FOR RESTING

###### TO CHECK IF WINDOWS OR LINUX :
import platform

system_platform = platform.system()


print(system_platform)

curr_file_path = os.path.dirname(os.path.abspath(__file__))

# TODO: implement two different choices of using: either use the keyboard or headset
#keyboard_user_mode = True


############################################################################################


def predicted_direction(model):
    global prediction, headset, profSelected, curr_file_path

    output_prediction = ml.generate_prediction(headset, model)

    # let the semaphore be free while the function is running
    with prediction_semaphore:
        prediction = output_prediction

def main_directory_checker():
    global curr_file_path
    masterProfilePath = os.path.join(curr_file_path, 'Profiles' )
    if not os.path.exists(masterProfilePath):
        os.makedirs(masterProfilePath)


def stop_joystick():
    global stop_fill, oval_fill_slow, oval_fill_brisk, oval_fill_left, oval_fill_right, oval_fill_reverse, oval_fill
    stop_fill = '#cc0202'
    oval_fill_slow = oval_fill
    oval_fill_brisk = oval_fill
    oval_fill_left = oval_fill
    oval_fill_right = oval_fill
    oval_fill_reverse = oval_fill


def slow_joystick():
    global stop_fill, oval_fill_slow, oval_fill_brisk, oval_fill_left, oval_fill_right, oval_fill_reverse, oval_fill
    stop_fill = oval_fill
    oval_fill_slow = '#57E964'
    oval_fill_brisk = oval_fill
    oval_fill_left = oval_fill
    oval_fill_right = oval_fill
    oval_fill_reverse = oval_fill


def reverse_joystick():
    global stop_fill, oval_fill_slow, oval_fill_brisk, oval_fill_left, oval_fill_right, oval_fill_reverse, oval_fill
    stop_fill = oval_fill
    oval_fill_slow = oval_fill
    oval_fill_brisk = oval_fill
    oval_fill_left = oval_fill
    oval_fill_right = oval_fill
    oval_fill_reverse = '#eb34e8'


def left_joystick():
    global stop_fill, oval_fill_slow, oval_fill_brisk, oval_fill_left, oval_fill_right, oval_fill_reverse, oval_fill
    stop_fill = oval_fill
    oval_fill_slow = oval_fill
    oval_fill_brisk = oval_fill
    oval_fill_left = '#34deeb'
    oval_fill_right = oval_fill
    oval_fill_reverse = oval_fill


def right_joystick():
    global stop_fill, oval_fill_slow, oval_fill_brisk, oval_fill_left, oval_fill_right, oval_fill_reverse, oval_fill
    stop_fill = oval_fill
    oval_fill_slow = oval_fill
    oval_fill_brisk = oval_fill
    oval_fill_left = oval_fill
    oval_fill_right = '#34deeb'
    oval_fill_reverse = oval_fill


i=0 # Used for testing frame counts of directional window (will be deleted in final version)
################################################################################################################
# "Directional Window" window updater
################################################################################################################
#headsetMode = True # this variable controls when to record data from headset
def directional_window():
    global keyboard_user_mode, headset, curr_file_path, start_headset_session_flag, \
        i, switch_state, prediction, flip_warning, collision_warning

    profile_path = os.path.join(curr_file_path, "Profiles", "Profile" + profSelected)
    model_path = os.path.join(profile_path, 'trained_model.h5')
    model = load_model(model_path)

    if flip_warning:
        directionalLevelStatusLabel.config(text="Unlevel", fg="#FF0000")
        directionalLevelStatusLabel.place(x=880, y=360)
    if collision_warning:
        directionalSafetyStatusLabel.config(text="Danger", fg="#FF0000")
        directionalSafetyStatusLabel.place(x=910, y=460)
    if not flip_warning:
        directionalLevelStatusLabel.config(text='Level', fg='#41FF00')
        directionalLevelStatusLabel.place(x=950, y=360)
    if not collision_warning:
        directionalSafetyStatusLabel.config(text='Safe', fg="#41FF00")
        directionalSafetyStatusLabel.place(x=970, y=460)


    # if we want to use the keyboard mode (rather than the headset outputs)
    if keyboardMode:

        if not switch_state:
            # kill the headset session
            try:
                init.end_session(headset)
            except:
                pass
            switch_state = True

        if keyboard.is_pressed("w"):
            slow_joystick()
            flip_warning, collision_warning = wc.receive_and_transmit_keyboard_input('w')
        elif keyboard.is_pressed("s"):
            reverse_joystick()
            flip_warning, collision_warning = wc.receive_and_transmit_keyboard_input('s')
        elif keyboard.is_pressed("a"):
            left_joystick()
            flip_warning, collision_warning = wc.receive_and_transmit_keyboard_input('a')
        elif keyboard.is_pressed("d"):
            right_joystick()
            flip_warning, collision_warning =  wc.receive_and_transmit_keyboard_input('d')
        else:
            stop_joystick()
        #---------------------------------------------------------------------------------------------------------
    else: # use the output from the headset

        # if this is the first time, start the session and wait 
        if switch_state:
            switch_state = False
            # start the session and wait a second and a half to start gathering data
            # thread this and display the stop joystick
            headset_init_thread = threading.Thread(target=init.threaded_init_loop,
                                                   args=(1, headset))
            headset_init_thread.start()
            stop_joystick()
            headset_init_thread.join()

        headset_prediction_thread = threading.Thread(target=predicted_direction, args=(model,))
        headset_prediction_thread.start()

        with prediction_semaphore:
            if prediction == 'forward':
                slow_joystick()
            elif prediction == 'backward':
                reverse_joystick()
            elif prediction == 'left':
                left_joystick()
            elif prediction == 'right':
                right_joystick()
            else: # the user is resting
                stop_joystick()

        headset_prediction_thread.join()


        

    #window.bind("<Left>", slow_joystick())

    stop_test_frame = canvas.create_polygon(
        ((canvas_width/2)-stop_size), ((canvas_height/2)+stop_size),
        ((canvas_width/2)-stop_size), ((canvas_height/2)-stop_size),
        ((canvas_width/2)+stop_size), ((canvas_height/2)-stop_size),
        ((canvas_width/2)+stop_size), ((canvas_height/2)+stop_size),
        outline='')

    stop_sign = canvas.create_polygon(
        ((canvas_width/2)-stop_size), ((canvas_height/2)+(stop_size*stop_edge_size)),       # first point
        ((canvas_width/2)-stop_size), ((canvas_height/2)-(stop_size*stop_edge_size)),       # second point
        ((canvas_width/2)-(stop_size*stop_edge_size)), ((canvas_height/2)-stop_size),       # third point
        ((canvas_width/2)+(stop_size*stop_edge_size)), ((canvas_height/2)-stop_size),       # fourth point
        ((canvas_width/2)+stop_size), ((canvas_height/2)-(stop_size*stop_edge_size)),       # fifth point
        ((canvas_width/2)+stop_size), ((canvas_height/2)+(stop_size*stop_edge_size)),       # sixth point
        ((canvas_width/2)+(stop_size*stop_edge_size)), ((canvas_height/2)+stop_size),       # seventh point
        ((canvas_width/2)-(stop_size*stop_edge_size)), ((canvas_height/2)+stop_size),       # eighth point
        outline='',
        fill=stop_fill)
        #activefill=stop_activefill)

    canvas.create_text((canvas_width/2),        # Stop indicator text creation
        (canvas_height/2),
        text=stop_text, 
        fill=stop_text_color,
        font=stop_text_font)


    #---------------------------------------------------------------------------------------------------------
    # Forward indicator

    forward_oval_slow = canvas.create_oval(
        ((canvas_width/2)-oval_size),(((canvas_height/2)-oval_spacing)-(oval_size)*(2/3)),
        ((canvas_width/2)+oval_size),(((canvas_height/2)-oval_spacing)+(oval_size)*(2/3)),
        fill=oval_fill_slow)
        #activefill=oval_activefill_slow)

    canvas.create_text(((canvas_width/2)),      # Forward indicator text creation
        (((canvas_height/2)-oval_spacing)),
        text=oval_text_slow, 
        fill=oval_text_color,
        font=oval_text_font)

    #---------------------------------------------------------------------------------------------------------
    # Reverse indicator

    reverse_oval = canvas.create_oval(
        ((canvas_width/2)-oval_size_left_right_reverse),(((canvas_height/2)+oval_spacing)-(oval_size)*(2/3)),
        ((canvas_width/2)+oval_size_left_right_reverse),(((canvas_height/2)+oval_spacing)+(oval_size)*(2/3)),
        fill=oval_fill_reverse)
        #activefill=oval_activefill_reverse)

    canvas.create_text(((canvas_width/2)),      # Reverse indicator text creation
        (((canvas_height/2)+(oval_spacing))),
        text=oval_text_reverse, 
        fill=oval_text_color,
        font=oval_text_font)

    #---------------------------------------------------------------------------------------------------------
    # Left indicator

    left_oval = canvas.create_oval(
        (((canvas_width/2)-oval_spacing)-oval_size),((canvas_height/2)-(oval_size)*(2/3)),
        (((canvas_width/2)-oval_spacing)+oval_size),((canvas_height/2)+(oval_size)*(2/3)),
        fill=oval_fill_left)
        #activefill=oval_activefill_left)

    canvas.create_text((((canvas_width/2)-oval_spacing)),        # Left indicator text creation
        (canvas_height/2),
        text=oval_text_left, 
        fill=oval_text_color,
        font=oval_text_font)

    #---------------------------------------------------------------------------------------------------------
    # Right indicator

    right_oval = canvas.create_oval(
        (((canvas_width/2)+oval_spacing)-oval_size),((canvas_height/2)-(oval_size)*(2/3)),
        (((canvas_width/2)+oval_spacing)+oval_size),((canvas_height/2)+(oval_size)*(2/3)),
        fill=oval_fill_right)
        #activefill=oval_activefill_right)

    canvas.create_text((((canvas_width/2)+oval_spacing)),        # Right indicator text creation
        (canvas_height/2),
        text=oval_text_right, 
        fill=oval_text_color,
        font=oval_text_font)
    #window.update_idletasks()
    #window.update()


def directional_window_loop():
    while (True):
        directional_window()
        directionalWindow.update_idletasks()
        directionalWindow.update()

def directional_window_headset_button():
    global headsetMode, keyboardMode
    keyboardMode = False
    headsetMode = True
    directionalModeKeyboardButton.config(state='active')
    directionalModeHeadsetButton.config(state='disabled')
    directionalModeStatusLabel.config(text="Mode: Headset")
    directionalModeStatusLabel.place(x=1150,y=35)



def directional_window_keyboard_button():
    global headsetMode, keyboardMode
    keyboardMode = True
    headsetMode = False
    directionalModeKeyboardButton.config(state='disabled')
    directionalModeHeadsetButton.config(state='active')
    directionalModeStatusLabel.config(text="Mode: Keyboard")
    directionalModeStatusLabel.place(x=1150,y=35)





################################################################################################################
# Window updaters
################################################################################################################

def update_system_init_window():
    systemInitWindow.update_idletasks()     # Updates "systemInitWindow" window even if not called upon to prevent errors 
    systemInitWindow.update()               # Updates "systemInitWindow" window even if not called upon to prevent errors 


def update_start_window():
    startWindow.update_idletasks()     # Updates "startWindow" window even if not called upon to prevent errors 
    startWindow.update()               # Updates "startWindow" window even if not called upon to prevent errors 
    

def update_new_profile_window():
    newProfileWindow.update_idletasks()     # Updates "newProfileWindow" window even if not called upon to prevent errors 
    newProfileWindow.update()               # Updates "newProfileWindow" window even if not called upon to prevent errors 



def update_load_profile_window():
    loadProfileWindow.update_idletasks()     # Updates "loadProfileWindow" window even if not called upon to prevent errors 
    loadProfileWindow.update()               # Updates "loadProfileWindow" window even if not called upon to prevent errors 



def update_create_profile_window():
    createProfileWindow.update_idletasks()     # Updates "createProfileWindow" window even if not called upon to prevent errors 
    createProfileWindow.update()               # Updates "createProfileWindow" window even if not called upon to prevent errors 



def update_training_welcome_window():
    trainingWelcomeWindow.update_idletasks()     # Updates "trainingWelcomeWindow" window even if not called upon to prevent errors 
    trainingWelcomeWindow.update()               # Updates "trainingWelcomeWindow" window even if not called upon to prevent errors 



def update_training_forward_window():
    trainingForwardWindow.update_idletasks()     # Updates "trainingForwardWindow" window even if not called upon to prevent errors 
    trainingForwardWindow.update()               # Updates "trainingForwardWindow" window even if not called upon to prevent errors 



def update_training_left_window():
    trainingLeftWindow.update_idletasks()     # Updates "trainingLeftWindow" window even if not called upon to prevent errors 
    trainingLeftWindow.update()               # Updates "trainingLeftWindow" window even if not called upon to prevent errors 



def update_training_right_window():
    trainingRightWindow.update_idletasks()     # Updates "trainingRightWindow" window even if not called upon to prevent errors 
    trainingRightWindow.update()               # Updates "trainingRightWindow" window even if not called upon to prevent errors 



def update_training_backward_window():
    trainingBackwardWindow.update_idletasks()     # Updates "trainingBackwardWindow" window even if not called upon to prevent errors 
    trainingBackwardWindow.update()               # Updates "trainingBackwardWindow" window even if not called upon to prevent errors 



def update_training_rest_window():
    trainingRestWindow.update_idletasks()     # Updates "trainingRestWindow" window even if not called upon to prevent errors 
    trainingRestWindow.update()               # Updates "trainingRestWindow" window even if not called upon to prevent errors 



def update_training_complete_window():
    trainingCompleteWindow.update_idletasks()     # Updates "trainingCompleteWindow" window even if not called upon to prevent errors 
    trainingCompleteWindow.update()               # Updates "trainingCompleteWindow" window even if not called upon to prevent errors 



# Guides user from the "Start" window to next screen
def start_selection(choice):
    global headset_directory, headset, pi_directory

    # while you doing this, connect to the headset
    headset, connection_status = init.connect_to_headset(headset_directory)
    pi_connection_status = wc.init_connection_to_pi(pi_directory)
    if not connection_status:
        print("Error: could not connect to headset")
        # exit(1)
    if not pi_connection_status:
        print("Error: could not connect to pi")
        exit(1)

    if (choice == 1):
        print("New profile option selected!")
        occupied_profile_check("new")
    elif (choice == 2):
        print("Load profile option selected!")
        occupied_profile_check("load")
    elif (choice == 3):
        print("Exiting application!")
        startWindow.destroy()                       # Exits the application



# Checks which profile slots are occupied
def occupied_profile_check(strChoice):
    global prof1Status, prof1Name, prof2Status, prof2Name, prof3Status, prof3Name
    occupiedProfiles = 0
    if (os.path.isfile('./Profiles/Profile1/profile1Config.txt')):
        prof1Status = True
        file = open('./Profiles/Profile1/profile1Config.txt', "r")
        fileName = file.readlines()
        prof1Name = fileName[0]
        prof1Name.strip()
        print(prof1Name)
        file.close()

    if (os.path.isfile('./Profiles/Profile2/profile2Config.txt')):
        prof2Status = True
        file = open('./Profiles/Profile2/profile2Config.txt', "r")
        prof2Name = file.readline()
        file.close()

    if (os.path.isfile('./Profiles/Profile3/profile3Config.txt')):
        prof3Status = True
        file = open('./Profiles/Profile3/profile3Config.txt', "r")
        prof3Name = file.readline()
        file.close()

    if (strChoice == "new"):                                    # **UNDER CONSTRUCTION** If ALL profiles are occupied AND "New Profile" was selected
        #print("profile occupied!")
        newProfileWindow.deiconify()
        startWindow.withdraw()
        new_profile_window(prof1Status, prof2Status, prof3Status)


    elif (strChoice == "load"):                                 # **UNDER CONSTRUCTION** If NO profiles are occupied AND "Load Profile" was selected
        print("profiles occupied!")
        loadProfileWindow.deiconify()
        startWindow.withdraw()
        load_profile_window()
    


def new_profile_window(fileOneOccupied, fileTwoOccupied, fileThreeOccupied):
    if (fileOneOccupied == True):
        print("profile 1 occupied!")
        newProfileOneButton.config(state='disabled')
        newProfileOneButton.config(text='Profile 1 **OCCUPIED**')
        newProfileOneButton.config(pady=35)

    if (fileTwoOccupied == True):
        print("profile 2 occupied!")
        newProfileTwoButton.config(state='disabled')
        newProfileTwoButton.config(text='Profile 2 **OCCUPIED**')
        newProfileTwoButton.config(pady=35)

    if (fileThreeOccupied == True):
        print("profile 3 occupied!")
        newProfileThreeButton.config(state='disabled')
        newProfileThreeButton.config(text='Profile 3 **OCCUPIED**')
        newProfileThreeButton.config(pady=35)



def load_profile_window():
    global prof1Status, prof1Name, prof2Status, prof2Name, prof3Status, prof3Name
    print("gooby")

    if (prof1Status != False):
        loadProfileOneButton.config(text="Profile 1 " + prof1Name)
        loadProfileOneButton.config(state='active')
    else:
        loadProfileOneButton.config(state='disabled')

    if (prof2Status != False):
        loadProfileTwoButton.config(text="Profile 2 " + prof2Name)
        loadProfileTwoButton.config(state='active')
    else:    
        loadProfileTwoButton.config(state='disabled')

    if (prof3Status != False):
        loadProfileThreeButton.config(text="Profile 3 " + prof3Name)
        loadProfileThreeButton.config(state='active')
    else:
        loadProfileThreeButton.config(state='disabled')




def profile_creator(profileNum):
    global profSelected

    if (profileNum == 1):
        print("Creating Profile 1")
        profSelected = "1"
        createProfileLabel.config(text="Profile 1 Creation")
        createProfileWindow.deiconify()
    elif (profileNum == 2):
        print("Creating Profile 2")
        profSelected = "2"
        createProfileLabel.config(text="Profile 2 Creation")
        createProfileWindow.deiconify()
    elif (profileNum == 3):
        print("Creating Profile 3")
        profSelected = "3"
        createProfileLabel.config(text="Profile 3 Creation")
        createProfileWindow.deiconify()



def new_profile_back_button():
    startWindow.deiconify()
    newProfileWindow.withdraw()
    createProfileWindow.withdraw()



def profile_loading_back_button():
    startWindow.deiconify()
    loadProfileWindow.withdraw()



def profile_creation_back_button():
    newProfileWindow.deiconify()
    createProfileWindow.withdraw()
    createProfileEntry.delete(0,tk.END)
    createProfileErrorLabel.config(text="")


def system_os_check():
    # Setup is Windows configured by default so no additional code for that.
    if (system_platform == "Linux"):
        systemInitLabel.config(text='Hello Linux user, please input ttyUSB port number for BCI headset and Pi below.')
        systemInitHeadsetComLabel.config(text="Headset: /dev/ttyUSB")
        systemInitHeadsetComLabel.place(x=47,y=118)
        systemInitHeadsetEntry.place(x=390,y=125)
        systemInitPiComLabel.config(text="Pi: /dev/ttyUSB")
        systemInitPiComLabel.place(x=137,y=188)
        systemInitPiEntry.place(x=390,y=185)
        


def system_init_continue_button():
    global headset_directory, pi_directory
    sysInitSpaceTest = " " in (systemInitHeadsetEntry.get())
    sysInitSpaceTest2 = " " in (systemInitPiEntry.get())
    
    if (len((systemInitHeadsetEntry.get())) > 0 and len((systemInitHeadsetEntry.get())) <= 3 and str(sysInitSpaceTest) == "False" 
        and (systemInitHeadsetEntry.get()).isdigit() and len((systemInitPiEntry.get())) > 0 and len((systemInitPiEntry.get())) <= 3 and str(sysInitSpaceTest) == "False" 
        and (systemInitPiEntry.get()).isdigit()):
        if (system_platform == "Windows"):
            headset_directory = "COM" + systemInitHeadsetEntry.get()
            pi_directory = "COM" + systemInitPiEntry.get()
        elif (system_platform == "Linux"):
            headset_directory = "/dev/ttyUSB" + systemInitHeadsetEntry.get()
            pi_directory = "/dev/ttyrfcomm" + systemInitPiEntry.get()
            # Add Linux COM code equivalent here if any is needed (FOR KALEB)

        print(headset_directory)
        print(pi_directory)
        systemInitWindow.withdraw()
        #print(str((systemInitHeadsetEntry.get()).isdigit())) #== "True"
        systemInitHeadsetEntry.delete(0,tk.END)
        systemInitPiEntry.delete(0,tk.END)
        systemInitErrorLabel.config(text="")
        startWindow.deiconify()
    else:
        if (len((systemInitHeadsetEntry.get())) <= 0 or len((systemInitPiEntry.get())) <= 0):
            systemInitErrorLabel.config(text="ERROR: Must contain at least one number!")
        elif (str((systemInitHeadsetEntry.get()).isdigit()) == "False" or str((systemInitPiEntry.get()).isdigit()) == "False"):
            systemInitErrorLabel.config(text="ERROR: Must only contain numbers!")
        elif (len((systemInitHeadsetEntry.get())) > 3 or len((systemInitPiEntry.get())) > 3):
            systemInitErrorLabel.config(text="ERROR: Too many numbers!")
        elif (str(sysInitSpaceTest) == "True" or str(sysInitSpaceTest2) == "True"):
            systemInitErrorLabel.config(text="ERROR: Must not contain spaces!")
        else:
            systemInitErrorLabel.config(text="ERROR please try again!")



def profile_creation_create_button():
    global profSelected
    spaceTest = " " in (createProfileEntry.get())
    
    if (len((createProfileEntry.get())) > 0 and len((createProfileEntry.get())) <= 10 and str(spaceTest) == "False"):
        startWindow.deiconify()
        newProfileWindow.withdraw()
        createProfileWindow.withdraw()
        if not os.path.exists("./Profiles/Profile"+profSelected):
            prof_dir = os.path.join("Profiles", "Profile" + profSelected)
            os.makedirs(prof_dir)              # creates individual profile folders if they do not exist.
            headset_data_dir = os.path.join(prof_dir, "headset_data")
            os.makedirs(headset_data_dir)
        file = open("./Profiles/Profile"+profSelected+"/"+"profile"+profSelected+"Config"+".txt", "w")
        file.write(createProfileEntry.get())
        file.close()
        createProfileEntry.delete(0,tk.END)
        createProfileErrorLabel.config(text="")
        trainingWelcomeWindow.deiconify()
        createProfileWindow.withdraw()
        startWindow.withdraw()
    else:
        if (len((createProfileEntry.get())) <= 0):
            createProfileErrorLabel.config(text="ERROR: Must contain at least one character!")
        elif (len((createProfileEntry.get())) > 10):
            createProfileErrorLabel.config(text="ERROR: Too many characters!")
        elif (str(spaceTest) == "True"):
            createProfileErrorLabel.config(text="ERROR: Must not contain spaces!")
        else:
            createProfileErrorLabel.config(text="ERROR please try again!")
    




def training_welcome_start_button():
    trainingForwardWindow.deiconify()
    trainingWelcomeWindow.withdraw()



def training_csv_populator(label, profile_path):
    global trainingFlag, headset

    headset_status = init.start_session(headset)
    if headset_status == False:
        print("Could not start session during training")
        exit(1)

    # wait until training finishes to gather the data
    while (trainingFlag != True):
        pass

    # after training, gather the data and release the session
    init.end_session(headset)
    init.gather_training_data(headset, label, profile_path)




def training_ready_button(direction):
    global trainingFlag, profSelected, curr_file_path
    labelBeginChoice = ""
    labelSessionChoice = ""
    directionButtonChoice = ""
    updaterChoice = ""
    if (direction == "Forward"):
        directionButtonChoice = trainingForwardReadyButton
        labelBeginChoice = trainingForwardBeginCountdownLabel
        labelSessionChoice = trainingForwardSessionCountdownLabel
        updaterChoice = update_training_forward_window
    elif (direction == "Left"):
        directionButtonChoice = trainingLeftReadyButton
        labelBeginChoice = trainingLeftBeginCountdownLabel
        labelSessionChoice = trainingLeftSessionCountdownLabel
        updaterChoice = update_training_left_window
    elif (direction == "Right"):
        directionButtonChoice = trainingRightReadyButton
        labelBeginChoice = trainingRightBeginCountdownLabel
        labelSessionChoice = trainingRightSessionCountdownLabel
        updaterChoice = update_training_right_window
    elif (direction == "Backward"):
        directionButtonChoice = trainingBackwardReadyButton
        labelBeginChoice = trainingBackwardBeginCountdownLabel
        labelSessionChoice = trainingBackwardSessionCountdownLabel
        updaterChoice = update_training_backward_window

    elif (direction == "Rest"):
        directionButtonChoice = trainingRestReadyButton
        labelBeginChoice = trainingRestBeginCountdownLabel
        labelSessionChoice = trainingRestSessionCountdownLabel
        updaterChoice = update_training_rest_window

    directionButtonChoice.config(state='disabled')
    labelBeginChoice.config(text='Countdown to begin: 3 sec')
    labelBeginChoice.config(fg="#41FF00")
    updaterChoice()
    print("3")
    time.sleep(1) # Sleep for one second
    labelBeginChoice.config(text='Countdown to begin: 2 sec')
    updaterChoice()
    print("2")
    time.sleep(1) # Sleep for one second
    labelBeginChoice.config(text='Countdown to begin: 1 sec')
    updaterChoice()
    print("1")
    time.sleep(1) # Sleep for one second
    if (labelBeginChoice == trainingRestBeginCountdownLabel):
        labelBeginChoice.config(text="DON'T THINK!")
    else:
        labelBeginChoice.config(text='THINK!')
    updaterChoice()
    print("0")
    
    # profile path 
    profile_dir = os.path.join(curr_file_path, 'Profiles', 'Profile' + profSelected)
    # display other countdown here
    # call training_csv_populator here to start populating csv file.
    # need to pass the current profile directory and the label
    thread1 = threading.Thread(target=training_csv_populator, args=(direction, profile_dir))
    thread1.start()
    #csvPopulation = asyncio.create_task(training_csv_populator())
    #trainingFlag = True
    labelSessionChoice.config(text='Session ends in: 5 sec')
    labelSessionChoice.config(fg="#41FF00")
    updaterChoice()
    time.sleep(1) # Sleep for one second
    labelSessionChoice.config(text='Session ends in: 4 sec')
    updaterChoice()
    time.sleep(1) # Sleep for one second
    labelSessionChoice.config(text='Session ends in: 3 sec')
    updaterChoice()
    time.sleep(1) # Sleep for one second
    labelSessionChoice.config(text='Session ends in: 2 sec')
    updaterChoice()
    time.sleep(1) # Sleep for one second
    labelSessionChoice.config(text='Session ends in: 1 sec')
    updaterChoice()
    time.sleep(1) # Sleep for one second
    labelSessionChoice.config(text=direction+' training complete!')
    updaterChoice()
    trainingFlag = True
    thread1.join()
    trainingFlag = False
    
    updaterChoice()
    directionButtonChoice.config(command=lambda: training_next_button(direction,directionButtonChoice))
    directionButtonChoice.config(text='Next')
    directionButtonChoice.config(state='active')



def training_next_button(direction, directionButtonChoice):
    global trainingFlag, profSelected
    if (direction == "Forward"):
        trainingForwardWindow.withdraw()
        trainingLeftWindow.deiconify()
        directionButtonChoice.config(command=lambda: training_ready_button("Forward"))
        directionButtonChoice.config(text='Ready')
    elif (direction == "Left"):
        trainingLeftWindow.withdraw()
        trainingRightWindow.deiconify()
        directionButtonChoice.config(command=lambda: training_ready_button("Left"))
        directionButtonChoice.config(text='Ready')
    elif (direction == "Right"):
        trainingRightWindow.withdraw()
        trainingBackwardWindow.deiconify()
        directionButtonChoice.config(command=lambda: training_ready_button("Right"))
        directionButtonChoice.config(text='Ready')
    elif (direction == "Backward"):
        trainingBackwardWindow.withdraw()
        trainingRestWindow.deiconify()                                                        
        directionButtonChoice.config(command=lambda: training_ready_button("Backward"))
        directionButtonChoice.config(text='Ready')
    elif (direction == "Rest"):
        trainingRestWindow.withdraw()
        trainingCompleteWindow.deiconify()                                                        
        directionButtonChoice.config(command=lambda: training_ready_button("Rest"))
        directionButtonChoice.config(text='Ready')

def training_complete_finish_button():
    # TODO: make a waiting screen for the model to train
    # after the training is complete
    # TODO: make y_train one hot encoded, SHUFFLE THE ARRAYS BEFORE TRAINING
    profile_path = os.path.join(curr_file_path, "Profiles", "Profile" + profSelected)
    accuracy = ml.train_the_model(profile_path)  # the model is saved to the user's directory       (TEMPORARILY DISABLED FOR TESTING BY SAMUEL) **********************************************************************************************

    # TODO: display to the user the accuaracy of their trained model
    # TODO: TRAIN MORE? OR USE THE SYSTEM?
    print(f'Your trained model\'s accuracy is {accuracy}')                                        #  (TEMPORARILY DISABLED FOR TESTING BY SAMUEL) **********************************************************************************************

    directionalWindow.deiconify()
    trainingCompleteWindow.withdraw()
    #thread2Directional = threading.Thread(target=directional_window_loop, args=())
    #thread2Directional.start()
    directional_window_loop()
    

def load_profile_button(pickedProfile):
    global profSelected
    if (pickedProfile == 1):
        profSelected = "1"
    elif (pickedProfile == 2):
        profSelected = "2"
    elif (pickedProfile == 3):
        profSelected = "3"

    directionalWindow.deiconify()
    loadProfileWindow.withdraw()
    #thread2Directional = threading.Thread(target=directional_window_loop, args=())
    #thread2Directional.start()
    directional_window_loop()



################################################################################################################
# "Directional Window" window creation and population
################################################################################################################   

# "Directional Window" window settings
directionalWindow = tk.Tk()
directionalWindow.title('Brain-controlled Car')
directionalWindow.geometry('1920x1080')

# changing "Directional Window" icon and background color
icon = tk.PhotoImage(file=os.path.join(curr_file_path, 'extra', 'wheelchairIcon2.png'))
directionalWindow.iconphoto(True, icon)
directionalWindow.config(background='black')
#directionalWindow.attributes('-fullscreen', True)

# Creates label to show mode status
directionalModeStatusLabel = tk.Label(directionalWindow, 
    text='Mode: ', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',50,'bold'),
    #relief=tk.RAISED,
    bd=6,
    padx=0,
    pady=0)
directionalModeStatusLabel.place(x=1150,y=35)

# Creates button for headset mode
directionalModeHeadsetButton = tk.Button(directionalWindow, 
    text='Headset', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',35,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=12,
    padx=7,
    pady=7,
    command=lambda: directional_window_headset_button())
directionalModeHeadsetButton.place(x=1110,y=150)

# Creates button for keyboard mode
directionalModeKeyboardButton = tk.Button(directionalWindow, 
    text='Keyboard', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',35,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=11,
    padx=5,
    pady=5,
    command=lambda: directional_window_keyboard_button())
directionalModeKeyboardButton.place(x=1500,y=150)

# Creates label to show level status
directionalLevelStatusLabel = tk.Label(directionalWindow, 
    text='Level', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',35,'bold'),
    relief=tk.RIDGE,
    bd=5,
    padx=5,
    pady=5)
directionalLevelStatusLabel.place(x=950,y=360)

# Creates label to show safety status
directionalSafetyStatusLabel = tk.Label(directionalWindow, 
    text='Safe', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',35,'bold'),
    relief=tk.RIDGE,
    bd=5,
    padx=5,
    pady=5)
directionalSafetyStatusLabel.place(x=970,y=460)

if (keyboardMode == True):
    directionalModeKeyboardButton.config(state='disabled')
    directionalModeStatusLabel.config(text='Mode: Keyboard')
    directionalModeStatusLabel.place(x=1150,y=35)




################################################################################################################
# "System Initializer" window creation and population
################################################################################################################   

# "System Initializer" window settings
systemInitWindow = tk.Tk()
systemInitWindow.title('System Initialization')
systemInitWindow.geometry('600x490')

# Make "Create Profile" window not resizable
systemInitWindow.resizable(False, False)

# changing "Create Profile" background color
systemInitWindow.config(background='black')

# Creates label to show which system you are running
systemInitLabel = tk.Label(systemInitWindow, 
    text='Hello Windows user, please input COM port number for BCI headset and Pi below.', 
    fg="#41FF00", 
    bg="black", 
    wraplength=500,
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
systemInitLabel.pack(padx=20,pady=10)

# Creates invisible label to space out packing
systemInitSpacerLabel = tk.Label(systemInitWindow, 
    text=' ', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,)
systemInitSpacerLabel.pack(padx=0,pady=35)

# Creates label to show "Headset: COM"
systemInitHeadsetComLabel = tk.Label(systemInitWindow, 
    text='Headset: COM', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
systemInitHeadsetComLabel.place(x=115,y=118)

# Creates an entry box for inputting headset COM port         
systemInitHeadsetEntry = tk.Entry(systemInitWindow,
    font=('Terminal',19),
    relief=tk.RAISED,
    width=3,
    bd=8,
    fg="#41FF00", 
    bg="black",
    insertbackground="#41FF00")
systemInitHeadsetEntry.place(x=330,y=125)

# Creates label to show "Pi: COM"
systemInitPiComLabel = tk.Label(systemInitWindow, 
    text='Pi: COM', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=0,
    pady=0)
systemInitPiComLabel.place(x=205,y=188)

# Creates an entry box for inputting pi COM port 
systemInitPiEntry = tk.Entry(systemInitWindow,
    font=('Terminal',19),
    relief=tk.RAISED,
    width=3,
    bd=8,
    fg="#41FF00", 
    bg="black",
    insertbackground="#41FF00")
systemInitPiEntry.place(x=330,y=185)

# Creates label to show a note for character limit
systemInitNoteLabel = tk.Label(systemInitWindow, 
    text='NOTE: Entry must be a number (MAX three digits)', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',16),
    #relief=tk.RAISED,
    wraplength=300,
    bd=6,
    padx=10,
    pady=10)
systemInitNoteLabel.pack(padx=20,pady=0)

# Creates label to show a note for character limit
systemInitErrorLabel = tk.Label(systemInitWindow, 
    text='', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',17),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
systemInitErrorLabel.pack(padx=20,pady=0)

# Creates "Create" button to open "Start" window
systemInitCreateButton = tk.Button(systemInitWindow, 
    text='Continue', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',18),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=20,
    pady=20,
    command=lambda: system_init_continue_button())
systemInitCreateButton.pack(padx=20,pady=20)




################################################################################################################
# "Start" window creation and population
################################################################################################################   

# "Start" window settings
startWindow = tk.Tk()
startWindow.title('Start')
startWindow.geometry('540x800')

# Make "Start" window not resizable
startWindow.resizable(False, False)

# changing "Start" background color
startWindow.config(background='black')

newProfileButton = tk.Button(startWindow, 
    text='New Profile', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',40,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=6,
    padx=60,
    pady=60,
    command=lambda: start_selection(1))
newProfileButton.pack(padx=20,pady=20)

loadProfileButton = tk.Button(startWindow, 
    text='Load Profile', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',40,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=6,
    padx=60,
    pady=60,
    command=lambda: start_selection(2))
loadProfileButton.pack(padx=20,pady=20)

exitAppButton = tk.Button(startWindow, 
    text='Exit Application',
    wraplength=380, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',40,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=60,
    pady=60,
    command=lambda: start_selection(3))
exitAppButton.pack(padx=20,pady=20)




################################################################################################################
# "New Profile" window creation and population
################################################################################################################   

# "New Profile" window settings
newProfileWindow = tk.Tk()
newProfileWindow.title('Profile Creation')
newProfileWindow.geometry('540x900')

# Make "New Profile" window not resizable
newProfileWindow.resizable(False, False)

# changing "New Profile" background color
newProfileWindow.config(background='black')

# Creates new profile buttons 1-3
newProfileOneButton = tk.Button(newProfileWindow, 
    text='New Profile 1',
    wraplength=500, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',40,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=6,
    padx=60,
    pady=60,
    command=lambda: profile_creator(1))
newProfileOneButton.pack(padx=20,pady=20)

newProfileTwoButton = tk.Button(newProfileWindow, 
    text='New Profile 2',
    wraplength=500, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',40,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=6,
    padx=60,
    pady=60,
    command=lambda: profile_creator(2))
newProfileTwoButton.pack(padx=20,pady=20)

newProfileThreeButton = tk.Button(newProfileWindow, 
    text='New Profile 3', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',40,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=60,
    pady=60,
    command=lambda: profile_creator(3))
newProfileThreeButton.pack(padx=20,pady=20)

newProfileBackButton = tk.Button(newProfileWindow, 
    text='Back', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=20,
    pady=20,
    command=lambda: new_profile_back_button())
newProfileBackButton.pack(padx=20,pady=0)




################################################################################################################
# "Training Welcome" window creation and population
################################################################################################################   

# "Training Welcome" window settings
trainingWelcomeWindow = tk.Tk()
trainingWelcomeWindow.title('Welcome to training!')
trainingWelcomeWindow.geometry('540x500')

# Make "Training Welcome" window not resizable
trainingWelcomeWindow.resizable(False, False)

# changing "Training Welcome" background color
trainingWelcomeWindow.config(background='black')


# Creates label to show the large "Welcome to training!" sign
trainingWelcomeMainLabel = tk.Label(trainingWelcomeWindow, 
    text='Welcome to training!',
    wraplength=600, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',23),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingWelcomeMainLabel.pack(padx=20,pady=0)

# Creates label to show the general overview message
trainingWelcomeOverviewLabel = tk.Label(trainingWelcomeWindow, 
    text='To finish setting up your profile, we will need to train ' 
    'both the computer as well as your brain to associate with ' 
    'directions such as "Forward", "Left", "Right", and "Backwards" '
    'in order to properly control the wheelchair.',
    wraplength=500, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingWelcomeOverviewLabel.pack(padx=20,pady=0)

# Creates "Start" button to proceed to "Training Forward" window
trainingWelcomeStartButton = tk.Button(trainingWelcomeWindow, 
    text='Start', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=20,
    pady=20,
    command=lambda: training_welcome_start_button())
trainingWelcomeStartButton.pack(padx=0,pady=50)




################################################################################################################
# "Training Forward" window creation and population
################################################################################################################   

# "Training Forward" window settings
trainingForwardWindow = Toplevel()
trainingForwardWindow.title('Training Forward')
trainingForwardWindow.geometry('640x600')

# Make "Training Forward" window not resizable
trainingForwardWindow.resizable(False, False)

# changing "Training Forward" background color
trainingForwardWindow.config(background='black')

# Import left fist clipart
left_fist_path = os.path.join(curr_file_path, "extra", "leftfist.png")
leftFist = PhotoImage(file=left_fist_path)

# Import right fist clipart
right_fist_path = os.path.join(curr_file_path, "extra", "rightfist.png")
rightFist = PhotoImage(file=right_fist_path)

# Show left fist on screen
trainingForwardLeftFist = tk.Label(trainingForwardWindow,
    bg="black", 
    padx=10,
    pady=10,
    image=leftFist)
trainingForwardLeftFist.place(x=100,y=300,anchor='w')

# Show right fist on screen
trainingForwardRightFist = tk.Label(trainingForwardWindow,
    bg="black", 
    padx=10,
    pady=10,
    image=rightFist)
trainingForwardRightFist.place(x=550,y=300,anchor='e')

# Creates label to show the instructions
trainingForwardInstructionsLabel = tk.Label(trainingForwardWindow, 
    text='For this exercise, you are to clench both fists until '
    'the five second timer is up. Press the "Ready" button to '
    'start the session.',
    wraplength=500, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingForwardInstructionsLabel.pack(padx=20,pady=20)

# Creates "Ready" button to proceed to the countdown to train forward.
trainingForwardReadyButton = tk.Button(trainingForwardWindow, 
    text='Ready', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=10,
    pady=10,
    command=lambda: training_ready_button("Forward"))
trainingForwardReadyButton.pack(padx=0,pady=20,side=tk.BOTTOM)

# Creates label for the countdown to indicate remaining session duration
trainingForwardSessionCountdownLabel = tk.Label(trainingForwardWindow, 
    text='Session ends in: 5 sec',
    wraplength=500, 
    fg="black", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingForwardSessionCountdownLabel.pack(padx=0,pady=0,side=tk.BOTTOM)

# Creates label for the countdown to begin
trainingForwardBeginCountdownLabel = tk.Label(trainingForwardWindow, 
    text='Countdown to begin: 3 sec',
    wraplength=500, 
    fg="black", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=0)
trainingForwardBeginCountdownLabel.pack(padx=0,pady=0,side=tk.BOTTOM)




################################################################################################################
# "Training Left" window creation and population
################################################################################################################   

# "Training Left" window settings
trainingLeftWindow = Toplevel()
trainingLeftWindow.title('Training Left')
trainingLeftWindow.geometry('640x600')

# Make "Training Left" window not resizable
trainingLeftWindow.resizable(False, False)

# changing "Training Left" background color
trainingLeftWindow.config(background='black')

# Import left fist clipart
leftFist1 = PhotoImage(file=left_fist_path)

# Show left fist on screen
trainingLeftWindowFist = tk.Label(trainingLeftWindow,
    bg="black", 
    padx=10,
    pady=10,
    image=leftFist1)
trainingLeftWindowFist.place(x=230,y=300,anchor='w')

# Creates label to show the instructions
trainingLeftInstructionsLabel = tk.Label(trainingLeftWindow, 
    text='For this exercise, you are to clench ONLY your left fist until '
    'the five second timer is up. Press the "Ready" button to '
    'start the session.',
    wraplength=500, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingLeftInstructionsLabel.pack(padx=20,pady=20)

# Creates "Ready" button to proceed to the countdown to train Left.
trainingLeftReadyButton = tk.Button(trainingLeftWindow, 
    text='Ready', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=10,
    pady=10,
    command=lambda: training_ready_button("Left"))
trainingLeftReadyButton.pack(padx=0,pady=20,side=tk.BOTTOM)

# Creates label for the countdown to indicate remaining session duration
trainingLeftSessionCountdownLabel = tk.Label(trainingLeftWindow, 
    text='Session ends in: 5 sec',
    wraplength=500, 
    fg="black", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingLeftSessionCountdownLabel.pack(padx=0,pady=0,side=tk.BOTTOM)

# Creates label for the countdown to begin
trainingLeftBeginCountdownLabel = tk.Label(trainingLeftWindow, 
    text='Countdown to begin: 3 sec',
    wraplength=500, 
    fg="black", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=0)
trainingLeftBeginCountdownLabel.pack(padx=0,pady=0,side=tk.BOTTOM)




################################################################################################################
# "Training Right" window creation and population
################################################################################################################   

# "Training Right" window settings
trainingRightWindow = Toplevel()
trainingRightWindow.title('Training Right')
trainingRightWindow.geometry('640x600')

# Make "Training Right" window not resizable
trainingRightWindow.resizable(False, False)

# changing "Training Right" background color
trainingRightWindow.config(background='black')

# Import right fist clipart
rightFist1 = PhotoImage(file=right_fist_path)

# Show right fist on screen
trainingRightWindowFist = tk.Label(trainingRightWindow,
    bg="black", 
    padx=10,
    pady=10,
    image=rightFist1)
trainingRightWindowFist.place(x=230,y=300,anchor='w')

# Creates label to show the instructions
trainingRightInstructionsLabel = tk.Label(trainingRightWindow, 
    text='For this exercise, you are to clench ONLY your right fist until '
    'the five second timer is up. Press the "Ready" button to '
    'start the session.',
    wraplength=500, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingRightInstructionsLabel.pack(padx=20,pady=20)

# Creates "Ready" button to proceed to the countdown to train Right.
trainingRightReadyButton = tk.Button(trainingRightWindow, 
    text='Ready', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=10,
    pady=10,
    command=lambda: training_ready_button("Right"))
trainingRightReadyButton.pack(padx=0,pady=20,side=tk.BOTTOM)

# Creates label for the countdown to indicate remaining session duration
trainingRightSessionCountdownLabel = tk.Label(trainingRightWindow, 
    text='Session ends in: 5 sec',
    wraplength=500, 
    fg="black", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingRightSessionCountdownLabel.pack(padx=0,pady=0,side=tk.BOTTOM)

# Creates label for the countdown to begin
trainingRightBeginCountdownLabel = tk.Label(trainingRightWindow, 
    text='Countdown to begin: 3 sec',
    wraplength=500, 
    fg="black", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=0)
trainingRightBeginCountdownLabel.pack(padx=0,pady=0,side=tk.BOTTOM)




################################################################################################################
# "Training Backward" window creation and population
################################################################################################################   

# "Training Backward" window settings
trainingBackwardWindow = Toplevel()
trainingBackwardWindow.title('Training Backward')
trainingBackwardWindow.geometry('640x600')

# Make "Training Backward" window not resizable
trainingBackwardWindow.resizable(False, False)

# changing "Training Backward" background color
trainingBackwardWindow.config(background='black')

# Import feet clipart
feet_dir = os.path.join(curr_file_path, "extra", "feet2.png")
feet = PhotoImage(file=feet_dir)

# Show feet on screen
trainingBackwardFeet = tk.Label(trainingBackwardWindow,
    bg="black", 
    padx=10,
    pady=10,
    image=feet)
trainingBackwardFeet.place(x=200,y=200)

# Creates label to show the instructions
trainingBackwardInstructionsLabel = tk.Label(trainingBackwardWindow, 
    text='For this exercise, you are to clench all toes until '
    'the five second timer is up. Press the "Ready" button to '
    'start the session.',
    wraplength=500, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingBackwardInstructionsLabel.pack(padx=20,pady=20)

# Creates "Ready" button to proceed to the countdown to train backward.
trainingBackwardReadyButton = tk.Button(trainingBackwardWindow, 
    text='Ready', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=10,
    pady=10,
    command=lambda: training_ready_button("Backward"))
trainingBackwardReadyButton.pack(padx=0,pady=20,side=tk.BOTTOM)

# Creates label for the countdown to indicate remaining session duration
trainingBackwardSessionCountdownLabel = tk.Label(trainingBackwardWindow, 
    text='Session ends in: 5 sec',
    wraplength=500, 
    fg="black", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingBackwardSessionCountdownLabel.pack(padx=0,pady=0,side=tk.BOTTOM)

# Creates label for the countdown to begin
trainingBackwardBeginCountdownLabel = tk.Label(trainingBackwardWindow, 
    text='Countdown to begin: 3 sec',
    wraplength=500, 
    fg="black", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=0)
trainingBackwardBeginCountdownLabel.pack(padx=0,pady=0,side=tk.BOTTOM)




################################################################################################################
# "Training Rest" window creation and population
################################################################################################################   

# "Training Rest" window settings
trainingRestWindow = Toplevel()
trainingRestWindow.title('Training Rest')
trainingRestWindow.geometry('640x600')

# Make "Training Rest" window not resizable
trainingRestWindow.resizable(False, False)

# changing "Training Rest" background color
trainingRestWindow.config(background='black')

# Import rest clipart 
rest_dir = os.path.join(curr_file_path, "extra", "relax2.png")
rest = PhotoImage(file=rest_dir)

# Show rest clipart on screen
trainingRest = tk.Label(trainingRestWindow,
    bg="black", 
    padx=10,
    pady=10,
    image=rest)
trainingRest.place(x=200,y=200)

# Creates label to show the instructions
trainingRestInstructionsLabel = tk.Label(trainingRestWindow, 
    text='For this exercise, you are to relax your body until '
    'the five second timer is up. Press the "Ready" button to '
    'start the session.',
    wraplength=500, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingRestInstructionsLabel.pack(padx=20,pady=20)

# Creates "Ready" button to proceed to the countdown to train rest.
trainingRestReadyButton = tk.Button(trainingRestWindow, 
    text='Ready', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=10,
    pady=10,
    command=lambda: training_ready_button("Rest"))
trainingRestReadyButton.pack(padx=0,pady=20,side=tk.BOTTOM)

# Creates label for the countdown to indicate remaining session duration
trainingRestSessionCountdownLabel = tk.Label(trainingRestWindow, 
    text='Session ends in: 5 sec',
    wraplength=500, 
    fg="black", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingRestSessionCountdownLabel.pack(padx=0,pady=0,side=tk.BOTTOM)

# Creates label for the countdown to begin
trainingRestBeginCountdownLabel = tk.Label(trainingRestWindow, 
    text='Countdown to begin: 3 sec',
    wraplength=500, 
    fg="black", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=0)
trainingRestBeginCountdownLabel.pack(padx=0,pady=0,side=tk.BOTTOM)




################################################################################################################
# "Training Complete" window creation and population
################################################################################################################   
# "Training Complete" window settings
trainingCompleteWindow = tk.Tk()
trainingCompleteWindow.title('Training Complete!')
trainingCompleteWindow.geometry('500x200')

# Make "Training Complete" window not resizable
trainingCompleteWindow.resizable(False, False)

# changing "Training Complete" background color
trainingCompleteWindow.config(background='black')


# Creates label to show the large "Training complete!" sign
trainingCompleteMainLabel = tk.Label(trainingCompleteWindow, 
    text='Training complete! Press "Finish" to get started.',
    wraplength=500, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',18),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
trainingCompleteMainLabel.pack(padx=20,pady=0)

# Creates "Finish" button to proceed to "Training Forward" window
trainingCompleteFinishButton = tk.Button(trainingCompleteWindow, 
    text='Finish', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=20,
    pady=20,
    command=lambda: training_complete_finish_button())
trainingCompleteFinishButton.pack(padx=0,pady=20)



################################################################################################################
# "Load Profile" window creation and population
################################################################################################################   

# "Load Profile" window settings
loadProfileWindow = tk.Tk()
loadProfileWindow.title('Profile Loading')
loadProfileWindow.geometry('540x900')

# Make "Load Profile" window not resizable
loadProfileWindow.resizable(False, False)

# changing "Load Profile" background color
loadProfileWindow.config(background='black')

# Creates load profile buttons 1-3
loadProfileOneButton = tk.Button(loadProfileWindow, 
    text='New Profile 1',
    wraplength=400, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',40,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=6,
    padx=60,
    pady=35,
    command=lambda: load_profile_button(1))
loadProfileOneButton.pack(padx=20,pady=20)

loadProfileTwoButton = tk.Button(loadProfileWindow, 
    text='New Profile 2',
    wraplength=400, 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',40,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=6,
    padx=60,
    pady=35,
    command=lambda: load_profile_button(2))
loadProfileTwoButton.pack(padx=20,pady=20)

loadProfileThreeButton = tk.Button(loadProfileWindow, 
    text='New Profile 3', 
    wraplength=400,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',40,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=60,
    pady=35,
    command=lambda: load_profile_button(3))
loadProfileThreeButton.pack(padx=20,pady=20)

loadProfileBackButton = tk.Button(loadProfileWindow, 
    text='Back', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=20,
    pady=20,
    command=lambda: profile_loading_back_button())
loadProfileBackButton.pack(padx=20,pady=0)



################################################################################################################
# "Create Profile" window creation and population
################################################################################################################   

# "Create Profile" window settings
createProfileWindow = tk.Tk()
createProfileWindow.title('Profile creation')
createProfileWindow.geometry('540x300')

# Make "Create Profile" window not resizable
createProfileWindow.resizable(False, False)

# changing "Create Profile" background color
createProfileWindow.config(background='black')

# Creates label to show which profile you are creating
createProfileLabel = tk.Label(createProfileWindow, 
    text='Hello world!', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
createProfileLabel.pack(padx=20,pady=0)

# Creates an entry box for inputting name of the created profile
createProfileEntry = tk.Entry(createProfileWindow,
    font=('Terminal',19),
    relief=tk.RAISED,
    bd=8,
    fg="#41FF00", 
    bg="black",
    insertbackground="#41FF00")
createProfileEntry.pack(padx=20,pady=0)

# Creates label to show a note for character limit
createProfileNoteLabel = tk.Label(createProfileWindow, 
    text='NOTE: Max ten characters', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
createProfileNoteLabel.pack(padx=20,pady=0)

# Creates label to show a note for character limit
createProfileErrorLabel = tk.Label(createProfileWindow, 
    text='', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    #relief=tk.RAISED,
    bd=6,
    padx=10,
    pady=10)
createProfileErrorLabel.pack(padx=20,pady=0)

# Creates "Back" button to return to "Create Profile" window
createProfileBackButton = tk.Button(createProfileWindow, 
    text='Back', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=20,
    pady=20,
    command=lambda: profile_creation_back_button())
createProfileBackButton.place(x=130,y=200)

# Creates "Create" button to return to "Start" window
createProfileCreateButton = tk.Button(createProfileWindow, 
    text='Create', 
    wraplength=500,
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=4,
    padx=20,
    pady=20,
    command=lambda: profile_creation_create_button())
createProfileCreateButton.place(x=300,y=200)


'''
user1TrainButton = tk.Button(startWindow, 
    text='Train', 
    fg="#41FF00", 
    bg="black", 
    font=('Terminal',15,'bold'),
    activeforeground="#41FF00",
    activebackground='black',
    relief=tk.RAISED,
    bd=6,
    padx=5,
    pady=5,
    command=lambda: profile_selection(3))
user1TrainButton.place(x=100,y=150)
'''


# turn on pygame for sound mixing
pygame.mixer.init()
pygame.mixer.music.load(os.path.join(curr_file_path, 'extra', 'OGGstartupMacG3.ogg'))
pygame.mixer.music.play(loops=0)





# menu
menu = tk.Menu(directionalWindow)


# sub menu 1
file_menu = tk.Menu(menu, tearoff=False) # the tearoff parameter is whether or not you want the default tearoff that opens a new window
file_menu.add_command(label='New', command=lambda: print('New file'))
file_menu.add_command(label='Open', command=lambda: print('Open file'))
file_menu.add_separator()

menu.add_cascade(label='File', menu=file_menu)


# sub menu 2
options_menu = tk.Menu(menu, tearoff=False)
options_menu.add_command(label='Joystick Mode', command=lambda: print('Joystick Mode'))
options_menu.add_separator()
options_check_startup_sound_str = tk.StringVar(value='on')
options_menu.add_checkbutton(label='Startup sound',  
	onvalue='on', 
	offvalue='off', 
	variable=options_check_startup_sound_str,
	command=lambda: print(options_check_startup_sound_str.get()))

menu.add_cascade(label='Options', menu=options_menu)




canvas_width = 600
canvas_height = 600

canvas = tk.Canvas(directionalWindow, 
	bg='black', 
	width=canvas_width, 
	height=canvas_height)
canvas.place(x=1175,y=320)



# create canvas grid lines for x and y
x_line = canvas.create_line(0, (canvas_height/2),
	canvas_width+2, (canvas_height/2), 
	fill='white')

y_line = canvas.create_line((canvas_width/2), 0,
	(canvas_width/2), canvas_height+2, 
	fill='white')


################################################################################################################

# Directional indicator values  
stop_size = 50 				# must be divisible by 5 
stop_edge_size = (2/5)
stop_text = 'STOP'
stop_text_color = 'black'
stop_text_font = ('Terminal',28)
stop_fill = 'grey'
stop_activefill = '#cc0202'
oval_size = 70
oval_spacing = (150*1.15)
oval_size_left_right_reverse = 70
oval_fill = 'grey'
oval_outline = 'grey'
oval_outline_width = 5
oval_text_color = 'black'
oval_text_font = ('Terminal',22)
oval_text_slow = 'FORWARD'
oval_fill_slow = 'grey'
oval_activefill_slow = '#57E964'
oval_text_brisk = 'BRISK'
oval_fill_brisk = 'grey'
oval_activefill_brisk = 'yellow'
oval_text_left = 'LEFT'
oval_fill_left = 'grey'
oval_activefill_left = '#34deeb'
oval_text_right = 'RIGHT'
oval_fill_right = 'grey'
oval_activefill_right = '#34deeb'
oval_text_reverse = 'REVERSE'
oval_fill_reverse = 'grey'
oval_activefill_reverse = '#eb34e8'

# for oval text visibility, use state=tk.HIDDEN to make the text invisible!




################################################################################################################
# System Checking
################################################################################################################

system_os_check()
main_directory_checker()
# user_directory_checker()




################################################################################################################
# Setup of diagnostics
################################################################################################################

################################################################################################################
# Setup of keyboard controls
################################################################################################################



directionalWindow.configure(menu=menu)
directionalWindow.withdraw()                           # Hides the "Directional Window" until needed
#systemInitWindow.withdraw()                 # Hides the "System Initializer" window until needed
startWindow.withdraw()                 # Hides the "Start" window until needed
newProfileWindow.withdraw()                 # Hides the "New Profile" window until needed
loadProfileWindow.withdraw()                # Hides the "Load Profile" window until needed
createProfileWindow.withdraw()              # Hides the "Create Profile" window until needed
trainingWelcomeWindow.withdraw()            # Hides the "Training Welcome" window until needed
trainingForwardWindow.withdraw()            # Hides the "Training Forward" window until needed
trainingLeftWindow.withdraw()               # Hides the "Training Left" window until needed
trainingRightWindow.withdraw()              # Hides the "Training Right" window until needed
trainingBackwardWindow.withdraw()           # Hides the "Training Backward" window until needed
trainingRestWindow.withdraw()               # Hides the "Training Rest" window until needed
trainingCompleteWindow.withdraw()           # Hides the "Training Complete" window until needed
#window.mainloop()
while(True):                                # This is the loop that keeps the GUI frames generating
    #window.withdraw()
    #directional_window()
    #window.update_idletasks()              # Updates "window" window even if not called upon to prevent errors 
    #window.update()                        # Updates "window" window even if not called upon to prevent errors 
    update_system_init_window()
    update_start_window()
    update_new_profile_window()
    update_load_profile_window()
    update_create_profile_window()
    update_training_welcome_window()
    update_training_forward_window()


#thread2Directional.join()