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

# TODO: implement user choice for config of com port / directory
# this is the directory that the heaadset is connected to
headset_directory = 'COM1' # for windows, a com port, for linux, usually '/dev/ttyUSB0'
curr_file_path = os.path.dirname(os.path.abspath(__file__))

# TODO: implement two different choices of using: either use the keyboard or headset
keyboard_user_mode = True


############################################################################################




def main_directory_checker():
    global curr_file_path
    masterProfilePath = os.path.join(curr_file_path, 'Profiles' )
    if not os.path.exists(masterProfilePath):
        os.makedirs(masterProfilePath)


def user_directory_checker():
    print("user_directory_checker()")




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


def brisk_joystick():
    global stop_fill, oval_fill_slow, oval_fill_brisk, oval_fill_left, oval_fill_right, oval_fill_reverse, oval_fill
    stop_fill = oval_fill
    oval_fill_slow = oval_fill
    oval_fill_brisk = 'yellow'
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



################################################################################################################
# "Directional Window" window updater
################################################################################################################

def directional_window():
    #if (lower_min_trigger < chanX.value < upper_min_trigger) and (lower_min_trigger < chanY.value < upper_min_trigger):
    #    stop_joystick()
    global keyboard_user_mode

    if keyboard_user_mode == True:
        if (keyboard.is_pressed("w")) and (keyboard.is_pressed("Shift")):
            brisk_joystick()
        elif (keyboard.is_pressed("w")):
            slow_joystick()
        elif (keyboard.is_pressed("s")):
            reverse_joystick()
        elif (keyboard.is_pressed("a")):
            left_joystick()
        elif (keyboard.is_pressed("d")):
            right_joystick()
        else:
            stop_joystick()
        #---------------------------------------------------------------------------------------------------------
    else: # use the output from the headset
    # Stop indicator
        # TODO: GENERATE PREDICTIONS FROM INCOMING DATA
        pass

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
        fill=stop_fill,
        activefill=stop_activefill)

    canvas.create_text((canvas_width/2),        # Stop indicator text creation
        (canvas_height/2),
        text=stop_text, 
        fill=stop_text_color,
        font=stop_text_font)


    #---------------------------------------------------------------------------------------------------------
    # Forward slow indicator

    forward_oval_slow = canvas.create_oval(
        ((canvas_width/2)-oval_size),(((canvas_height/2)-oval_spacing_slow)-(oval_size)*(2/3)),
        ((canvas_width/2)+oval_size),(((canvas_height/2)-oval_spacing_slow)+(oval_size)*(2/3)),
        fill=oval_fill_slow,
        activefill=oval_activefill_slow)

    canvas.create_text(((canvas_width/2)),      # Forward slow indicator text creation
        (((canvas_height/2)-oval_spacing_slow)),
        text=oval_text_slow, 
        fill=oval_text_color,
        font=oval_text_font)

    #---------------------------------------------------------------------------------------------------------
    # Forward brisk indicator

    forward_oval_brisk = canvas.create_oval(
        ((canvas_width/2)-oval_size),(((canvas_height/2)-(oval_spacing_brisk))-(oval_size)*(2/3)),
        ((canvas_width/2)+oval_size),(((canvas_height/2)-(oval_spacing_brisk))+(oval_size)*(2/3)),
        fill=oval_fill_brisk,
        activefill=oval_activefill_brisk)

    canvas.create_text(((canvas_width/2)),      # Forward brisk indicator text creation
        (((canvas_height/2)-(oval_spacing_brisk))),
        text=oval_text_brisk, 
        fill=oval_text_color, 
        font=oval_text_font)

    #---------------------------------------------------------------------------------------------------------
    # Reverse indicator

    reverse_oval = canvas.create_oval(
        ((canvas_width/2)-oval_size_left_right_reverse),(((canvas_height/2)+oval_spacing_left_right_reverse)-(oval_size)*(2/3)),
        ((canvas_width/2)+oval_size_left_right_reverse),(((canvas_height/2)+oval_spacing_left_right_reverse)+(oval_size)*(2/3)),
        fill=oval_fill_reverse,
        activefill=oval_activefill_reverse)

    canvas.create_text(((canvas_width/2)),      # Reverse indicator text creation
        (((canvas_height/2)+(oval_spacing_left_right_reverse))),
        text=oval_text_reverse, 
        fill=oval_text_color,
        font=oval_text_font)

    #---------------------------------------------------------------------------------------------------------
    # Left indicator

    left_oval = canvas.create_oval(
        (((canvas_width/2)-oval_spacing_left_right_reverse)-oval_size),((canvas_height/2)-(oval_size)*(2/3)),
        (((canvas_width/2)-oval_spacing_left_right_reverse)+oval_size),((canvas_height/2)+(oval_size)*(2/3)),
        fill=oval_fill_left,
        activefill=oval_activefill_left)

    canvas.create_text((((canvas_width/2)-oval_spacing_left_right_reverse)),        # Left indicator text creation
        (canvas_height/2),
        text=oval_text_left, 
        fill=oval_text_color,
        font=oval_text_font)

    #---------------------------------------------------------------------------------------------------------
    # Right indicator

    right_oval = canvas.create_oval(
        (((canvas_width/2)+oval_spacing_left_right_reverse)-oval_size),((canvas_height/2)-(oval_size)*(2/3)),
        (((canvas_width/2)+oval_spacing_left_right_reverse)+oval_size),((canvas_height/2)+(oval_size)*(2/3)),
        fill=oval_fill_right,
        activefill=oval_activefill_right)

    canvas.create_text((((canvas_width/2)+oval_spacing_left_right_reverse)),        # Right indicator text creation
        (canvas_height/2),
        text=oval_text_right, 
        fill=oval_text_color,
        font=oval_text_font)
    window.update_idletasks()
    window.update()



################################################################################################################
# Window updaters
################################################################################################################

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



def update_training_complete_window():
    trainingCompleteWindow.update_idletasks()     # Updates "trainingCompleteWindow" window even if not called upon to prevent errors 
    trainingCompleteWindow.update()               # Updates "trainingCompleteWindow" window even if not called upon to prevent errors 



# Guides user from the "Start" window to next screen
def start_selection(choice):
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



def profile_creation_create_button():
    global profSelected
    spaceTest = " " in (createProfileEntry.get())
    
    if (len((createProfileEntry.get())) > 0 and len((createProfileEntry.get())) <= 10 and str(spaceTest) == "False"):
        startWindow.deiconify()
        newProfileWindow.withdraw()
        createProfileWindow.withdraw()
        if not os.path.exists("./Profiles/Profile"+profSelected):
            os.makedirs("./Profiles/Profile"+profSelected)              # creates individual profile folders if they do not exist.
        file = open("./Profiles/Profile"+profSelected+"/"+"profile"+profSelected+"Config"+".txt", "w")
        file.write(createProfileEntry.get())
        file.close()
        createProfileEntry.delete(0,tk.END)
        createProfileErrorLabel.config(text="")
    else:
        if (len((createProfileEntry.get())) <= 0):
            createProfileErrorLabel.config(text="ERROR: Must contain at least one character!")
        elif (len((createProfileEntry.get())) > 10):
            createProfileErrorLabel.config(text="ERROR: Too many characters!")
        elif (str(spaceTest) == "True"):
            createProfileErrorLabel.config(text="ERROR: Must not contain spaces!")
        else:
            createProfileErrorLabel.config(text="ERROR please try again!")
    trainingWelcomeWindow.deiconify()
    createProfileWindow.withdraw()
    startWindow.withdraw()




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
    init.gather_training_data(headset, label, profile_path)
    init.end_session(headset)




def training_ready_button(direction):
    global trainingFlag, profSelected
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
    labelBeginChoice.config(text='THINK!')
    updaterChoice()
    print("0")
    
    # profile path
    global profSelected, curr_file_path
    profile_dir = os.path.join(curr_file_path, profSelected)
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
        trainingCompleteWindow.deiconify()                                                        # CHANGE THIS LATER
        directionButtonChoice.config(command=lambda: training_ready_button("Backward"))
        directionButtonChoice.config(text='Ready')


def training_complete_finish_button():
    window.deiconify()
    trainingCompleteWindow.withdraw()
    directional_window()




# "Directional Window" window settings
window = tk.Tk()
window.title('Brain-controlled Car')
window.geometry('1920x1080')

# changing "Directional Window" icon and background color
icon = tk.PhotoImage(file=os.path.join(curr_file_path, 'extra', 'wheelchairIcon2.png'))
window.iconphoto(True, icon)
window.config(background='black')





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

# Show left fist on screen
trainingBackwardFeet = tk.Label(trainingBackwardWindow,
    bg="black", 
    padx=10,
    pady=10,
    image=feet)
trainingBackwardFeet.place(x=200,y=200)

# Creates label to show the instructions
trainingBackwardInstructionsLabel = tk.Label(trainingBackwardWindow, 
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
# "Training Complete" window creation and population
################################################################################################################   
# TODO: make a waiting screen for the model to train
# after the training is complete, train the model
 


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
    pady=35)
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
    pady=35)
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
    pady=35)
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
menu = tk.Menu(window)


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




canvas_width = 800
canvas_height = 800

canvas = tk.Canvas(window, 
	bg='black', 
	width=canvas_width, 
	height=canvas_height)
canvas.place(x=1000,y=100)



# create canvas grid lines for x and y
x_line = canvas.create_line(0, (canvas_height/2),
	canvas_width+2, (canvas_height/2), 
	fill='white')

y_line = canvas.create_line((canvas_width/2), 0,
	(canvas_width/2), canvas_height+2, 
	fill='white')


################################################################################################################

# Directional indicator values  
stop_size = 60 				# must be divisible by 5 
stop_edge_size = (2/5)
stop_text = 'STOP'
stop_text_color = 'black'
stop_text_font = ('Terminal',28)
stop_fill = 'grey'
stop_activefill = '#cc0202'
oval_size = 80
oval_spacing = 150
oval_spacing_multiplier = 1.9
oval_spacing_slow = oval_spacing*1.0
oval_spacing_brisk = oval_spacing*1.9
oval_size_left_right_reverse = 90
oval_spacing_left_right_reverse = (oval_spacing*1.4)
oval_fill = 'grey'
oval_outline = 'grey'
oval_outline_width = 5
oval_text_color = 'black'
oval_text_font = ('Terminal',22)
oval_text_slow = 'SLOW'
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
# Directory checking
################################################################################################################

main_directory_checker()
user_directory_checker()




################################################################################################################
# Setup of diagnostics
################################################################################################################
headset, connection_status = init.diagnostic_test(headset_directory)
if connection_status == False:
    print("Could not connect to headset")
    exit(1)



################################################################################################################
# Setup of keyboard controls
################################################################################################################



window.configure(menu=menu)
window.withdraw()                           # Hides the "Directional Window" until needed
newProfileWindow.withdraw()                 # Hides the "New Profile" window until needed
loadProfileWindow.withdraw()                # Hides the "Load Profile" window until needed
createProfileWindow.withdraw()              # Hides the "Create Profile" window until needed
trainingWelcomeWindow.withdraw()            # Hides the "Training Welcome" window until needed
trainingForwardWindow.withdraw()            # Hides the "Training Forward" window until needed
trainingLeftWindow.withdraw()               # Hides the "Training Left" window until needed
trainingRightWindow.withdraw()              # Hides the "Training Right" window until needed
trainingBackwardWindow.withdraw()           # Hides the "Training Backward" window until needed
trainingCompleteWindow.withdraw()           # Hides the "Training Complete" window until needed
#window.mainloop()
while(True):                                # This is the loop that keeps the GUI frames generating
    #window.withdraw()
    #directional_window()
    window.update_idletasks()              # Updates "window" window even if not called upon to prevent errors 
    window.update()                        # Updates "window" window even if not called upon to prevent errors 
    update_start_window()
    update_new_profile_window()
    update_load_profile_window()
    update_create_profile_window()
    update_training_welcome_window()
    update_training_forward_window()


