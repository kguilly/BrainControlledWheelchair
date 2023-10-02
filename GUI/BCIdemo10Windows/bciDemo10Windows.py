import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import pygame
import time
import keyboard
import os.path

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


############################################################################################



def diagnostic_test():
    print("Placeholder diagnostic testing...")


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
    # Stop indicator

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
    #print("sus")
    startWindow.update_idletasks()     # Updates "startWindow" window even if not called upon to prevent errors 
    startWindow.update()               # Updates "startWindow" window even if not called upon to prevent errors 
    

def update_new_profile_window():
    #print("sus")
    newProfileWindow.update_idletasks()     # Updates "newProfileWindow" window even if not called upon to prevent errors 
    newProfileWindow.update()               # Updates "newProfileWindow" window even if not called upon to prevent errors 



def update_load_profile_window():
    #print("sus")
    loadProfileWindow.update_idletasks()     # Updates "loadProfileWindow" window even if not called upon to prevent errors 
    loadProfileWindow.update()               # Updates "loadProfileWindow" window even if not called upon to prevent errors 



def update_create_profile_window():
    #print("sus")
    createProfileWindow.update_idletasks()     # Updates "createProfileWindow" window even if not called upon to prevent errors 
    createProfileWindow.update()               # Updates "createProfileWindow" window even if not called upon to prevent errors 




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
    if (os.path.isfile('./file1.txt')):
        prof1Status = True
        file = open("file1.txt", "r")
        fileName = file.readlines()
        prof1Name = fileName[0]
        prof1Name.strip()
        print(prof1Name)
        file.close()

    if (os.path.isfile('./file2.txt')):
        prof2Status = True
        file = open("file2.txt", "r")
        prof2Name = file.readline()
        file.close()

    if (os.path.isfile('./file3.txt')):
        prof3Status = True
        file = open("file3.txt", "r")
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
        loadProfileOneButton.config(text="Profile " + prof1Name)
        loadProfileOneButton.config(state='active')
    else:
        loadProfileOneButton.config(state='disabled')

    if (prof2Status != False):
        loadProfileTwoButton.config(text="Profile " + prof2Name)
        loadProfileTwoButton.config(state='active')
    else:    
        loadProfileTwoButton.config(state='disabled')

    if (prof3Status != False):
        loadProfileThreeButton.config(text="Profile " + prof3Name)
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
        file = open("file"+profSelected+".txt", "w")
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




# "Directional Window" window settings
window = tk.Tk()
window.title('Brain-controlled Car')
window.geometry('1920x1080')

# changing "Directional Window" icon and background color
icon = tk.PhotoImage(file='wheelchairIcon2.png')
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
# "Load Profile" window creation and population
################################################################################################################   

# "Load Profile" window settings
loadProfileWindow = tk.Tk()
loadProfileWindow.title('Profile Loading')
loadProfileWindow.geometry('540x900')

# Make "New Profile" window not resizable
loadProfileWindow.resizable(False, False)

# changing "New Profile" background color
loadProfileWindow.config(background='black')

# Creates new profile buttons 1-3
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
pygame.mixer.music.load('OGGstartupMacG3.ogg')
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
# Setup of diagnostics
################################################################################################################

diagnostic_test()



################################################################################################################
# Setup of keyboard controls
################################################################################################################



window.configure(menu=menu)
window.withdraw()                           # Hides the "Directional Window" until needed
newProfileWindow.withdraw()                 # Hides the "New Profile" window until needed
loadProfileWindow.withdraw()                # Hides the "Load Profile" window until needed
createProfileWindow.withdraw()              # Hides the "Create Profile" window until needed
#window.mainloop()
while(True):                                # This is the loop that keeps the GUI frames generating
    #window.withdraw()
    #directional_window()
    #window.update_idletasks()              # Updates "window" window even if not called upon to prevent errors 
    #window.update()                        # Updates "window" window even if not called upon to prevent errors 
    update_start_window()
    update_new_profile_window()
    update_load_profile_window()
    update_create_profile_window()


