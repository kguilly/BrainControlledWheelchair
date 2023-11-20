# this function will implement keyboard controls
# has to connect to the wheelchiar com port

###### TO CHECK IF WINDOWS OR LINUX :
import platform
import pygame
import time
import serial 
import keyboard








#GLOBAL VARIABLES
serial_port = '/dev/rfcomm0'
ser = serial.Serial(serial_port, baudrate=921600)




system_platform = platform.system()
if system_platform == "Linux":
    serial_port = '/dev/ttyUSB0'

elif system_platform == "Windows":
    serial_port = 'COM3'

elif system_platform == 'Darwin':
    serial_port = 'idk'
    print("we hate mac")
    exit(1)


# code will received a keyboard input and will transmit based on it
# trans

def receive_and_transmit_keyboard_input():
    global ser
    flip_warning, collision_warning = None, None
    
    message = ''
    if (keyboard.is_pressed("w")) and (keyboard.is_pressed("Shift")):
        message = 'f'
    elif (keyboard.is_pressed("w")):
        message = 'f'
    elif (keyboard.is_pressed("s")):
        message = 'b'
    elif (keyboard.is_pressed("a")):
        message = 'l'
    elif (keyboard.is_pressed("d")):
        message = 'r'
    else:
        return
    
    received_data = receive_data()

    if received_data == 'l':
        # the rc is level
        # do nothing
        flip_warning = False
    elif receive_data == 'f':
        # the rc is flipped
        # stop, display wanring
        flip_warning = True 
    elif receive_data == 's':
        # no collision
        collision_warning = False
    elif receive_data == 'c':
        # collision
        # stop, display warning
        collision_warning = True
        return flip_warning, collision_warning
    
    try:
        ser.write(message.encode())

        return flip_warning, collision_warning
    except: 
        return flip_warning, collision_warning

    
def receive_data(): 
    global ser
    data = ser.read()
    return data