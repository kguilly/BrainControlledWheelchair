# this function will implement keyboard controls
# has to connect to the wheelchiar com port

###### TO CHECK IF WINDOWS OR LINUX :
import platform
import pygame
import time
import serial 
import keyboard



#GLOBAL VARIABLES
serial_port = None
# ser = serial.Serial(serial_port, baudrate=921600)
ser = None
def init_connection_to_pi(port):
    global serial_port, ser

    try:
        serial_port = port
        ser = serial.Serial(port, baudrate=921600)
        # test write
        return True

    except:
        False

# code will received a keyboard input and will transmit based on it
# trans

def receive_and_transmit_keyboard_input(key_press):
    global ser
    flip_warning, collision_warning = None, None
    
    message = ''
    if key_press == 'w':
        message = 'f'
    elif key_press == 's':
        message = 'b'
    elif key_press == 'a':
        message = 'r'
    elif key_press == 'd':
        message = 'l'
    else:
        return
    
    received_data = receive_data(ser)
    if 'l' in received_data:
        # the rc is level
        # do nothing
        flip_warning = False
    elif 'f' in received_data:
        # the rc is flipped
        # stop, display wanring
        flip_warning = True 
    elif 's' in received_data:
        # no collision
        collision_warning = False
    elif 'c' in received_data:
        # collision
        # stop, display warning
        collision_warning = True
        return flip_warning, collision_warning
    
    try:
        ser.write(message.encode())

        return flip_warning, collision_warning
    except: 
        return flip_warning, collision_warning

    
def receive_data(ser_obj):
    try:
        data = str(ser_obj.read(1))
        # print(data)
        return data
    except:
        return None