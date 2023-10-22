from machine import Pin, PWM
import utime
import math

led = machine.Pin(25, machine.Pin.OUT)

# Pins for locomotion system
motorPWMFreq = 1000 # speed control PWM frequency
pwmLeftWheel = PWM(Pin(22)) # Left Wheel PWM speed control
pwmLeftWheel.freq(motorPWMFreq)
leftWheelForawrd = machine.Pin(21, machine.Pin.OUT, Pin.PULL_DOWN)  # Left wheel forward control
leftWheelReverse = machine.Pin(20, machine.Pin.OUT, Pin.PULL_DOWN)  # Left wheel reverse control
rightWheelForawrd = machine.Pin(19, machine.Pin.OUT, Pin.PULL_DOWN) # Right wheel forward control
rightWheelReverse = machine.Pin(18, machine.Pin.OUT, Pin.PULL_DOWN) # Right wheel reverse control
pwmRightWheel = PWM(Pin(17))# Right Wheel PWM speed control
pwmRightWheel.freq(motorPWMFreq)

# Initialization 
leftWheelForawrd.value(0)
leftWheelReverse.value(0)
rightWheelForawrd.value(0)
rightWheelReverse.value(0)
pwmRightWheel.duty_u16(0)
pwmLeftWheel.duty_u16(0)

def main():
    while True:
        leftWheelForawrd.value(1)
        leftWheelReverse.value(0)
        rightWheelForawrd.value(1)
        rightWheelReverse.value(0)
        
        utime.sleep_ms(10)
        
        print("Testing forward speed increase")
        for i in range(65025):
        #for i in range(10):
           pwmRightWheel.duty_u16(i)
           pwmLeftWheel.duty_u16(i)
           utime.sleep_us(1000)
           led.toggle()
           print("Testing forward ", int((i/65025)*100), '%', end = '\r')
            
        utime.sleep_ms(100)
        pwmRightWheel.duty_u16(0)
        pwmLeftWheel.duty_u16(0)
        leftWheelForawrd.value(0)
        leftWheelReverse.value(1)
        rightWheelForawrd.value(0)
        rightWheelReverse.value(1)
        
        utime.sleep_ms(100)
        
        print("Testing reverse speed increase")
        for i in range(65025):
        #for i in range(10):
           pwmRightWheel.duty_u16(i)
           pwmLeftWheel.duty_u16(i)
           utime.sleep_us(1000)
           led.toggle()
           print("Testing reverse ", int((i/65025)*100), '%', end = '\r')
           
        utime.sleep_ms(100)
        pwmRightWheel.duty_u16(0)
        pwmLeftWheel.duty_u16(0)
        leftWheelForawrd.value(0)
        leftWheelReverse.value(0)
        rightWheelForawrd.value(0)
        rightWheelReverse.value(0)
        
        print("Rapid switch test")
        utime.sleep_ms(10)
        pwmRightWheel.duty_u16(65025)
        pwmLeftWheel.duty_u16(65025)
        utime.sleep_ms(10)
        
        for i in range(100):
            leftWheelForawrd.value(1)
            leftWheelReverse.value(0)
            rightWheelForawrd.value(1)
            rightWheelReverse.value(0)
            utime.sleep_ms(100)
            leftWheelForawrd.value(0)
            leftWheelReverse.value(1)
            rightWheelForawrd.value(0)
            rightWheelReverse.value(1)
            utime.sleep_ms(100)
            led.toggle()
            print("Testing rapid switch ", int((i/100)*100), '%', end = '\r')
        
        print("DONE")
        utime.sleep_ms(100000)
    

if __name__ == "__main__":
    main()








