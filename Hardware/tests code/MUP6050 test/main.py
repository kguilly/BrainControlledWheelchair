from machine import I2C, Pin
from imu import MPU6050
import utime
import math

led = machine.Pin(25, machine.Pin.OUT)

sda=machine.Pin(0)
scl=machine.Pin(1)

i2c=machine.I2C(0, sda=sda, scl=scl, freq=400000)
imu = MPU6050(i2c)

def movingAverage(data):
    avg = sum(data)/len(data)
    del data[0]
    #print(len(data))
    return round(avg,1)

def getIMUData():
    ax=imu.accel.x
    ay=imu.accel.y
    az=imu.accel.z
    gx=imu.gyro.x
    gy=imu.gyro.y
    gz=imu.gyro.z
    
    return ax, ay, az, gx, gy, gz

def calculateTilts(ax, ay, az):
    RAD_TO_DEG = 180.0/math.pi
    tiltX = math.atan2(ay, math.sqrt(ax*ax + az*az))*RAD_TO_DEG
    tiltY = math.atan2(-ax, math.sqrt(ay*ay + az*az))*RAD_TO_DEG
    tiltZ = 90 - math.atan2(az, math.sqrt(ax*ax + ay*ay))*RAD_TO_DEG
    
    return tiltX, tiltY, tiltZ

def main():
    
    xAngles = [0]*30
    yAngles = [0]*30
    zAngles = [0]*30
    
    message = ""
    
    while True:
        utime.sleep_ms(10)
        led.toggle()
        ax, ay, az, gx, gy, gz = getIMUData()
        tiltX, tiltY, tiltZ = calculateTilts(ax, ay, az)
        xAngles.append(round(tiltX, 1))
        yAngles.append(round(tiltY, 1))
        zAngles.append(round(tiltZ, 1))
            
        tiltX = movingAverage(xAngles)
        tiltY = movingAverage(yAngles)
        tiltZ = movingAverage(zAngles)
        if tiltX > 45 or tiltX < -45 or tiltY > 45 or tiltY < -45:
            message = " I'm flipping"

        else:
            message = " I'm stable             "
        
        print(f"Tilt x: {tiltX:.2f}, Tilt y: {tiltY:.2f}, Tilt z: {tiltZ:.2f}", message, end='\r')
            

if __name__ == "__main__":
    main()