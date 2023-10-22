import machine
sda=machine.Pin(0)
scl=machine.Pin(1)

i2c=machine.I2C(0, sda=sda, scl=scl, freq=400000)
addresses = i2c.scan()
count = 1
for address in addresses:
    print(count, " I2C address: ", hex(address))
    count+=1