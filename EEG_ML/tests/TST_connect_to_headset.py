
from pyOpenBCI import OpenBCICyton

# perform this headset while the USB port is plugged in and the headset is off

def print_raw(sample):
    print(sample.channels_data)

board = OpenBCICyton(port='/dev/ttyUSB0', daisy=True)

board.start_stream(print_raw)
