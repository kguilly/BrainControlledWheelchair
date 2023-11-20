# BrainControlledWheelchair
This is a repository to house the machine learning code and Overleaf Project for the iterations of the Senior Design Report for the Brain-Controlled Wheelchair. 

# For Connecting to Linux: 
- HEADSET: must give access to usb port
    - use `dmesg | grep tty` to see which usb path is connected
    - give access rights to that path with the following commands:
      `$ sudo usermod -aG dialout $USER`
      `$ sudo chmod a+rw <path>`
- WHEELCHAIR: must assign bluetooth communication to an rfcomm port
  - Open bluetooth settings, connect to 'RC Wheelchair', password is 1234
      - Will show disconected, this is ok
  - Open terminal, type `hcitool scan`, mark down the bluetooth address associated to the device
  - type `sudo rfcomm bind 0 <address> 1` to bind the device to `/dev/rfcomm1` 

`Design_Assignments` File Structure: 
- First subdirectory separates the different assignments for design 1 and 2. Only the final assignment is included for design 1. 

`figs/`
- This is a folder to house the figures of the project. Made a subfolder for each section of the paper

`refs.bib`
- Where to house citations

`conference_101719.tex`
- The LaTex document that produces the PDF


[BrainFlow Documentation](https://brainflow.readthedocs.io/en/stable/Examples.html)
