import pyedflib
import numpy as np
import os


def map_to_samples(total_secs, onset_sec, next_onset_sec, total_samples): 
    # return a 2 element array, start sample and end sample

    start_sample = int((onset_sec * total_samples) / total_secs)
    end_sample = int((next_onset_sec * total_samples) / total_secs)

    return [start_sample, end_sample]

label_mapping = {
    1 : "Rest",
    2 : "Squeeze Both Fists",
    3 : "Squeeze Both Feet",
    4 : "Squeeze Left Hand" ,
    5 : "Squeeze Right Hand" ,
}

path = '/home/kaleb/Documents/eeg_dataset/files/S001'
all_files = os.listdir(path)
edf_files = [file for file in all_files if file.endswith('.edf')]

# if the file is not one of interest, continue (1, 2, 4, 6, 8, 10, 12, 14)
files_to_skip = {1, 2, 4, 6, 8, 10, 12, 14}

left_fist_files = {3, 7, 11}
right_fist_files = {5, 9, 13}

X = []
Y = []

for file in edf_files: 
    file_path = os.path.join(path, file)
    file_eeg_data = []

    file_num = int(file_path[-6] + file_path[-5])    
    if file_num in files_to_skip: # skip the files we don't need
        print("Skipping file: ", file_path)
        continue

    # grab the eeg data and the associated annotations
    edf_data = pyedflib.EdfReader(file_path)
    annotations = edf_data.readAnnotations()

    total_secs = edf_data.getFileDuration()
    total_samples = edf_data.getNSamples()

    # read each channel into an array
    for channel in range(64):
        arr = edf_data.readSignal(channel)
        file_eeg_data.append(arr)

    edf_data.close()

    # make the arrays into 3d numpy arrays
    file_eeg_data = np.stack(file_eeg_data)
    annotations = np.stack(annotations)

    # go through each annotation, extract relevant task information
    for i in range(annotations.shape[1]): 
        sec = annotations[0][i]
        task = annotations[2][i]
        
        #IMPORTANT: EACH ARRAY NEEDS TO BE OF SHAPE (1, 64, SAMPLES) B4 APPENDING
        
        # for all files used (3, 7, 11, 5, 9, 13), t0 is rest
        if task == 'T0': 
            # get start and end samples, append to X and Y arrs
            pass

        # if this is 3, 7, or 11, then t1 = squeeze left fist, 
        # t2 = squeeze both fists
        # if this is 5, 9, or 13, then t1 = squeeze right fist,
        # t2 = squeeze both feet
        elif task == 'T1':
            if file_num in left_fist_files:
                pass
            elif file_num in right_fist_files:
                pass

        elif task == 'T2':
            if file_num in left_fist_files:
                pass
            elif file_num in right_fist_files:
                pass

        else:
            print("This is some other task: ", task)

        


        print(sec, task)

