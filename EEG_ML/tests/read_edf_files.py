import pyedflib
import numpy as np
import os


def map_to_samples(total_secs, onset_sec, next_onset_sec, total_samples):
    # return a 2 element array, start sample and end sample
    start_sample = int((onset_sec * total_samples) / total_secs)
    end_sample = int((next_onset_sec * total_samples) / total_secs)

    return [start_sample, end_sample]


def reader(passed_path, patient_num):
    label_mapping = {
        1: "Rest",
        2: "Squeeze Both Fists",
        3: "Squeeze Both Feet",
        4: "Squeeze Left Hand",
        5: "Squeeze Right Hand",
    }

    str_patient_num = str(patient_num).zfill(3)
    path = os.path.join(passed_path, 'S' + str_patient_num)
    
    all_files = os.listdir(path)
    edf_files = [file for file in all_files if file.endswith('.edf')]

    # if the file is not one of interest, continue (1, 2, 4, 6, 8, 10, 12, 14)
    files_to_skip = {1, 2, 4, 6, 8, 10, 12, 14}

    left_right_files = {3, 7, 11}
    feet_hands_files = {5, 9, 13}

    X = []
    Y = []

    for file in edf_files:
        file_path = os.path.join(path, file)
        file_eeg_data = []

        file_num = int(file_path[-6] + file_path[-5])
        if file_num in files_to_skip:  # skip the files we don't need
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
        file_eeg_data = np.stack(file_eeg_data)  # shape (channels[64], samples[2000])
        annotations = np.stack(annotations)

        # go through each annotation, extract relevant task information
        for i in range(annotations.shape[1]):
            sec = float(annotations[0][i])
            task = annotations[2][i]

            try:
                next_sec = float(annotations[0][i + 1])
            except:
                next_sec = total_secs

            # IMPORTANT: EACH ARRAY NEEDS TO BE OF SHAPE (1(trials), 64(channels), SAMPLES) B4 APPENDING
            samples = []
            label = 0
            # for all files used (3, 7, 11, 5, 9, 13), t0 is rest
            if task == 'T0':
                # get start and end samples, append to X and Y arrs
                start_end_sample_arr = map_to_samples(total_secs, sec, next_sec, total_samples[0])
                samples = file_eeg_data[:, start_end_sample_arr[0]:start_end_sample_arr[1]]
                label = 1

            # if this is 3, 7, or 11, then t1 = squeeze left fist,
            # t2 = squeeze both fists
            # if this is 5, 9, or 13, then t1 = squeeze right fist,
            # t2 = squeeze both feet
            elif task == 'T1':
                if file_num in left_right_files:
                    # t1 is squeeze left fist
                    start_end_sample_arr = map_to_samples(total_secs, sec, next_sec, total_samples[0])
                    samples = file_eeg_data[:, start_end_sample_arr[0]:start_end_sample_arr[1]]
                    label = 4
                elif file_num in feet_hands_files:
                    # t1 is squeeze both fists
                    start_end_sample_arr = map_to_samples(total_secs, sec, next_sec, total_samples[0])
                    samples = file_eeg_data[:, start_end_sample_arr[0]:start_end_sample_arr[1]]
                    label = 2

            elif task == 'T2':
                if file_num in left_right_files:
                    # t2 is squeeze right fist
                    start_end_sample_arr = map_to_samples(total_secs, sec, next_sec, total_samples[0])
                    samples = file_eeg_data[:, start_end_sample_arr[0]:start_end_sample_arr[1]]
                    label = 5

                elif file_num in feet_hands_files:
                    # t2 is squeeze both feet
                    start_end_sample_arr = map_to_samples(total_secs, sec, next_sec, total_samples[0])
                    samples = file_eeg_data[:, start_end_sample_arr[0]:start_end_sample_arr[1]]
                    label = 3

            else:
                print("This is some other task: ", task)

            samples = np.stack(samples)
            samples = samples.reshape((1, 64, len(samples[0])))

            X.append(samples)
            Y.append(label)

    # calculate the num samples in each
    samps = np.array([subarr.shape[2] for subarr in X])
    absolute_min = int(np.median(samps) / 2)

    filtered_X = []  # will drop the arrs that are too short from x

    # drop all the arrays less than min_len, and chop all the ones down that are longer
    for subarray in X:
        size = subarray.shape[2]
        if size > absolute_min:
            filtered_X.append(subarray)

    # find the lowest number in the filtered array, and reshape all elems to fit
    samps = np.array([subarr.shape[2] for subarr in filtered_X])
    min_samps = np.min(samps)

    reshaped_filtered_X = []
    # reshape the arrays to all be the same size
    for arr in filtered_X:
        num_samps = arr.shape[2]
        modified_arr = arr[0][:, :min_samps]
        reshaped_filtered_X.append(modified_arr)

    X = np.array(reshaped_filtered_X)
    Y = np.array(Y)
    # print("X.shape: ", X.shape)
    # print("Y.shape: ", Y.shape)

    return X,Y


def split_by_second(X, Y, sample_rate, num_channels):
    
    trial_duration = X.shape[2]
    num_segments = trial_duration // sample_rate

    X_sec = np.empty((0, num_channels, sample_rate))
    Y_sec = []

    count = -1
    for trial in X:
        count += 1
        for i in range(num_segments):
            start_idx = i * (sample_rate)
            end_idx = start_idx + sample_rate
            segment = trial[:, start_idx:end_idx]
            try:
                X_sec = np.vstack((X_sec, segment[np.newaxis]))
            except:
                X_sec = segment[np.newaxis] # X_sec = np.array(1, segment[np.newaxis])
            Y_sec.append(Y[count])

    return X_sec, Y_sec


def convolutional_split(X, Y, samples_to_jump_by, trial_len, num_channels):
    '''
    Func to split the training data in a convolutional manner
    X: your training data
    Y: your labels for training data
    samples_to_jump_by: the number of samples to skip b/w each trial. For ex,
    if trial_len is 100, then a good num for this may be 10.
    trial_len: desired number of samples for each trial
    num_channels: the number of electrodes on the cap
    '''

    # can either combine all training data into a single trial
    # or i can just split the data in the way that it is now
    # DECISION: just use it the way it is now, may result in a better
    # accuracy
    X_mod = np.empty((0, num_channels, trial_len))
    Y_mod = []

    count = -1
    for trial in X:
        start_idx = 0
        end_idx = trial_len
        count += 1
        while True:
            try:
                # segment the array and append to X_mod
                segment = trial[:, start_idx:end_idx]
                if len(segment[1]) < trial_len:
                    break
                try:
                    X_mod = np.vstack((X_mod, segment[np.newaxis]))
                except:
                    X_mod = segment[np.newaxis]
                Y_mod.append(Y[count])

                start_idx += samples_to_jump_by
                end_idx += samples_to_jump_by
            except:
                break

    return X_mod, Y_mod
