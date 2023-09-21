import pyedflib
import numpy as np
import os


def map_to_samples(total_secs, onset_sec, next_onset_sec, total_samples):
    # return a 2 element array, start sample and end sample
    start_sample = int((onset_sec * total_samples) / total_secs)
    end_sample = int((next_onset_sec * total_samples) / total_secs)

    return [start_sample, end_sample]


def reader():
    label_mapping = {
        1: "Rest",
        2: "Squeeze Both Fists",
        3: "Squeeze Both Feet",
        4: "Squeeze Left Hand",
        5: "Squeeze Right Hand",
    }

    path = '/home/kaleb/Documents/eeg_dataset/files/S001'
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
    print("X.shape: ", X.shape)
    print("Y.shape: ", Y.shape)

    return X,Y