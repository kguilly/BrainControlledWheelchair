'''
This is the first attempt at using a random forest to classify the EEG data
and rank the importance of the electrodes

# first need to perform feature extraction
    # need to translate the 3d array into a 2d array
    # spectral analysis? mean, variance, and other domain-specific features

'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import read_edf_files as ref
from scipy.signal import welch

def compute_psd(trail_data):
    f, psd = welch(trail_data, fs=160)
    return psd

kernels, chans = 1, 64
label_mapping = {
        1: "Rest",
        2: "Squeeze Both Fists",
        3: "Squeeze Both Feet",
        4: "Squeeze Left Hand",
        5: "Squeeze Right Hand",
    }

electrode_mapping = {
    1: '',
    2: '',
    3: '',
    4: '',
    5: '',
    6: '',
    7: '',
    8: '',
    9: '',
    10: '',
    11: '',
    12: '',
    13: '',
    14: '',
    15: '',
    16: '',
    17: '',
    18: '',
    19: '',
    20: '',
    21: '',
    22: '',
    23: '',
    24: '',
    25: '',
    26: '',
    27: '',
    28: '',
    29: '',
    30: '',
    11: '',
    12: '',
    13: '',
    14: '',
    15: '',
    16: '',
    17: '',
    18: '',
    19: '',
    20: '',
}

X, Y = ref.reader(passed_path='/home/kaleb/Documents/eeg_dataset/files/', patient_num=2)

# flatten X to be a 2d matrix of size (trials, channels[mean, variance, power spectral density])
################## USE ONLY SPECTRAL ANALYSIS FOR NOW  
psd_values = [] # power spectral density

for trial in X:
    trial_psd = [np.mean(compute_psd(channel_data)) for channel_data in trial]
    psd_values.append(trial_psd)

X_flattened = np.array(psd_values)



X_train, X_test, y_train, y_test = train_test_split(X_flattened, Y, stratify=Y, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_flattened, Y)


# feature importance based on mean decrease in impurity
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

feature_names = [f"electrode {i}" for i in range(X.shape[1])]

sorted_importances = importances[sorted_indices]
sorted_feature_names = np.array(feature_names)[sorted_indices]


std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)


forest_importances = pd.Series(sorted_importances, index=sorted_feature_names)
top_16 = sorted_feature_names[:16]
print(top_16)

exit()
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()
print("done")

## for each subject, send through random forest

for i in range(1, 110): 
    X, Y = ref.reader(passed_path='/home/kaleb/Documents/eeg_dataset/files/', patient_num=i)

    # flatten X to be a 2d matrix of size (trials, channels[mean, variance, power spectral density])
    ################## USE ONLY SPECTRAL ANALYSIS FOR NOW  
    psd_values = [] # power spectral density

    for trial in X:
        trial_psd = [np.mean(compute_psd(channel_data)) for channel_data in trial]
        psd_values.append(trial_psd)

    X_flattened = np.array(psd_values)



    X_train, X_test, y_train, y_test = train_test_split(X_flattened, Y, stratify=Y, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_flattened, Y)


    # feature importance based on mean decrease in impurity
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    feature_names = [f"electrode {i}" for i in range(X.shape[1])]

    sorted_importances = importances[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]


    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)


    forest_importances = pd.Series(sorted_importances, index=sorted_feature_names)


# permutation feature importance 