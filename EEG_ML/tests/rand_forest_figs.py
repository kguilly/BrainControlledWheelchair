'''
This is test 0.2.3.4 of the Brain-Controlled Wheelchair Senior Design II
project. This test uses a random forest classifier and converts the electrode
data to tabular by find the power spectral density of each channel for each trial.
For each subject, feature importances are selected and tallied. The summation of the
feature importances for each subject are saved to the test_data directory of this repository

TEST FAILURE, use electrode selection from literature
'''
import os.path

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
    1: 'FC5',
    2: 'FC3',
    3: 'FC1',
    4: 'FCz',
    5: 'FC2',
    6: 'FC4',
    7: 'FC6',
    8: 'C5',
    9: 'C3',
    10: 'C1',
    11: 'Cz',
    12: 'C2',
    13: 'C4',
    14: 'C6',
    15: 'CP5',
    16: 'CP3',
    17: 'CP1',
    18: 'CPz',
    19: 'CP2',
    20: 'CP4',
    21: 'CP6',
    22: 'Fp1',
    23: 'Fpz',
    24: 'Fp2',
    25: 'AF7',
    26: 'AF3',
    27: 'AFz',
    28: 'AF4',
    29: 'AF8',
    30: 'F7',
    31: 'F5',
    32: 'F3',
    33: 'F1',
    34: 'Fz',
    35: 'F2',
    36: 'F4',
    37: 'F6',
    38: 'F8',
    39: 'FT7',
    40: 'FT8',
    41: 'T7',
    42: 'T8',
    43: 'T9',
    44: 'T10',
    45: 'TP7',
    46: 'TP8',
    47: 'P7',
    48: 'P5',
    49: 'P3',
    50: 'P1',
    51: 'Pz',
    52: 'P2',
    53: 'P4',
    54: 'P6',
    55: 'P8',
    56: 'PO7',
    57: 'PO3',
    58: 'POz',
    59: 'PO4',
    60: 'PO8',
    61: 'O1',
    62: 'Oz',
    63: 'O2',
    64: 'Iz',
}
## for each subject, send through random forest
subject = 81

feature_importance_df = pd.DataFrame(columns=['Electrode Name', 'Electrode Number', 'Feature Importance'])

X, Y = ref.reader(passed_path='/home/kaleb/Documents/eeg_dataset/files/', patient_num=subject)
X, Y = ref.split_by_second(X, Y, 160)

psd_values = [] # power spectral density
for trial in X:
    trial_psd = [np.mean(compute_psd(channel_data)) for channel_data in trial]
    psd_values.append(trial_psd)

X_flattened = np.array(psd_values)

# X_train, X_test, y_train, y_test = train_test_split(X_flattened, Y, stratify=Y, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_flattened, Y)


# feature importance based on mean decrease in impurity
importances = rf.feature_importances_
forest_importances = pd.Series(importances, index=[electrode_mapping[key] for key in sorted(electrode_mapping.keys())])
sorted_importances = forest_importances.sort_values(ascending=False)
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

fig, ax = plt.subplots()
sorted_importances.plot.bar(yerr=std,ax=ax)
ax.set_ylabel("Mean Decrease in Impurity")
ax.set_title(f"Subject {subject}")
fig.tight_layout()
fig.show()
plt.show()




