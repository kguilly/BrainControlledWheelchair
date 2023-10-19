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

'''
X, Y = ref.reader(passed_path='/home/kaleb/Documents/eeg_dataset/files/', patient_num=1)
X, Y = ref.split_by_second(X, Y, 160)
# flatten X to be a 2d matrix of size (trials, channels[mean, variance, power spectral density])
################## USE ONLY SPECTRAL ANALYSIS FOR NOW  
psd_values = [] # power spectral density

for trial in X:
    trial_psd = [np.mean(compute_psd(channel_data)) for channel_data in trial]
    psd_values.append(trial_psd)

X_flattened = np.array(psd_values)



X_train, X_test, y_train, y_test = train_test_split(X_flattened, Y, stratify=Y, random_state=42)
rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)


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

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()


print("Now on to permutation importance analysis")
from sklearn.inspection import permutation_importance

r = permutation_importance(rf, X_train, y_train, n_repeats=30, random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] -2 * r.importances_std[i] > 0:
        print(feature_names[i], " ", r.importances_mean[i], " +/- ", r.importances_std[i])


print("################\nNow on to Lasso")
from sklearn.linear_model import LassoCV
lasso=LassoCV(max_iter=5000).fit(X_flattened, Y)

feature_importances = lasso.coef_
print(feature_importances)


exit()
'''

## for each subject, send through random forest
top_electrodes = np.empty((0, 16))
for i in range(1, 110): 
    X, Y = ref.reader(passed_path='/home/kaleb/Documents/eeg_dataset/files/', patient_num=i)
    X, Y = ref.split_by_second(X, Y, 160)
    
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

    feature_names = [int(i+1) for i in range(X.shape[1])]

    sorted_importances = importances[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]

    # grab the top 16 and store them in an array
    top_16 = sorted_feature_names[:16]
    top_electrodes = np.vstack((top_electrodes, top_16))


# go through top_electrodes and find the electrodes that occurred the most amount of times
flattened_top_electrodes = top_electrodes.flatten()
unique_electrodes, counts = np.unique(flattened_top_electrodes, return_counts=True)

sorted_indices = np.argsort(-counts)
top_16_electrodes = unique_electrodes[sorted_indices[:16]]
top_16_occurrences = counts[sorted_indices[:16]]

print("\n\n#########################\nTop 16 electrodes and their counts: ")
for electrode, occurrence in zip(top_16_electrodes, top_16_occurrences):
    print(f"Electrode {electrode} occurred {occurrence} times. ")

data = []
for electrode_num, occurrence in zip(unique_electrodes, counts):
    electrode_name = electrode_mapping.get(electrode_num)
    data.append([electrode_num, electrode_name, occurrence])

df = pd.DataFrame(data, columns=['Electrode Num', 'Electrode Name', 'Occurrences'])
df = df.sort_values(by='Occurrences', ascending=False)
df.to_csv('top_electrodes.csv', index=False)



