"""
Example linear baseline model.
- load data of one subject
- split to train and test sets
- mean variance normalize
- train the linear backward model
- reconstruct the envelope of the test eeg and report correlation measure
"""

import pickle
import glob
import os
import numpy as np
import json
from scipy.stats import pearsonr
from task2_regression.util import time_lags, normalize
from task2_regression.models import linear


def prepare_data(eeg_path, envelope_dir):
    """"
    Prepares eeg and envelope data to be used by linear backward model

    parameters
    ----------
    eeg_path: str
        path to a pickle file containing eeg data
    envelope_dir: str
        Directory containing envelope of speech files

    returns
    -------
    tuple (numpy.ndarray, numpy.ndarray)
        A tuple containing (time lag EEG, envelope of speech)
    """

    data = pickle.load(open(eeg_path, 'rb'))
    eeg = data['eeg']

    # Load the corresponding envelope based on stimulus name
    stimulus = data['stimulus']
    stimulus = stimulus.split('.')[0]
    speech = np.load(os.path.join(envelope_dir, stimulus + '.npz'))
    envelope = speech['env']

    # Make sure eeg and envelope have the same length
    minimum_length = min(eeg.shape[0], len(envelope))
    eeg = eeg[:minimum_length]
    envelope = envelope[:minimum_length]

    # Mean variance normalize data
    mean_var_nor = normalize.MeanVarNor(eeg, envelope)
    eeg, envelope = mean_var_nor(eeg, envelope)

    # Construct the time lag EEG (context window of 400 ms)
    eeg, envelope = time_lags.time_lag_matrix(eeg, envelope=envelope, num_lags=26)
    return eeg, envelope

# Envelope data directory
# Go one directory up
os.chdir("..")
envelope_dir = os.path.join(os.getcwd(), 'create_data', 'data_dir', 'train_dir', 'envelope')

# Preprocessed EEG data directory
# The path to the challenge dataset is in /util/dataset_root_dir.json
dataset_path_file = os.path.join(os.getcwd(), 'util', 'dataset_root_dir.json')
with open(dataset_path_file, 'r') as f:
    dataset_root_dir = json.load(f)
# Change back to the current dir
os.chdir('create_data')
preprocessed_eeg_dir = os.path.join(dataset_root_dir, 'train', 'preprocessed_eeg')

all_subjects = glob.glob(os.path.join(preprocessed_eeg_dir, 'sub*'))
# You can use indexing to select a subject of subjects
# for faster run
all_subjects = all_subjects[0:]

results_dict = {}
for subject in all_subjects:
    subject_name = subject.split('/')[-1]
    results_dict[subject_name] = {}
    all_recordings = glob.glob(os.path.join(subject, '*', 'sub*.pkl'))
    # Keep the last recording as test set and use the rest as train set
    train_recordings = all_recordings[:-1]
    test_recording = all_recordings[-1]
    eeg_train = []
    envelope_train = []
    for recording in train_recordings:
        eeg, envelope = prepare_data(recording, envelope_dir)
        eeg_train.append(eeg)
        envelope_train.append(envelope)

    # Concatenate all the recordings to one numpy array
    eeg_train = np.concatenate(eeg_train, axis=0)
    envelope_train = np.concatenate(envelope_train, axis=0)

    # Linear model
    model = linear.LinearBackwardModel(ridge_param=1)
    model.train(eeg_train, envelope_train)
    results_dict[subject_name]['model'] = model.get_model()
    del eeg_train, envelope_train

    # Test data
    eeg, envelope = prepare_data(test_recording, envelope_dir)
    envelope_reconstructed = model.predict(eeg)

    # pearsonr correlation as a measure of envelope reconstruction
    corr = pearsonr(envelope, envelope_reconstructed)[0]
    results_dict[subject_name]['corr'] = corr

# Save results
results_folder = os.path.join(os.getcwd(), 'results')
os.makedirs(results_folder, exist_ok=True)
with open(os.path.join(results_folder, 'results.pkl'), 'wb') as f:
    pickle.dump(results_dict, f)