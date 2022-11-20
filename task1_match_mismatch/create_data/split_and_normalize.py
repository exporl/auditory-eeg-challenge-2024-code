"""
- Load preprocessed eeg data and its corresponding stimulus from each subject
- Split data to train, validation, and test sets
- Mean variance normalize data
- Save the split data
"""


import os
import glob
import pickle
import numpy as np
import json
from task1_match_mismatch.util.normalize import MeanVarNor

# Dataset directories for eeg and stimulus (speech)
# The path to the challenge dataset is in /util/dataset_root_dir.json
os.chdir('..')
dataset_path_file = os.path.join(os.getcwd(), 'util', 'dataset_root_dir.json')
os.chdir('create_data')
with open(dataset_path_file, 'r') as f:
    dataset_root_dir = json.load(f)
preprocessed_eeg_dir = os.path.join(dataset_root_dir, 'train', 'preprocessed_eeg')

current_dir = os.getcwd()
speech_dir = os.path.join(current_dir, 'data_dir/train_dir/speech_features')

# Where to save the split data
save_data_dir = os.path.join(current_dir, 'data_dir', 'train_dir')
os.makedirs(save_data_dir, exist_ok=True)
os.makedirs(os.path.join(save_data_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(save_data_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(save_data_dir, 'test'), exist_ok=True)


def split(eeg, env, splits=(0.8, 0.1, 0.1)):
    """
    Splits data into 3 parts.
     1. train (default: 80 %)
     2. val   (default: 10 %)
     3. test  (default: 10 %)

    parameters
    ---------
    eeg: numpy.ndarray
        EEG data
    env: numpy.ndarray
        Envelope of speech
    splits: tuple (float, float, float)
        split ratios for train, validation, and test sets

    returns
    -------
        tuple(numpy.ndarray, ..., numpy.ndarray)
        (EEG for the train set, envelope for the train set,
        EEG for the validation set, EEG for the validation set,
        EEG for the test set, EEG for the test set).
    """

    min_len = np.minimum(eeg.shape[0], env.shape[0])

    # Train
    train_eeg = eeg[0:int(min_len*splits[0]), :]
    train_env = env[0:int(min_len*splits[0])]

    # Validation
    val_eeg = eeg[int(min_len*splits[0]):int(min_len*(splits[0]+splits[1])), :]
    val_env = env[int(min_len*splits[0]):int(min_len*(splits[0]+splits[1]))]

    # Test
    test_eeg = eeg[int(min_len*(splits[0]+splits[1])):int(min_len), :]
    test_env = env[int(min_len*(splits[0]+splits[1])):int(min_len)]

    return train_eeg, train_env, val_eeg, val_env, test_eeg, test_env


if __name__ == '__main__':
    all_subjects = glob.glob(os.path.join(preprocessed_eeg_dir, 'sub*'))
    # You can use indexing to select a subject of subjects
    # for faster run
    all_subjects = all_subjects[0:]

    # Loop over subjects
    for subject in all_subjects:
        all_recordings = glob.glob(os.path.join(subject, '*', 'sub*.pkl'))
        for recording in all_recordings:
            data = pickle.load(open(recording, 'rb'))
            eeg = data['eeg']

            # Find the stimulus name, so that you can load
            # the corresponding envelope cache
            stimulus = data['stimulus']
            stimulus = stimulus.split('.')[0]
            speech = np.load(os.path.join(speech_dir, stimulus + '.npz'))
            env = speech['env']

            # Split to train, validation, test
            train_eeg, train_env, val_eeg, val_env, test_eeg, test_env = split(eeg, env)

            # Mean variance normalization
            mean_var_nor = MeanVarNor(train_eeg, train_env)
            train_eeg, train_env = mean_var_nor(train_eeg, train_env)
            val_eeg, val_env = mean_var_nor(val_eeg, val_env)
            test_eeg, test_env = mean_var_nor(test_eeg, test_env)

            # Save the train, validation, and test sets in .npy format
            # Concatenate envelope as a last dimension to EEG
            train = np.concatenate([train_eeg, np.expand_dims(train_env, axis=1)], axis=1)
            val = np.concatenate([val_eeg, np.expand_dims(val_env, axis=1)], axis=1)
            test = np.concatenate([test_eeg, np.expand_dims(test_env, axis=1)], axis=1)
            name = data['subject'] + '_-_' + stimulus
            np.save(os.path.join(save_data_dir, 'train', name + '_-_train.npy'), train)
            np.save(os.path.join(save_data_dir, 'val', name + '_-_val.npy'), val)
            np.save(os.path.join(save_data_dir, 'test', name + '_-_test.npy'), test)







