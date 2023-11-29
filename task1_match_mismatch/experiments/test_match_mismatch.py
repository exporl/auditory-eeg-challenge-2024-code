"""
Sample code to generate labels for test dataset of
match-mismatch task. The requested format for submitting the labels is
as follows:
for each subject a json file containing a python dictionary in the
format of  ==> {'sample_id': prediction, ... }.

"""

import os
import glob
import json
import numpy as np
import glob
import json
import logging
import os, sys
import tensorflow as tf

import sys
# add base path to sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from task1_match_mismatch.models.dilated_convolutional_model import dilation_model

from util.dataset_generator import DataGenerator, batch_equalizer_fn, create_tf_dataset





if __name__ == '__main__':

    # Parameters
    # Length of the decision window
    window_length_s = 5
    fs = 64

    window_length = window_length_s * fs  # 5 seconds
    # Hop length between two consecutive decision windows
    hop_length = 64

    epochs = 100
    patience = 5
    batch_size = 64
    number_mismatch = 4  # or 4

    # Get the path to the config gile
    experiments_folder = os.path.dirname(__file__)
    task_folder = os.path.dirname(experiments_folder)
    util_folder = os.path.join(os.path.dirname(task_folder), "util")
    config_path = os.path.join(util_folder, 'config.json')

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test
    data_folder = os.path.join(config["dataset_folder"],  config["test_folder"], 'TASK1_match_mismatch')
    eeg_folder = os.path.join(data_folder, 'preprocessed_eeg')
    stimulus_folder = os.path.join(data_folder, 'stimulus')

    # # stimulus feature which will be used for training the model. Can be either 'envelope' ( dimension 1) or 'mel' (dimension 28)
    # stimulus_features = ["envelope"]
    # stimulus_dimension = 1

    # uncomment if you want to train with the mel spectrogram stimulus representation
    stimulus_features = ["mel"]
    stimulus_dimension = 10

    features = ["eeg"] + stimulus_features

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder,
                                  "results_dilated_convolutional_model_{}_MM_{}_s_{}".format(number_mismatch,
                                                                                             window_length_s,
                                                                                             stimulus_features[0]))

    # create dilation model
    model = dilation_model(time_window=window_length, eeg_input_dimension=64, env_input_dimension=stimulus_dimension,
                           num_mismatched_segments=number_mismatch)

    model_path = os.path.join(results_folder,
                              "model_{}_MM_{}_s_{}.h5".format(number_mismatch, window_length_s, stimulus_features[0]))
    model.load_weights(model_path)



    test_eeg_mapping = glob.glob(os.path.join(data_folder, 'sub*mapping.json'))

    test_stimuli = glob.glob(os.path.join(stimulus_folder, f'*{stimulus_features[0]}*chunks.npz'))

    #load all test stimuli
    test_stimuli_data = {}
    for stimulus_path in test_stimuli:
        test_stimuli_data = dict(test_stimuli_data, **np.load(stimulus_path))

    for sub_stimulus_mapping in test_eeg_mapping:
        subject = os.path.basename(sub_stimulus_mapping).split('_')[0]

        # load stimulus mapping
        sub_stimulus_mapping = json.load(open(sub_stimulus_mapping))

        #load eeg data
        sub_path = os.path.join(eeg_folder, f'{subject}_eeg.npz')
        sub_eeg_data = dict(np.load(sub_path))



        data_eeg =  np.stack([[sub_eeg_data[value['eeg']]]  for key, value in sub_stimulus_mapping.items() ])
        # change dim 0 and 1 of eeg and unstack
        data_eeg = np.swapaxes(data_eeg, 0, 1)
        data_eeg = list(data_eeg)

        data_stimuli = np.stack([[test_stimuli_data[x] for x in value['stimulus']] for key, value in sub_stimulus_mapping.items()])
        # change dim 0 and 1 of stimulus and unstack
        data_stimuli = np.swapaxes(data_stimuli, 0, 1)
        data_stimuli = list(data_stimuli)

        id_list= list(sub_stimulus_mapping.keys())


        predictions = model.predict(data_eeg + data_stimuli)
        labels = np.argmax(predictions, axis=1)

        sub = dict(zip(id_list, [int(x) for x in labels]))

        prediction_dir = os.path.join(os.path.dirname(__file__), 'predictions')
        os.makedirs(prediction_dir, exist_ok=True)
        with open(os.path.join(prediction_dir, subject + '.json'), 'w') as f:
            json.dump(sub, f)


