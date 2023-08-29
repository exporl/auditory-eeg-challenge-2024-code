"""Split data in sets and normalize (per recording)."""
import glob
import json
import os
import pickle


import numpy as np


if __name__ == "__main__":

    # Arguments for splitting and normalizing
    speech_features = ['envelope', 'mel']
    splits = [80, 10, 10]
    split_names = ['train', 'val', 'test']
    overwrite = False

    # Calculate the split fraction
    split_fractions = [x/sum(splits) for x in splits]

    # Get the path to the config file
    task_folder = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(task_folder, 'util', 'config.json')

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Construct the necessary paths
    processed_eeg_folder = os.path.join(config["dataset_folder"],config['derivatives_folder'], f"{config['preprocessed_eeg_folder']}")
    processed_stimuli_folder = os.path.join(config["dataset_folder"],config['derivatives_folder'], f"{config['preprocessed_stimuli_folder']}")
    split_data_folder = os.path.join(config["dataset_folder"],config['derivatives_folder'], config["split_folder"])

    # Create the output folder
    os.makedirs(split_data_folder, exist_ok=True)

    # Find all subjects
    all_subjects = glob.glob(os.path.join(processed_eeg_folder, "sub*"))
    nb_subjects = len(all_subjects)
    print(f"Found {nb_subjects} subjects to split/normalize")

    # Loop over subjects
    for subject_index, subject_path in enumerate(all_subjects):
        subject = os.path.basename(subject_path)
        print(f"Starting with subject {subject} ({subject_index + 1}/{nb_subjects})...")
        # Find all recordings
        all_recordings = glob.glob(os.path.join(subject_path, "*", "*.npy"))
        print(f"\tFound {len(all_recordings)} recordings for subject {subject}.")
        # Loop over recordings
        for recording_index, recording in enumerate(all_recordings):
            print(f"\tStarting with recording {recording} ({recording_index + 1}/{len(all_recordings)})...")

            # Load EEG from disk
            print(f"\t\tLoading EEG for {recording}")
            eeg = np.load(recording)

            # swap axes to have time as first dimension
            eeg = np.swapaxes(eeg, 0, 1)

            # keep only the 64 channels
            eeg = eeg[:, :64]

            # retrieve the stimulus name from the filename
            stimulus_filename = recording.split('_eeg.')[0].split('-audio-')[1]

            # Retrieve EEG data and pointer to the stimulus
            shortest_length = eeg.shape[0]

            # Create mapping between feature name and feature data
            all_data_for_recording = {"eeg": eeg}

            # Find corresponding stimuli for the EEG recording
            for feature_name in speech_features:
                # Load feature from disk
                print(f"\t\tLoading {feature_name} for recording {recording} ")
                stimulus_feature_path = os.path.join(
                    processed_stimuli_folder,
                    stimulus_filename + "_-_" + feature_name + ".npy",
                )
                feature = np.load(stimulus_feature_path)
                # Calculate the shortest length
                shortest_length = min(feature.shape[0], shortest_length)
                # Update all_data_for_recording
                all_data_for_recording[feature_name] = feature

            # Do the actual splitting
            print(f"\t\tSplitting/normalizing recording {recording}...")
            for feature_name, feature in all_data_for_recording.items():
                start_index = 0
                feature_mean = None
                feature_std = None

                for split_name, split_fraction in zip(split_names, split_fractions):
                    end_index = start_index + int(shortest_length * split_fraction)

                    # Cut the feature to the shortest length
                    cut_feature = feature[start_index:end_index, ...]

                    # Normalize the feature
                    if feature_mean is None:
                        feature_mean = np.mean(cut_feature, axis=0)
                        feature_std = np.std(cut_feature, axis=0)
                    norm_feature = (cut_feature - feature_mean)/feature_std

                    # Save the feature
                    save_filename = f"{split_name}_-_{subject}_-_{stimulus_filename}_-_{feature_name}.npy"
                    save_path = os.path.join(split_data_folder, save_filename)
                    if not os.path.exists(save_path) or overwrite:
                        np.save(save_path, cut_feature)
                    else:
                        print(f"\t\tSkipping {save_filename} because it already exists")
                    start_index = end_index