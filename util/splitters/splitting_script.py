"""Script to split data per recording."""
import glob
import logging
import os
import pickle

import numpy as np


def split_per_recording(
    splitter, config, stimulus_features, cut_to_shortest_length=True
):
    """Split data per recording.

    Parameters
    ----------
    splitter : Splitter
        Class to do the actual splitting
    config : dict
        Config dictionary
    stimulus_features : Sequence[str]
        Names for the speech features to include.
    cut_to_shortest_length : bool
        Whether to cut all the features to the same length.
        WARNING: not doing this might lead to de-synchronization between the
        speech and EEG features
    """
    processed_eeg_folder = os.path.join(
        config["dataset_folder"],
        config["preprocessed_eeg_folder"],
    )
    processed_stimuli_folder = os.path.join(
        config["dataset_folder"],
        config["preprocessed_stimuli_folder"],
    )
    split_data_folder = os.path.join(
        config["dataset_folder"],
        config["split_folder"],
    )
    os.makedirs(split_data_folder, exist_ok=True)

    all_subjects = glob.glob(os.path.join(processed_eeg_folder, "sub*"))
    nb_subjects = len(all_subjects)
    logging.info(f"Found {nb_subjects} subjects to split/normalize")

    # Loop over subjects
    for subject_index, subject_path in enumerate(all_subjects):
        subject = os.path.basename(subject_path)
        logging.info(
            f"Starting with subject {subject} "
            f"({subject_index + 1}/{nb_subjects})..."
        )
        all_recordings = glob.glob(os.path.join(subject_path, "*", "sub*.pkl"))
        nb_recordings = len(all_recordings)
        logging.info(f"\tFound {nb_recordings} recordings for subject {subject}.")
        # Loop over recordings
        for recording_index, recording in enumerate(all_recordings):
            logging.info(
                f"\tStarting with recording {recording} "
                f"({recording_index + 1}/{nb_recordings})..."
            )
            # Load EEG from disk
            logging.info(f"\t\tLoading EEG for {recording}")
            with open(recording, "rb") as fp:
                data = pickle.load(fp)
            # Retrieve EEG data and pointer to the stimulus
            eeg, stimulus_filename = data["eeg"], data["stimulus"]
            shortest_length = eeg.shape[0]
            stimulus_filename_parts = stimulus_filename.split(".")
            # Create mapping between feature name and feature data
            all_data_for_recording = {"eeg": eeg}
            # Find corresponding stimuli for the EEG recording
            for feature_name in stimulus_features:
                # Load feature from disk
                logging.info(f"\t\tLoading {feature_name} for recording {recording} ")
                stimulus_feature_path = os.path.join(
                    processed_stimuli_folder,
                    stimulus_filename_parts[0] + "_" + feature_name + ".npy",
                )
                feature = np.load(stimulus_feature_path)
                # Calculate shortest length
                shortest_length = min(feature.shape[0], shortest_length)
                # Update all_data_for_recording
                all_data_for_recording[feature_name] = feature

            # Do the actual splitting
            logging.info(f"\t\tSplitting/normalizing recording {recording}...")
            splitter(
                all_data_for_recording,
                subject,
                stimulus_filename_parts[0],
                shortest_length if cut_to_shortest_length else None,
            )
