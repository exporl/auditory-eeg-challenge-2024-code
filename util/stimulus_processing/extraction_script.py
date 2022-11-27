"""Script to extract speech features."""
import glob
import logging
import os

import numpy as np

from util.config import load_config


def extract_speech_features(config_path, feature_dict, overwrite=False):
    """Extract the speech features.

    Parameters
    ----------
    config_path : Union[str, pathlib.Path]
        Path to a config file.
    feature_dict : Mapping[str, FeatureExtractor]
        Dictionary that maps a feature name to a FeatureExtractor method
    overwrite : bool
        Whether to overwrite already existing speech features with the same
        name.
    """
    logging.info(f'Extracting features {",".join(feature_dict.keys())}')

    # Load the config for this task
    config = load_config(config_path)
    # Get the downloaded data directory
    dataset_folder = config["dataset_folder"]

    source_stimuli_folder = os.path.join(dataset_folder, config["raw_stimuli_folder"])
    # Get the path to save the preprocessed files
    output_stimuli_folder = os.path.join(
        dataset_folder, config["preprocessed_stimuli_folder"]
    )

    # Create the save directory if it didn't exist already
    os.makedirs(output_stimuli_folder, exist_ok=True)

    # Find the stimuli file
    speech_files = glob.glob(os.path.join(source_stimuli_folder, "*.npz"))

    # Preprocess the stimuli
    nb_speech_files = len(speech_files)
    logging.info("Found %u stimuli files", nb_speech_files)
    for index, filepath in enumerate(speech_files):
        # Loop over each speech file and create envelope and mel spectrogram
        # and save them
        filename = os.path.basename(filepath)

        # If the cache already exists then skip
        logging.info(f"Preprocessing {filepath} ({index+1}/{nb_speech_files})")

        # Calculate the relevant features
        for feature_name, feature_extractor in feature_dict.items():
            save_path = os.path.join(
                output_stimuli_folder,
                filename.replace(".npz", "_" + feature_name + ".npy"),
            )
            if os.path.exists(save_path) and not overwrite:
                logging.info(
                    f"Skipping extracting feature {feature_name} as "
                    f"{save_path}  already exists (use the "
                    f'"overwrite" flag to overwrite already existing files)',
                )
                continue

            feature = feature_extractor(filepath)
            # Save the feature
            np.save(save_path, feature)
            logging.info(f"\tSaved at {save_path}")
