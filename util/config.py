"""Script to load the configuration for a certain task."""
import json
import os


def load_config(config_path):
    """Load a config file of a task.

    Parameters
    ----------
    config_path : Union[str, pathlib.Path]
        Path to the JSON config file.

    Returns
    -------
    dict
        Dictionary of the configuration options

    Raises
    ------
    ValueError
        If a required configuration option is not set.
    """

    # Open the config file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Check if the user filled out the dataset_root_dir
    if config.get("dataset_folder", None) is None:
        raise ValueError(
            "You have to add the path to the data you downloaded for the "
            f"challenge in '{config_path}' (the key should be "
            "'dataset_folder' and the value should be the path were you"
            " downloaded the dataset)."
        )

    if config.get("raw_eeg_folder", None) is None:
        raise ValueError(
            f"No 'raw_eeg_folder' in the config file at {config_path}. "
            f"Set this attribute to an appropriate folder name to store the "
            f"raw EEG. "
            f"(The folder will be created under {config['dataset_folder']})"
        )

    if config.get("raw_stimuli_folder", None) is None:
        raise ValueError(
            f"No 'raw_stimuli_folder' in the config file at {config_path}. "
            f"Set this attribute to an appropriate folder name to store the "
            f"raw stimuli files. "
            f"(The folder will be created under {config['dataset_folder']})"
        )

    if config.get("preprocessed_eeg_folder", None) is None:
        raise ValueError(
            f"No 'preprocessed_eeg_folder' in the config file at {config_path}"
            f". Set this attribute to an appropriate folder name to store the "
            f"preprocessed EEG. "
            f"(The folder will be created under {config['dataset_folder']})"
        )

    if config.get("preprocessed_stimuli_folder", None) is None:
        raise ValueError(
            f"No 'preprocessed_stimuli_folder' in the config file at "
            f"{config_path}. "
            f"Set this attribute to an appropriate folder name to store the "
            f"preprocessed stimuli. "
            f"(The folder will be created under {config['dataset_folder']})"
        )

    if config.get("split_folder", None) is None:
        raise ValueError(
            f"No 'split_folder' in the config file at {config_path}."
            f"Set this attribute to an appropriate folder name to store the "
            f"preprocessed, split and (optionally) normalized EEG and "
            f"stimulus data. "
            f"(The folder will be created under {config['dataset_folder']})"
        )
    return config


def check_config_path(config_path):
    """Check whether a config path exists (or give the default path).

    Parameters
    ----------
    config_path : Union[str, pathlib.Path]
        Path for the config file

    Returns
    -------
    Union[str, pathlib.Path]
        Path to the config file.

    Raises
    ------
    FileNotFoundError
        When no existing config path could be found.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.json",
        )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"You must specify an existing config.json file "
            f'("{config_path}" doesn\'t exist)'
        )
    return config_path
