"""Run the default preprocessing pipeline on sparrKULee.
This script runs the necessary prereprocessing steps on the sparrKULee dataset, starting from the MFW caches,
to arrive at the fully preprocessed files.
The MWF caches are downloaded from the challenge website and should be placed in the folder specified by the
raw_eeg_dir variable.
The preprocessed EEG will be saved in the folder specified by the preprocessed_eeg_dir variable.
The caches have been synchronized with the stimulus data and should all have a length of 5 seconds.

On the MWF caches, the following preprocessing steps have been performed:
( see the original SparrKULee.py file for reference as to what these steps do)
 eeg_steps = [
        LinkStimulusToBrainResponse(
            stimulus_data=stimulus_steps,
            extract_stimuli_information_fn=BIDSAPRStimulusInfoExtractor(),
            grouper=BIDSStimulusGrouper(
                bids_root=root_dir,
                mapping={"stim_file": "stimulus_path", "trigger_file": "trigger_path"},
                subfolders=["stimuli", "eeg"],
            ),
        ),
        LoadEEGNumpy(unit_multiplier=1e6, channels_to_select=list(range(64))),
        SosFiltFilt(
            scipy.signal.butter(1, 0.5, "highpass", fs=1024, output="sos"),
            emulate_matlab=True,
            axis=1,
        ),
        InterpolateArtifacts(),
        AlignPeriodicBlockTriggers(biosemi_trigger_processing_fn),
        SplitEpochs(),
        ArtifactRemovalMWF(),
        DefaultSave(after_wiener_filter_dir,
                    {'eeg': 'data'},
                    filename_fn=bids_filename_fn,
                    clear_output=True,
                    overwrite=overwrite),
    ]



"""
import argparse
import datetime
import gzip
import json
import logging
import os
from typing import Any, Dict, Sequence

import librosa
import numpy as np
import math
import scipy.signal.windows
from brain_pipe.dataloaders.path import GlobLoader
from brain_pipe.pipeline.default import DefaultPipeline
from brain_pipe.preprocessing.brain.artifact import (
    InterpolateArtifacts,
    ArtifactRemovalMWF,
)
from brain_pipe.preprocessing.brain.eeg.biosemi import (
    biosemi_trigger_processing_fn,
)
from brain_pipe.preprocessing.brain.eeg.load import LoadEEGNumpy
from brain_pipe.preprocessing.brain.epochs import SplitEpochs
from brain_pipe.preprocessing.brain.link import (
    LinkStimulusToBrainResponse,
    BIDSStimulusInfoExtractor,
)
from brain_pipe.preprocessing.brain.rereference import CommonAverageRereference
from brain_pipe.preprocessing.brain.trigger import (
    AlignPeriodicBlockTriggers,
)
from brain_pipe.preprocessing.filter import SosFiltFilt
from brain_pipe.preprocessing.resample import ResamplePoly
from brain_pipe.preprocessing.stimulus.audio.envelope import GammatoneEnvelope
from brain_pipe.preprocessing.stimulus.audio.spectrogram import LibrosaMelSpectrogram

from brain_pipe.preprocessing.stimulus.load import LoadStimuli
from brain_pipe.runner.default import DefaultRunner
from brain_pipe.save.default import DefaultSave
# from mel import DefaultSave
from brain_pipe.utils.log import default_logging, DefaultFormatter
from brain_pipe.utils.path import BIDSStimulusGrouper

from typing import Dict, Any, Sequence, Optional, Union, Mapping

import numpy as np

from brain_pipe.pipeline.base import PipelineStep
import glob

class LoadEEGNumpyTest(PipelineStep):
    """Load EEG data.

    This step uses MNE to load EEG data.
    """

    def __init__(
            self, keys={"data_path": "data"}, copy_data_dict=False, *mne_args, **mne_kwargs
    ):
        """Create a new LoadEEG instance.

        Parameters
        ----------
        eeg_path_key: str
            The key of the EEG path in the data dict.
        eeg_data_key: str
            The key of the EEG data in the data dict.
        """
        super().__init__(copy_data_dict=copy_data_dict)
        self.keys = self.parse_dict_keys(keys, "keys")
        self.mne_args = mne_args
        self.mne_kwargs = mne_kwargs


    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Load EEG data from a npy file.

        Parameters
        ----------
        data_dict: Dict[str, Any]
            The data dict containing the EEG path.

        Returns
        -------
        Dict[str, Any]
            The data dict with the EEG data and the EEG info.
        """
        for from_key, to_key in self.keys.items():
            path = data_dict[from_key]

            # Support for gzipped files.
            raw =np.load(path)
            # swap axes
            raw = np.swapaxes(raw, 0, 1)


            data_dict['data'] = raw
            data_dict['eeg_key'] = os.path.basename(path)

            data_dict['data_fs'] = 1024

        return data_dict

class BIDSAPRStimulusInfoExtractor(BIDSStimulusInfoExtractor):
    """Extract BIDS compliant stimulus information from an .apr file."""

    def __call__(self, brain_dict: Dict[str, Any]):
        """Extract BIDS compliant stimulus information from an events.tsv file.

        Parameters
        ----------
        brain_dict: Dict[str, Any]
            The data dict containing the brain data path.

        Returns
        -------
        Sequence[Dict[str, Any]]
            The extracted event information. Each dict contains the information
            of one row in the events.tsv file
        """
        event_info = super().__call__(brain_dict)
        # Find the apr file
        path = brain_dict[self.brain_path_key]
        apr_path = "_".join(path.split("_")[:-1]) + "_eeg.apr"
        # Read apr file
        apr_data = self.get_apr_data(apr_path)
        # Add apr data to event info
        for e_i in event_info:
            e_i.update(apr_data)
        return event_info

    def get_apr_data(self, apr_path: str):
        """Get the SNR from an .apr file.

        Parameters
        ----------
        apr_path: str
            Path to the .apr file.

        Returns
        -------
        Dict[str, Any]
            The SNR.
        """
        import xml.etree.ElementTree as ET

        apr_data = {}
        tree = ET.parse(apr_path)
        root = tree.getroot()

        # Get SNR
        interactive_elements = root.findall(".//interactive/entry")
        for element in interactive_elements:
            description_element = element.find("description")
            if description_element.text == "SNR":
                apr_data["snr"] = element.find("new_value").text
        if "snr" not in apr_data:
            logging.warning(f"Could not find SNR in {apr_path}.")
            apr_data["snr"] = 100.0
        return apr_data


def test_filename_fn(data_dict, feature_name, set_name=None):
    """Default function to generate a filename for the data.

    Parameters
    ----------
    data_dict: Dict[str, Any]
        The data dict containing the data to save.
    feature_name: str
        The name of the feature.
    set_name: Optional[str]
        The name of the set. If no set name is given, the set name is not
        included in the filename.

    Returns
    -------
    str
        The filename.
    """


    eeg_key = data_dict['eeg_key']

    return eeg_key


def temp_unpack_data(data_path):
    data = dict(np.load(data_path))
    # save all keys, values in separate data path.
    for key, value in data.items():
        np.save(os.path.join(os.path.dirname(data_path) ,  key + '.npy'), value)


def run_preprocessing_pipeline(
        root_dir,
        preprocessed_eeg_dir,
        nb_processes=4,
        overwrite=False,
        log_path="sparrKULee.log",
):
    """Construct and run the preprocessing on SparrKULee.

    Parameters
    ----------
    root_dir: str
        The root directory of the dataset.
    preprocessed_eeg_dir:
        The directory where the preprocessed EEG should be saved.
    nb_processes: int
        The number of processes to use. If -1, the number of processes is
        automatically determined.
    overwrite: bool
        Whether to overwrite existing files.
    log_path: str
        The path to the log file.
    """
    #########
    # PATHS #
    #########
    os.makedirs(preprocessed_eeg_dir, exist_ok=True)

    ###########
    # LOGGING #
    ###########
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(DefaultFormatter())
    default_logging(handlers=[handler])

    ################
    # DATA LOADING #
    ################
    logging.info("Retrieving layout...")
    data_loader = GlobLoader(
        [os.path.join(root_dir,   "sub*.npy")],
        filter_fns=[],
        key="data_path",
    )

    #########################
    # RUNNING THE PIPELINE  #
    #########################

    logging.info("Starting with the EEG preprocessing")
    logging.info("===================================")

    eeg_steps = [
        LoadEEGNumpyTest(),
        CommonAverageRereference(),
        ResamplePoly(64, axis=1),
        DefaultSave(
            preprocessed_eeg_dir,
            {"eeg": "data"},
            overwrite=overwrite,
            clear_output=True,
            filename_fn=test_filename_fn,
        ),
    ]

    #########################
    # RUNNING THE PIPELINE  #
    #########################

    logging.info("Starting with the EEG preprocessing")
    logging.info("===================================")

    # Create data_dicts for the EEG files
    # Create the EEG pipeline
    eeg_pipeline = DefaultPipeline(steps=eeg_steps)

    DefaultRunner(
        nb_processes=nb_processes,
        logging_config=lambda: None,
    ).run(
        [(data_loader, eeg_pipeline)],

    )


if __name__ == "__main__":
    # Load the config
    # get the top folder of the dataset
    challenge_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(challenge_folder, 'util', 'config.json'), "r") as f:
        config = json.load(f)

    # Set the correct paths as default arguments
    dataset_folder = config["dataset_folder"]
    test_folder = os.path.join(dataset_folder, config["test_folder"])
    task = 'TASK1_match_mismatch'  # [' TASK1_match_mismatch', 'TASK2_regression']

    preprocessed_eeg_folder = os.path.join(
        test_folder, task, f'{config["preprocessed_eeg_folder"]}'
    )
    raw_eeg_dir = os.path.join(test_folder, task, 'MWFilter_eeg')
    # unpack the data

    raw_eeg_data = glob.glob(os.path.join(raw_eeg_dir, '*_mwf.npz'))
    for data_path in raw_eeg_data:
        print(f'processing {data_path}')
        temp_unpack_data(data_path)

    default_log_folder = os.path.dirname(os.path.abspath(__file__))
    # Parse arguments from the command line
    parser = argparse.ArgumentParser(description="Preprocess the auditory EEG dataset")
    parser.add_argument(
        "--nb_processes",
        type=int,
        default=1,
        help="Number of processes to use for the preprocessing. "
             "The default is to use all available cores (-1).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--log_path", type=str, default=os.path.join(
            default_log_folder,
            "sparrKULee_{datetime}.log"
        )
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default=raw_eeg_dir,
        help="Path to the folder where the dataset is downloaded",
    )

    parser.add_argument(
        "--preprocessed_raw_eeg_path",
        type=str,
        default=preprocessed_eeg_folder,
        help="Path to the folder where the preprocessed EEG will be saved",
    )
    args = parser.parse_args()

    # Run the preprocessing pipeline
    run_preprocessing_pipeline(
        args.dataset_folder,
        args.preprocessed_raw_eeg_path,
        args.nb_processes,
        args.overwrite,
        args.log_path.format(
            datetime=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
    )