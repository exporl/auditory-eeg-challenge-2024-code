"""Code to sequentially split data."""
import logging
import os

import numpy as np

from util.splitters.base import Splitter


class SequentialSplitter(Splitter):
    """Splitter that extracts sets sequentially from recordings."""
    def __init__(
        self, split_data_folder, splits, split_names, normalizer=None, overwrite=False
    ):
        """Initialize a SequentialSplitter.

        Parameters
        ----------
        split_data_folder : Union[str,pathlib.Path]
            Folder to eventually save the split data
        splits : Union[Sequence[int], Sequence[float]]
            A list of ints or floats signifying the relative size of each split.
            The first split will be regarded as the train split.
            The number of splits should correspond to the number of split_names
        split_names : Sequence[str]
            The names for each split.
            The first split will be regarded as the train split.
            The number of splits should correspond to the number of split_names
        normalizer : Optional[Normalizer]
            A normalizer that will normalize the data. If None, no normalization
            will be done.
        overwrite : bool
            Whether to overwrite already existing data.
        """
        super(SequentialSplitter, self).__init__(
            split_data_folder, splits, split_names, normalizer
        )
        self.overwrite = overwrite

    def __call__(self, all_data_for_recording, subject, stimulus, shortest_length=None):
        """Split data into sets.

        Parameters
        ----------
        all_data_for_recording : Mapping[str, np.ndarray]
            Mapping between the name of a feature and it's corresponding data
        subject : str
            Code of the subject
        stimulus : str
            Name of the stimulus
        shortest_length : Optional[int]
            The length of the shortest feature.
        """
        self.normalizer.new_recording()
        for feature_name, feature in all_data_for_recording.items():
            start_index = 0
            feature_length = feature.shape[0]
            if shortest_length is not None:
                feature_length = shortest_length

            for split_name, split_fraction in zip(self.split_names, self.splits):
                save_filename = (
                    f"{split_name}_-_{subject}_-_{stimulus}_-_{feature_name}.npy"
                )
                save_path = os.path.join(self.split_data_folder, save_filename)
                end_index = start_index + int(feature_length * split_fraction)

                cut_feature = feature[start_index:end_index, ...]
                if self.normalizer is not None:
                    cut_feature = self.normalizer(feature_name, cut_feature)

                if not os.path.exists(save_path) or self.overwrite:
                    np.save(save_path, cut_feature)
                else:
                    logging.info(
                        f"\t\tSkipping {save_filename} because it already "
                        f"exists (you can choose to overwrite these files "
                        f'by setting the "--overwrite" flag).'
                    )
                start_index = end_index
