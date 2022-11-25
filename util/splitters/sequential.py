import logging
import os

import numpy as np

from util.splitters.base import Splitter


class SequentialSplitter(Splitter):
    def __init__(
        self, split_data_folder, splits, split_names, normalizer=None, overwrite=False
    ):
        super(SequentialSplitter, self).__init__(
            split_data_folder, splits, split_names, normalizer
        )
        self.overwrite = overwrite

    def __call__(self, all_data_for_recording, subject, stimulus, shortest_length=None):
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
