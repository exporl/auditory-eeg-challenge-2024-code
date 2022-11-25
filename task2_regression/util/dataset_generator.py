"""This module creates a dataset generator to be used in the regression task."""
import numpy as np

from util.dataset_generator import DataGenerator


class RegressionDataGenerator(DataGenerator):
    """Generate data."""

    @property
    def nb_features(self):
        return len(self.files)

    @property
    def feature_dims(self):
        eeg_dim = np.load(self.files[0][0]).shape[-1]
        speech_feature_dims = []
        for speech_feature_path in self.files[0][1:]:
            speech_feature_dims += [np.load(speech_feature_path).shape[-1]]
        return [eeg_dim] + speech_feature_dims

    def prepare_data(self, data):
        return super(RegressionDataGenerator, self).prepare_data(data)
