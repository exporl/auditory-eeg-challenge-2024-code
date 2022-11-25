import numpy as np

from util.normalizers.base import Normalizer


def standardize(data, mean=None, std=None, axis=0):
    if mean is None:
        mean = np.mean(data, axis=axis)
    if std is None:
        std = np.std(data, axis=axis)
    return (data - mean) / std, mean, std


class Standardizer(Normalizer):
    def __init__(self):
        self.precomputed_parameters = {}

    def __call__(self, feature_name, feature):
        if feature_name not in self.precomputed_parameters:
            norm_feature, mean, std = standardize(feature)
            self.precomputed_parameters[feature_name] = {
                "mean": mean,
                "std": std
            }
        else:
            parameters = self.precomputed_parameters[feature_name]
            norm_feature, _, _ = standardize(feature, **parameters)
        return norm_feature

    def new_recording(self):
        self.precomputed_parameters = {}
