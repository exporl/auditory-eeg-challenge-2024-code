"""Code that standardizes data."""
import numpy as np

from util.normalizers.base import Normalizer


def standardize(data, mean=None, std=None, axis=0):
    """Standardize data by subtraction mean and dividing by std.

    Parameters
    ----------
    data : np.ndarray
        The data.
    mean : Optional[float]
        The mean of the data. If None, it will be computed
    std : Optional[float]
        The std of the data. If None, it will be computed
    axis : int
        Axis along which to standardize

    Returns
    -------
    np.ndarray
        Standardized data
    """
    if mean is None:
        mean = np.mean(data, axis=axis)
    if std is None:
        std = np.std(data, axis=axis)
    return (data - mean) / std, mean, std


class Standardizer(Normalizer):
    """Object that standardizes data."""
    def __init__(self):
        """Initialize the Standardizer."""
        self.precomputed_parameters = {}

    def __call__(self, feature_name, feature):
        """Standardize the data.

        Parameters
        ----------
        feature_name : str
            Name of the feature
        feature : np.ndarray
            Data of the feature

        Returns
        -------
        np.ndarray
            Standardized data (centered mean and std of 1)
        """
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
        """Signal that a new recording will be handled."""
        self.precomputed_parameters = {}
