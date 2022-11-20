""" This module contains linear backward model"""

import numpy as np


class LinearBackwardModel:
    """A class used to represent a linear backward model.

    attributes
    ----------
    ridge_param: float
        Lambda parameter in ridge regression
    model: numpy.ndarray
        Linear backward model

    methods
    -------
    train(eeg, envelope)
        Finds the analytical solution for the linear backward model
    """

    def __init__(self, ridge_param):
        self.ridge_param = ridge_param
        self.model = None

    def train(self, eeg, envelope):
        """Finds the analytical solution for the linear backward model.
        The solution is based on minimum least squares error.

        parameters
        ----------
        eeg: numpy.ndarray
            EEG data with shape (n_samples, n_channels)
        envelope: numpy.ndarray
            Envelope of speech with shape (n_samples,)

        """
        auto_corr = np.matmul(eeg.T, eeg) / np.size(eeg, 0)
        regression_matrix = np.eye(np.shape(eeg)[1])
        auto_corr = auto_corr + self.ridge_param * regression_matrix
        cross_corr = np.matmul(eeg.T, envelope) / np.size(eeg, 0)
        self.model = np.linalg.solve(auto_corr, cross_corr)
        return

    def predict(self, eeg):
        """Reconstructs the stimulus envelope from a given EEG.

        parameters
        ----------
        eeg: numpy.ndarray
            EEG data for which we want to reconstruct the stimulus envelope for.
        """
        if self.model is None:
            raise ValueError('linear model is not trained yet! First, use train method to train the model.')
        else:
            return np.matmul(eeg, self.model)

    def get_model(self):
        """ Returns the linear model"""
        return self.model