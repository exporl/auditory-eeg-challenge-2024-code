"""Base code for normalization."""
import abc


class Normalizer:
    """Class that can normalize feature data."""
    @abc.abstractmethod
    def __call__(self, feature_name, feature):
        """Normalize feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature
        feature : np.ndarray
            Data of the feature

        Returns
        -------
        np.ndarray
            Normalized data for the feature
        """
        pass

    @abc.abstractmethod
    def new_recording(self):
        """Signal that a new recording will be handled."""
        pass
