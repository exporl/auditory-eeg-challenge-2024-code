"""Feature extraction base class."""
import abc


class FeatureExtractor(abc.ABC):
    """Object that can extract features."""

    @abc.abstractmethod
    def __call__(self, audio_path):
        """Extract a feature from an audio_path.

        Parameters
        ----------
        audio_path : Union[str, pathlib.Path]
            Path to an audio file.

        Returns
        -------
        np.ndarray
            Feature in numpy array format.
        """
        pass
