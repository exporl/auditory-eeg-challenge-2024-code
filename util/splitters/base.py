"""Base code for splitting data into sets."""
import abc
import logging


class Splitter(abc.ABC):
    """Base class for splitting data into sets."""
    def __init__(self, split_data_folder, splits, split_names, normalizer=None):
        """Initialize the splitter.

        Parameters
        ----------
        split_data_folder : Union[str,pathlib.Path]
            Path to the folder where the data should be put after being split
            into sets
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
        """
        self.split_data_folder = split_data_folder
        self.splits = self.get_split_fraction(splits)
        self.split_names = split_names
        if normalizer is None:
            logging.warning("No normalization will be applied.")
        self.normalizer = normalizer

    @staticmethod
    def get_split_fraction(splits):
        """Compute the correct split fractions.

        Parameters
        ----------
        splits : Union[Sequence[int], Sequence[float]]
            A list of ints or floats signifying the relative size of each split.
            The first split will be regarded as the train split.

        Returns
        -------
        Sequence[float]
            The normalized size of each set/split.
        """
        return [x / sum(splits) for x in splits]

    @abc.abstractmethod
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
        pass
