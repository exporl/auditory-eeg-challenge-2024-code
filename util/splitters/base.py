import abc
import logging


class Splitter(abc.ABC):
    def __init__(self, split_data_folder, splits, split_names, normalizer=None):
        self.split_data_folder = split_data_folder
        self.splits = self.get_split_fraction(splits)
        self.split_names = split_names
        if normalizer is None:
            logging.warning("No normalization will be applied.")
        self.normalizer = normalizer

    @staticmethod
    def get_split_fraction(splits):
        return [x / sum(splits) for x in splits]

    @abc.abstractmethod
    def __call__(self, all_data_for_recording, subject, stimulus, shortest_length=None):
        pass
