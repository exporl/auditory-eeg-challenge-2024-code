"""Generator tools to load data when training/evaluating models."""

import abc
import itertools
import os
from typing import Sequence

import numpy as np
import tensorflow as tf


def create_tf_dataset(
    data_generator,
    window_length,
    batch_equalizer_fn=None,
    hop_length=64,
    batch_size=64,
    data_types=tf.float32,
):
    """Creates a tf.data.Dataset.

    This will be used to create a dataset generator that will
    pass windowed data to a model in both tasks.

    parameters
    ---------
    data_generator: DataGenerator
        A data generator.
    window_length: int
        Length of the decision window in samples.
    batch_equalizer_fn: Callable
        Function that will be applied on the data after batching (using
        the `map` method from tf.data.Dataset). In the match/mismatch task,
        this function creates the imposter segments and labels.
    hop_length: int
        Hop length between two consecutive decision windows.
    batch_size: Optional[int]
        If not None, specifies the batch size. In the match/mismatch task,
        this amount will be doubled by the default_batch_equalizer_fn

    returns
    -------
    tf.data.Dataset
        A Dataset object that generates data to train/evaluate models
        efficiently
    """
    if not isinstance(data_types, Sequence):
        data_types = [data_types] * len(data_generator.feature_dims)
    # create tf dataset from generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=tuple(
            tf.TensorSpec(shape=(None, x), dtype=data_types[index])
            for index, x in enumerate(data_generator.feature_dims)
        ),
    )
    # window dataset
    dataset = dataset.map(
        lambda *args: [
            tf.signal.frame(arg, window_length, hop_length, axis=0)
            for arg in args
        ]
    )

    # batch data
    dataset = dataset.interleave(
        lambda *args: tf.data.Dataset.from_tensor_slices(args),
        cycle_length=4,
        block_length=16,
    )
    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    if batch_equalizer_fn is not None:
        # Create the labels and make sure classes are balanced
        dataset = dataset.map(batch_equalizer_fn)

    return dataset


def default_group_key_fn(path):
    """Groups paths based on their set, subject and stimulus.

    Parameters
    ----------
    path : Union[str, pathlib.Path]
        Path to a saved feature (already preprocessed and split).

    Returns
    -------
    str
        A key to group the files based on set, subject and stimulus
    """
    return "_-_".join(os.path.basename(path).split("_-_")[:3])


def default_feature_sort_function(feature_name):
    """Sort features by name.

    Parameters
    ----------
    feature_name : str
        The name of the current feature

    Returns
    -------
    str
        Key string that will be used for sorting
    """
    return "0" if "eeg" else feature_name


class DataGenerator(abc.ABC):
    """Python generator to load preprocessed and split datas"""

    def __init__(
        self,
        files,
        window_length,
        group_key_fn=default_group_key_fn,
        feature_sort_fn=default_feature_sort_function,
        as_tf_tensors=True,
    ):
        """Initialize the DataGenerator.

        Parameters
        ----------
        files: Sequence[Union[str, pathlib.Path]]
            Files to load.
        window_length: int
            Length of the decision window.
        group_key_fn: Callable[[Union[str, pathlib.Path]], str]



        """
        self.group_key_fn = group_key_fn
        self.feature_sort_fn = feature_sort_fn
        self.window_length = window_length
        self.as_tf_tensors = as_tf_tensors
        self.files = self.group_recordings(files)

    def group_recordings(self, files):
        new_files = []
        grouped = itertools.groupby(sorted(files), self.group_key_fn)
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=self.feature_sort_fn)]
        return new_files

    @property
    @abc.abstractmethod
    def nb_features(self):
        """Count the number of features of one recording of this dataset."""
        pass

    @property
    @abc.abstractmethod
    def feature_dims(self):
        """Get the dimensions of the features of each recording."""
        pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, recording_index):
        """Get data for a certain recording.

        Parameters
        ----------
        recording_index: int
            Index of the recording in this dataset

        Returns
        -------
        Union[Tuple[tf.Tensor,...], Tuple[np.ndarray,...]]
            The features corresponding to the recording_index recording
        """
        data = []
        for feature in self.files[recording_index]:
            data += [np.load(feature).astype(np.float32)]

        features = self.prepare_data(data)
        if self.as_tf_tensors:
            return tuple(tf.constant(x) for x in features)
        else:
            return features

    @abc.abstractmethod
    def prepare_data(self, data):
        """Modify data.

        Parameters
        ----------
        data: Sequence[np.ndarray]
            List of feature data for a certain recording

        Returns
        -------
        Sequence[np.ndarray]
            List of modified feature data for a certain recording.
        """
        return data

    def __call__(self):
        """Load data for the next recording.

        Yields
        -------
        Union[Tuple[tf.Tensor,...], Tuple[np.ndarray,...]]
            The features corresponding to the recording_index recording
        """
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)

            if idx == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        """Change state at the end of an epoch."""
        np.random.shuffle(self.files)
