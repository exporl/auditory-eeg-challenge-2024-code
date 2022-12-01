"""Code for the dataset_generator for task2."""
import itertools
import os

import numpy as np
import tensorflow as tf


def create_tf_dataset(
    data_generator,
    window_length,
    batch_equalizer_fn=None,
    hop_length=64,
    batch_size=64,
    data_types=(tf.float32, tf.float32),
    feature_dims=(64, 1)
):
    """Creates a tf.data.Dataset.

    This will be used to create a dataset generator that will
    pass windowed data to a model in both tasks.

    Parameters
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
    data_types: Union[Sequence[tf.dtype], tf.dtype]
        The data types that the individual features of data_generator should
        be cast to. If you only specify a single datatype, it will be chosen
        for all EEG/speech features.

    Returns
    -------
    tf.data.Dataset
        A Dataset object that generates data to train/evaluate models
        efficiently
    """
    # create tf dataset from generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=tuple(
            tf.TensorSpec(shape=(None, x), dtype=data_types[index])
            for index, x in enumerate(feature_dims)
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





class RegressionDataGenerator:
    """Generate data for the regression task."""

    def __init__(
        self,
        files,
        window_length,
    ):
        """Initialize the DataGenerator.

        Parameters
        ----------
        files: Sequence[Union[str, pathlib.Path]]
            Files to load.
        window_length: int
            Length of the decision window.
        """
        self.window_length = window_length
        self.files = self.group_recordings(files)

    def group_recordings(self, files):
        """Group recordings and corresponding stimuli.

        Parameters
        ----------
        files : Sequence[Union[str, pathlib.Path]]
            List of filepaths to preprocessed and split EEG and speech features

        Returns
        -------
        list
            Files grouped by the self.group_key_fn and subsequently sorted
            by the self.feature_sort_fn.
        """
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

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

        return tuple(tf.constant(x) for x in data)


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

