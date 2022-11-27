"""Code for the dataset_generator for task1."""
import numpy as np
import tensorflow as tf

from util.dataset_generator import (
    DataGenerator,
    default_group_key_fn,
    default_feature_sort_function,
)


@tf.function
def default_batch_equalizer_fn(*args):
    """Batch equalizer.
    Prepares the inputs for a model to be trained in
    match-mismatch task. It makes sure that match_env
    and mismatch_env are equally presented as a first
    envelope in match-mismatch task.

    Parameters
    ----------
    args : Sequence[tf.Tensor]
        List of tensors representing feature data

    Returns
    -------
    Tuple[Tuple[tf.Tensor], tf.Tensor]
        Tuple of the EEG/speech features serving as the input to the model and
        the labels for the match/mismatch task

    Notes
    -----
    This function will also double the batch size. E.g. if the batch size of
    the elements in each of the args was 32, the output features will have
    a batch size of 64.
    """
    eeg = args[0]
    new_eeg = tf.concat([eeg, eeg], axis=0)
    all_features = [new_eeg]
    for match, mismatch in zip(args[1::2], args[2::2]):
        stimulus_feature1 = tf.concat([match, mismatch], axis=0)
        stimulus_feature2 = tf.concat([mismatch, match], axis=0)
        all_features += [stimulus_feature1, stimulus_feature2]
    labels = tf.concat(
        [
            tf.tile(tf.constant([[0]]), [tf.shape(eeg)[0], 1]),
            tf.tile(tf.constant([[1]]), [tf.shape(eeg)[0], 1]),
        ],
        axis=0,
    )

    # print(new_eeg.shape, env1.shape, env2.shape, labels.shape)
    return tuple(all_features), labels


class MatchMismatchDataGenerator(DataGenerator):
    """Generate data."""

    def __init__(
        self,
        files,
        window_length,
        group_key_fn=default_group_key_fn,
        feature_sort_fn=default_feature_sort_function,
        as_tf_tensors=True,
        spacing=64,
    ):
        """Initialize the DataGenerator.

        Parameters
        ----------
        files: Sequence[Union[str, pathlib.Path]]
            Files to load.
        window_length: int
            Length of the decision window.
        group_key_fn: Callable[[Union[str, pathlib.Path]], str]
            Function that creates group keys from file names. These group
            keys will be used to group files of the same recording together
        feature_sort_fn: Callable[[str], str]
            Sorting function for feature names
        as_tf_tensors: bool
            Whether to cast resulting numpy arrays to tf.Tensor objects
        spacing: int
            Spacing between the matched and mismatched segment in samples
        """
        super().__init__(
            files, window_length, group_key_fn, feature_sort_fn, as_tf_tensors
        )
        self.spacing = spacing

    @property
    def nb_features(self):
        """Count the number of features of one recording of this dataset."""
        return 2 * len(self.files[0]) - 1

    @property
    def feature_dims(self):
        """Get the dimensions of the features of each recording."""
        eeg_dim = np.load(self.files[0][0]).shape[-1]
        speech_feature_dims = []
        for speech_feature_path in self.files[0][1:]:
            speech_feature_dims += [np.load(speech_feature_path).shape[-1]] * 2
        return [eeg_dim] + speech_feature_dims

    def prepare_data(self, data):
        """Creates mismatch (imposter) envelope.

        Parameters
        ----------
        data: Sequence[numpy.ndarray]
            Data to create an imposter for.

        Returns
        -------
        tuple (numpy.ndarray, numpy.ndarray, numpy.ndarray, ...)
            (EEG, matched stimulus feature, mismatched stimulus feature, ...).
        """
        data = super(MatchMismatchDataGenerator, self).prepare_data(data)
        eeg = data[0]
        new_length = eeg.shape[0] - self.window_length - self.spacing
        resulting_data = [eeg[:new_length, ...]]
        for stimulus_feature in data[1:]:
            match_feature = stimulus_feature[:new_length, ...]
            mismatch_feature = stimulus_feature[
                self.spacing + self.window_length:, ...
            ]
            resulting_data += [match_feature, mismatch_feature]
        return resulting_data
