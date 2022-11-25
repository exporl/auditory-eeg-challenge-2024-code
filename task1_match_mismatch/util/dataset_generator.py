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
    eeg: tf.Tensor
        EEG data.
    match_env: tf.Tensor
        Matched envelope
    mismatch_env: tf.Tensor
        Mismatched envelope

    Returns
    -------
    tf.Tensor
        EEG data.
    tf.Tensor
        envelope 1.
    tf.Tensor
        envelope 2.
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
        super().__init__(
            files, window_length, group_key_fn, feature_sort_fn, as_tf_tensors
        )
        self.spacing = spacing

    @property
    def nb_features(self):
        return 2 * len(self.files[0]) - 1

    @property
    def feature_dims(self):
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
