import numpy as np
import tensorflow as tf


@tf.function
def batch_equalizer(eeg, match_env, mismatch_env):
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
    new_eeg = tf.concat([eeg, eeg], axis=0)
    env1 = tf.concat([match_env, mismatch_env], axis=0)
    env2 = tf.concat([mismatch_env, match_env], axis=0)
    # print(new_eeg.shape, env1.shape, env2.shape)

    labels = tf.concat(
        [
            tf.tile(
                tf.constant([[0]]), [tf.shape(eeg)[0], 1]
            ),
            tf.tile(
                tf.constant([[1]]), [tf.shape(eeg)[0], 1]
            ),
        ],
        axis=0,
    )

    # print(new_eeg.shape, env1.shape, env2.shape, labels.shape)
    return (new_eeg, env1, env2), labels


def create_generator(files, window_length, hop_length=64, spacing=64, batch_size=16):
    """ Creates a tf.data.Dataset.
    This will be used to create a dataset generator that will
    pass data to a model in match-mismatch task.

    parameters
    ---------
    files: list
        A list of data recordings
    window_length: int
        Length of the decision window
    hop_length: int
        Hop length between two consecutive decision windows
    spacing: int
        Number of samples (space) between end of matched speech and beginning of mismatched speech
    batch_size: int
        Batch-size, the actual batch-size will be 2 * batch_size due to batch_equalizer()

    returns
    -------
    tf.data.Dataset
        A generater that generates data for match-mismatch task
    """
    gen = GenerateData(files, window_length, spacing)

    # create tf dataset from generator
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([None, 64]), tf.TensorShape([None, 1]), tf.TensorShape([None, 1]))
    )
    # window dataset
    dataset = dataset.map(lambda x, y, z: (tf.signal.frame(x, window_length, hop_length, axis=0),
                                           tf.signal.frame(y, window_length, hop_length, axis=0),
                                           tf.signal.frame(z, window_length, hop_length, axis=0)))

    # batch data
    dataset = dataset.interleave(lambda x, y, z: tf.data.Dataset.from_tensor_slices((x, y, z)), cycle_length=4,
                                 block_length=16).batch(batch_size, drop_remainder=True)

    # prepare each batch
    dataset = dataset.map(lambda x, y, z: batch_equalizer(x, y, z))

    return dataset


class GenerateData:
    """Generate data."""

    def __init__(self, files, window_length, spacing):
        """Initialize.

        Parameters
        ----------
        window_length: int
            Length of the decision window.
        spacing: int
            Number of samples (space) between end of matched speech and beginning of mismatched speech
        """
        self.files = files
        self.window_length = window_length
        self.spacing = spacing

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Get item.

        Parameters
        ----------
        idx: int
            Index to get.

        Returns
        -------
        tuple (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            (EEG, matched envelope, mismatched envelope).
        """
        data = np.load(self.files[idx])
        return self.create_imposter(data)

    def create_imposter(self, data):
        """Creates mismatch (imposter) envelope.

        Parameters
        ----------
        data: numpy.ndarray
            Data to create an imposter for.

        Returns
        -------
        tuple (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            (EEG, matched envelope, mismatched envelope).
        """
        new_length = data.shape[0] - self.window_length - self.spacing
        eeg = data[0:new_length, 0:64]
        match_env = data[0:new_length, 64:]
        mismatch_env = data[self.spacing + self.window_length:, 64:]
        return (eeg, match_env, mismatch_env)

    def __call__(self):
        """Call.

        Parameters
        ----------

        Returns
        -------
        tuple (numpy.ndarray, numpy.ndarray, numpy.adarray)
            EEG, matched envelope, mismatched envelope.
        """
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)

            if idx == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        """End of epoch.
            Shuffles eeg data file paths.
        """
        np.random.shuffle(self.files)
