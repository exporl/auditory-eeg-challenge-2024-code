"""This module creates a dataset generator to be used in the regression task."""
import numpy as np
import tensorflow as tf
import pickle
import os


def create_generator(eeg_files, envelope_dir, window_length, hop_length, batch_size):
    """ Creates a tf.data.Dataset.
        This will be used to create a dataset generator that will
        pass data batches to a model.

        parameters
        ---------
        eeg_files: list
            A list of EEG data recordings
        envelope_dir: str
            Directory containing envelope of stimuli
        window_length: int
            Length of the decision window
        hop_length: int
            Hop length between two consecutive decision windows
        batch_size: int
            Batch-size

        returns
        -------
        tf.data.Dataset
            A generater that generates data for the regression task.
        """
    gen = GenerateData(eeg_files, envelope_dir, window_length, hop_length)

    # Create tf dataset from generator
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([None, 64]), tf.TensorShape([None, 1]))
    )
    # Window dataset
    dataset = dataset.map(lambda x, y: (tf.signal.frame(x, window_length, hop_length, axis=0),
                                        tf.signal.frame(y, window_length, hop_length, axis=0)))

    # Batch
    dataset = dataset.interleave(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)), cycle_length=4,
                                 block_length=16).batch(batch_size, drop_remainder=True)

    return dataset


class GenerateData:
    """Generate data."""

    def __init__(self, eeg_files, envelope_dir, window_length, hop_length):
        """Initialize.

        Parameters
        ----------
        eeg_files: list
            A list of EEG data recordings
        envelope_dir: str
            Directory containing envelope of stimuli
        window_length: int
            Length of the decision window.
        hop_length: int
            Hop length between two consecutive decision windows.
        """

        self.window_length = window_length
        self.hop_length = hop_length
        self.eeg_files = eeg_files
        self.envelope_dir = envelope_dir

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        """Get item.

        Parameters
        ----------
        idx: int
            Index to get.

        Returns
        -------
        tuple (numpy.ndarray, numpy.ndarray)
            (EEG, envelope).
        """

        data = pickle.load(open(self.eeg_files[idx], 'rb'))

        return self.extract_features(data)

    def extract_features(self, data):
        """Extracts EEG and envelope.

        parameters
        ----------
        data: pickle object
            pickle object containg EEG data

        retruns
        -------
        tuple(numpy.ndarray, numpy.ndarray)
            EEG , envelope of stimulus
        """
        eeg = data['eeg']

        # load the corresponding envelope based on stimulus name
        stimulus = data['stimulus']
        stimulus = stimulus.split('.')[0]
        speech = np.load(os.path.join(self.envelope_dir, stimulus + '.npz'))
        envelope = speech['env']
        if len(envelope.shape) == 1:
            envelope = np.expand_dims(envelope, axis=1)

        # make sure eeg and envelope have the same length
        minimum_length = min(eeg.shape[0], len(envelope))
        eeg = eeg[:minimum_length]
        envelope = envelope[:minimum_length]

        return eeg, envelope

    def __call__(self):
        """Call.

        Parameters
        ----------

        Returns
        -------
        numpy.ndarray
            EEG, envelope
        """
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)

            if idx == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        """End of epoch.
            Shuffles eeg data file paths.
        """
        np.random.shuffle(self.eeg_files)
