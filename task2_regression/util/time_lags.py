import numpy as np


def time_lag_matrix(eeg, envelope=None, num_lags=26):
    """Create a time-lag matrix from a 2D numpy array.

    Parameters
    ----------
    eeg: numpy.ndarray
        2D numpy array with shape (n_samples, n_channels)
    envelope: numpy.ndarray
        Envelope of speech
    num_lags: int
        Number of time lags to use.

    Returns
    -------
    np.ndarray
        2D numpy array with shape (n_samples, n_channels* num_lags)
    """

    num_channels = np.shape(eeg)[1]
    # Create a time-lag matrix
    for i in range(0, num_lags):
        # Roll the array to the right
        eeg_t = np.roll(eeg, -i, axis=0)

        # zero-pad at the end
        if i > 0:
            eeg_t[-i:, :] = 0

        if i == 0:
            final_array = eeg_t
        else:
            final_array = np.concatenate((final_array, eeg_t), axis=1)

    # Shuffle the columns such that they are ordered by time lag
    final_array = final_array[
        :,
        list(
            np.concatenate(
                [
                    np.arange(i, final_array.shape[1], num_channels)
                    for i in range(num_channels)
                ]
            )
        ),
    ]

    time_len = final_array.shape[0]
    # Add bias term
    final_array = np.concatenate((final_array, np.ones((time_len, 1))), axis=1)
    if envelope is None:
        return final_array
    else:
        return final_array, envelope
