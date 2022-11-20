"""General utilities."""
import numpy as np


def window_data(data, window_length, hop):
    """Window data into overlapping windows.

    Parameters
    ----------
    data: np.ndarray
        Data to window. Shape (n_samples, n_channels)
    window_length: int
        Length of the window in samples.
    hop: int
        Hop size in samples.

    Returns
    -------
    np.ndarray
        Windowed data. Shape (n_windows, window_length, n_channels)
    """
    new_data = np.empty(
        ((data.shape[0] - window_length) // hop, window_length, data.shape[1])
    )
    for i in range(new_data.shape[0]):
        new_data[i, :, :] = data[
            i * hop : i * hop + window_length, :  # noqa: E203 E501
        ]
    return new_data
