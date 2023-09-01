"""Code to calculate mel spectrograms."""
import math

import librosa
import numpy as np
import scipy.signal

def calculate_mel_spectrogram(
    audio_path,
    target_fs=64,
    fmin=0,
    fmax=5000,
    nb_filters=10,
    hop_length=None,
    win_length=None,
):
    """Calculates mel spectrogram of a raw speech file. This function makes the same calucation as
    in the sparrKULee pipeline and is the regression objective for task 2.

    Parameters
    ---------
    audio_path: str
        audio file path
    target_fs: int
        Sampling frequency of the calculated mel spectrogram
    fmin: Union[float, int]
        Minimum center frequency used in mel filter matrix
    fmax: Union[float, int]
        Maximum center frequency used in mel filter matrix
    nb_filters: int
        Number of mel spectrogram frequency bands
    hop_length: int
        Hop length (in samples) used for calculation of the spectrogram
    win_length: int
        Window length (in samples) of each frame

    Returns
    -------
    numpy.ndarray
        Mel spectrogram
    """

    # unzip audio file


    speech = dict(np.load(audio_path))
    audio, fs = speech["audio"], speech["fs"]
    if not hop_length:
        hop_length = int((1 / target_fs) * fs)  # this will downsample the signal to target_fs Hz
    if not win_length:
        win_length = int(0.025 * fs)  # 25 milli seconds

    # Finds the closest power of 2
    # that is bigger than win_length
    n_fft = int(math.pow(2, math.ceil(math.log2(win_length))))

    # DC removal
    audio = audio - np.mean(audio)

    mel_spectrogram = librosa.feature.melspectrogram(audio, window='hann',
                                       sr=fs, n_fft=n_fft, hop_length=hop_length,
                                       win_length=win_length, fmin=fmin, fmax=fmax, htk=False, norm='slaney',
                                       n_mels=nb_filters, center=False)


    return mel_spectrogram



# 'Center freqs' of mel bands - uniformly spaced between limits
# mel_f:  [   0.        ,  147.02442191,  324.92910187,  540.19997145,
#         800.6852341 , 1115.88148983, 1497.27995596, 1958.78540639,
#        2517.22310262, 3192.95219807, 4010.6079787 , 5000.        ]

