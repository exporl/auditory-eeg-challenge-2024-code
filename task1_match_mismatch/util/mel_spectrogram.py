# mel_spectrogram.py

import os
import math
import numpy as np
import json
from scipy import signal


def get_mel_matrix(fs, number_of_filters, n_fft, fmin, fmax):
    """
    Calculates filter matrix, which will be multiplied
    with spectrogram to get the mel spectrgoram.

    parameters
    ---------
    fs: int
        sampling frequency of speech data
    number_of_filters: int
        number of mel spectrogram frequency bands
    n_fft: int
        Fast Fourier Transform (FFT) window length
    fmin: int
        minimum center frequency used in mel filter matrix
    fmax: int
        maximum center frequency used in mel filter matrix

    returns
    -------
    numpy.ndarray
        mel filter matrix


    """
    low_mel_center = 2595 * np.log10(1 + fmin / 700)
    high_mel_center = 2595 * np.log10(1 + fmax / 700)
    distance = (high_mel_center - low_mel_center) / (number_of_filters - 1)

    low_mel = low_mel_center - distance
    start_mel = low_mel + np.arange(number_of_filters) * distance
    start_freq = 700 * (np.power(10, start_mel / 2595) - 1)
    start_bin = np.int32(np.ceil((n_fft / fs) * start_freq))

    end_mel = low_mel + np.arange(2, number_of_filters + 2) * distance
    end_freq = 700 * (np.power(10, end_mel / 2595) - 1)
    end_bin = np.int32(np.ceil((n_fft / fs) * end_freq))

    total_len = end_bin - start_bin + 1
    low_len = np.array(list(start_bin[1:]) + [end_bin[-2]]) - start_bin + 1
    high_len = total_len - low_len + 1

    mel_matrix = np.zeros((number_of_filters, n_fft))

    for k in range(number_of_filters):
        if k == 0 and (start_bin[0] <= 0):
            mel_matrix[0, 0:start_bin[k] + low_len[k] - 1] = np.arange(2 - start_bin[0], low_len[0] + 1) / low_len[0]
            mel_matrix[0, end_bin[0] - high_len[0]:end_bin[0]] = np.arange(high_len[0], 0, -1) / high_len[0]
        else:
            mel_matrix[k, start_bin[k] - 1:start_bin[k] + low_len[k] - 1] = np.arange(1, low_len[k] + 1) / low_len[k]
            mel_matrix[k, end_bin[k] - high_len[k]:end_bin[k]] = np.arange(high_len[k], 0, -1) / high_len[k]

    return mel_matrix


def calculate_mel_spectrogram(audio_path, target_fs=64, fmin=50, fmax=5000,
                              number_of_filters=28, hop_length=None, win_length=None):
    """Calculates mel spectrogram of a raw speech file.

    parameters
    ---------
    audio_path: str
        audio file path
    target_fs: int
        Sampling frequency of the calculated mel spectrogram
    fmin: int
        Minimum center frequency used in mel filter matrix
    fmax: int
        Maximum center frequency used in mel filter matrix
    number_of_filters: int
        Number of mel spectrogram frequency bands
    hop_length: int
        Hop length (in samples) used for calculation of the spectrogram
    win_length: int
        Window length (in samples) of each frame

    returns
    -------
    numpy.ndarray
        Mel spectrogram
    """

    speech = np.load(audio_path)
    audio, fs = speech['audio'], speech['fs']
    if not hop_length:
        hop_length = int((1 / 128) * fs)  # this will downsample the signal to 128 Hz
    if not win_length:
        win_length = int(0.025 * fs)      # 25 milli seconds

    # Finds the closest power of 2
    # that is bigger than win_length
    n_fft = int(math.pow(2, math.ceil(math.log2(win_length))))
    mel_floor = np.exp(-50)

    mel_matrix = get_mel_matrix(fs=fs, number_of_filters=number_of_filters, n_fft=n_fft, fmin=fmin, fmax=fmax)

    # Truncate
    number_of_frames = int(np.floor((len(audio) - win_length + hop_length) / hop_length))
    speech = audio[0:number_of_frames * hop_length + win_length - hop_length]

    # DC removal
    speech = speech - np.mean(speech)

    # Framing
    start_indices = np.arange(0, len(speech) - 1 - win_length + hop_length, hop_length)
    inside_frame_indices = np.arange(win_length)

    frame_indies = np.repeat(np.expand_dims(start_indices, axis=0), repeats=win_length, axis=0) + \
                   np.repeat(np.expand_dims(inside_frame_indices, axis=1), repeats=number_of_frames, axis=1)

    framed_speech = speech[frame_indies]

    # Windowing
    window = np.expand_dims(np.hanning(win_length), axis=1)
    xw = np.multiply(framed_speech, window)
    spectrogram = np.fft.fft(xw, n_fft, axis=0)
    spectrogram[0, :] = 0
    spectrogram = np.abs(spectrogram[0:int(n_fft / 2) + 1, :])
    mel_matrix = mel_matrix[:, 0:int(n_fft / 2) + 1].T
    mel = np.matmul(spectrogram.T, mel_matrix)
    mel = np.where(mel < mel_floor, mel_floor, mel)
    mel_spectrogram = np.power(mel, 0.6)

    # Downsample to 64 Hz
    mel_spectrogram = signal.resample_poly(mel_spectrogram, target_fs, int(fs / hop_length))
    return mel_spectrogram


if __name__ == '__main__':
    # Calculate mel spectrogram of a sample audio file
    # The path to the challenge dataset is in /util/dataset_root_dir.json
    dataset_path_file = os.path.join(os.getcwd(), 'dataset_root_dir.json')
    with open(dataset_path_file, 'r') as f:
        dataset_root_dir = json.load(f)
    audio_file = os.path.join(dataset_root_dir, 'train', 'stimuli', 'podcast_1.npz')
    mel = calculate_mel_spectrogram(audio_file)
    print(mel.shape)
