# envelope.py

from scipy import signal
from brian2 import *
from brian2hears import *
import numpy as np
import json
import os


def calculate_envelope(audio_path, power_factor=0.6, target_fs=64):
    """ Calculates envelope of a
     raw speech file.

     -- the calculation of the envelope is based on the
     following publication:
     "Auditory-Inspired Speech Envelope Extraction Methods for Improved EEG-Based
      Auditory Attention Detection in a Cocktail Party Scenario" (Biesmans et al., 2016)

     parameters
     ----------
     audio_path: str
        Audio file path
     power_factor: float
        Power used in power law relation, which is used to model
        relation between perceived loudness and actual speech
        intensity
     target_fs: int
        Sampling frequency of the calculated envelope

    returns
    -------
    numpy.ndarray
        Envelope of a speech
     """
    speech = np.load(audio_path)
    audio, fs = speech['audio'], speech['fs']

    sound = Sound(audio)
    # 28 center frequencies from 50 Hz till 5kHz
    center_frequencies = erbspace(50 * Hz, 5 * kHz, 28)
    filter_bank = Gammatone(sound, center_frequencies)
    output = filter_bank.process()
    output = np.abs(output)
    output = np.power(output, power_factor)
    envelope = np.mean(output, axis=1)
    # Downsample to 64 Hz
    envelope = signal.resample_poly(envelope, target_fs, fs)
    return envelope


if __name__ == '__main__':
    # Calculate envelope of a sample audio file
    # The path to the challenge dataset is in /util/dataset_root_dir.json
    dataset_path_file = os.path.join(os.getcwd(), 'dataset_root_dir.json')
    with open(dataset_path_file, 'r') as f:
        dataset_root_dir = json.load(f)
    audio_file = os.path.join(dataset_root_dir, 'train', 'stimuli', 'podcast_1.npz')
    env = calculate_envelope(audio_file)
    print(env.shape)
