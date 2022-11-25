"""Code to calculate speech envelopes."""
import numpy as np

from brian2 import Hz, kHz
from brian2hears import Sound, erbspace, Gammatone, Filterbank
from scipy import signal
from util.stimulus_processing.feature_extraction.base import FeatureExtractor


def calculate_envelope(audio_path, power_factor=0.6, target_fs=64):
    """Calculates gammatone envelope of a raw speech file.

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
    audio, fs = speech["audio"], speech["fs"]
    del speech

    sound = Sound(audio, samplerate=fs * Hz)
    # 28 center frequencies from 50 Hz till 5kHz
    center_frequencies = erbspace(50 * Hz, 5 * kHz, 28)
    filter_bank = Gammatone(sound, center_frequencies)
    envelope_calculation = EnvelopeFromGammatone(filter_bank, power_factor)
    output = envelope_calculation.process()

    # Downsample to 64 Hz
    envelope = signal.resample_poly(output, target_fs, fs)
    return envelope


class EnvelopeFromGammatone(Filterbank):
    """Class to calculate an envelope from a gammatone filterbank."""

    def __init__(self, source, power_factor):
        super().__init__(source)
        self.power_factor = power_factor
        self.nchannels = 1

    def buffer_apply(self, input_):
        return np.reshape(
            np.sum(np.power(np.abs(input_), self.power_factor), axis=1, keepdims=True),
            (np.shape(input_)[0], self.nchannels),
        )


class GammatoneEnvelope(FeatureExtractor):
    def __init__(self, power_factor=0.6, target_fs=64):
        self.power_factor = power_factor
        self.target_fs = target_fs

    def __call__(self, audiopath):
        return calculate_envelope(audiopath, self.power_factor, self.target_fs)
