# normalize.py

import numpy as np


class MeanVarNor:
    """class to mean variance normalize EEG and speech data.
    """
    def __init__(self, eeg_train, speech_train, axis=0):
        # EEG
        self.mean_eeg = np.mean(eeg_train, axis=axis)
        self.std_eeg = np.std(eeg_train, axis=axis)
        # Envelope
        self.mean_speech = np.mean(speech_train, axis=axis)
        self.std_speech = np.std(speech_train, axis=axis)

    def __call__(self, eeg, speech):
        eeg = (eeg - self.mean_eeg) / self.std_eeg
        speech = (speech - self.mean_speech) / self.std_speech
        return eeg, speech
