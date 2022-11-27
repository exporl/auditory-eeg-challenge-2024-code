"""Feature factory function for speech features."""
from util.stimulus_processing.feature_extraction import envelope, mel_spectrogram


def speech_feature_factory(feature_name, **kwargs):
    """Create a speech feature based on a name.

    Parameters
    ----------
    feature_name : str
        Name of the speech feature
    kwargs : dict
        Extra keyword arguments to pass to speeh feature implementation

    Returns
    -------
    FeatureExtractor
        FeatureExtractor that can extract a speech feature.
    """
    if feature_name in ["env", "envelope"]:
        return envelope.GammatoneEnvelope()
    elif feature_name in ["mel", "mel_spectrogram"]:
        return mel_spectrogram.MelSpectrogram()
    else:
        raise ValueError(
            f'Unknown feature "{feature_name}": '
            f'no implementation for feature extraction available'
        )
