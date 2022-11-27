"""Factory for normalizers."""
from util.normalizers.standardizer import Standardizer


def normalizer_factory(normalizer_name, **kwargs):
    """Create a new normalizer based on normalizer_name.

    Parameters
    ----------
    normalizer_name : str
        Name of the normalizer to create
    kwargs : dict
        Optional additional keyword arguments.

    Returns
    -------
    Normalizer
        Normalizer corresponding to normalizer_name
    """
    if normalizer_name.lower() == "standardizer":
        return Standardizer()
    else:
        raise ValueError(f'Unknown normalizer "{normalizer_name}"')
