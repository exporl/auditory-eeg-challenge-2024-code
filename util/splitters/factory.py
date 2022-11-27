"""Factory function for splitters."""
from util.splitters.sequential import SequentialSplitter


def splitter_factory(splitter_name, **kwargs):
    """Choose a Splitter by name.

    Parameters
    ----------
    splitter_name : str
        Name of the splitter.
    kwargs : dict
        Optional additional keyword arguments for the splitter

    Returns
    -------
    Splitter
        The chosen splitter.
    """
    if splitter_name.lower() == "sequential":
        return SequentialSplitter(**kwargs)
    else:
        raise ValueError(f'Unknown splitter "{splitter_name}"')
