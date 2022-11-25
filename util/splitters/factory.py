from util.splitters.sequential import SequentialSplitter


def splitter_factory(splitter_name, **kwargs):
    if splitter_name.lower() == "sequential":
        return SequentialSplitter(**kwargs)
    else:
        raise ValueError(f'Unknown splitter "{splitter_name}"')
