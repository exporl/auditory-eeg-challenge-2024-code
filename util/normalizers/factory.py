from util.normalizers.standardizer import Standardizer


def normalizer_factory(normalizer_name, **kwargs):
    if normalizer_name.lower() == "standardizer":
        return Standardizer()
    else:
        raise ValueError(f'Unknown normalizer "{normalizer_name}"')
