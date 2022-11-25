import abc


class Normalizer:
    @abc.abstractmethod
    def __call__(self, feature_name, feature):
        pass

    @abc.abstractmethod
    def new_recording(self):
        pass
