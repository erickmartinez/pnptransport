import abc


class ConductivityInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def estimate_conductivity(self):
        pass

    @property
    @abc.abstractmethod
    def concentration_profile(self):
        pass

    @concentration_profile.setter
    @abc.abstractmethod
    def concentration_profile(self, value):
        pass
