from abc import ABC, abstractmethod

class Dataset(ABC):

    @abstractmethod
    def load(self):
        pass