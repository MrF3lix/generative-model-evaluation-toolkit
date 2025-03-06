from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def generate(self, inputs):
        pass
