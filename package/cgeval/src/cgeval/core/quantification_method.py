from abc import ABC, abstractmethod

class QuantificationMethod(ABC):

    @abstractmethod
    def eval(self, dataloader, inference):
        pass