from abc import ABC, abstractmethod
from cgeval import Report

class QuantificationMethod(ABC):

    @abstractmethod
    def eval(self, dataloader, inference) -> Report:
        pass