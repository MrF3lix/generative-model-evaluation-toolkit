from abc import ABC, abstractmethod

class Classifier(ABC):
    @abstractmethod
    def classify(self, dataloader: object) -> object:
        pass
 