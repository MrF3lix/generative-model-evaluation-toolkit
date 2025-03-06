from abc import ABC, abstractmethod

class Classifier(ABC):

    @abstractmethod
    def classify(self, inputs: object) -> object:
        pass
 