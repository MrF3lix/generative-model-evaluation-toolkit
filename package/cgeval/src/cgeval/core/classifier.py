from abc import ABC, abstractmethod, abstractproperty

class Classifier(ABC):
    @abstractmethod
    def classify(self, inputs: object) -> object:
        pass
 