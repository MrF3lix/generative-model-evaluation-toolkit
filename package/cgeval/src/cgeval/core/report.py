from abc import ABC, abstractmethod

class Report(ABC):
    @abstractmethod
    def __str__(self):
        pass
 