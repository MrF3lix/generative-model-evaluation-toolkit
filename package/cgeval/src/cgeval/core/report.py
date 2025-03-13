import json
from abc import ABC, abstractmethod

class Report(ABC):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def toJSON(self):
        pass

    def save(self, filename):
        data = self.report
        with open(filename, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=2)