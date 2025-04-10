import json
import numpy as np
from abc import ABC, abstractmethod

class Report(ABC):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def toJSON(self):
        pass

    def load(self, filename):
        with open(filename, 'r') as f:
            report_data = json.load(f)

        self.report = report_data['report']
        self.samples = dict(map(lambda kv: (kv[0], np.array(kv[1], dtype=np.float64)), report_data['samples'].items()))

    def save(self, filename):
        data = {
            'report': self.report,
            'samples': dict(map(lambda kv: (kv[0], kv[1].tolist()), self.samples.items()))
        }
        with open(filename, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=2)