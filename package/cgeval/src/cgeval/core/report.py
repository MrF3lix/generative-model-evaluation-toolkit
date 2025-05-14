import json
import numpy as np
from abc import ABC, abstractmethod

class Report(dict):
    @abstractmethod
    def __str__(self):
        pass
    
    def __getattr__(self, attr):
        return self[attr]

    @abstractmethod
    def toJSON(self):
        pass

    def load(self, filename):
        with open(filename, 'r') as f:
            report_data = json.load(f)

        self.report = report_data['report']
        self.dist_report = report_data['dist']
        self.samples = dict(map(lambda kv: (kv[0], np.array(kv[1], dtype=np.float64)), report_data['samples'].items()))

    def save(self, filename):
        data = self.toJSON()
        with open(filename, 'w') as f:
            f.write(data)