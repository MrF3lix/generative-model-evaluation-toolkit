
import json

from cgeval import Report

class GenericReport(Report):
    def __init__(self):
        pass
    
    def __str__(self):
        return str(self.samples)
    
    def toJSON(self):
        return json.dumps(
            self.report,
            sort_keys=True,
            indent=2
        )