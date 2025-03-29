import json
from prettytable import PrettyTable

from cgeval import Report

class CountReport(Report):
    def __init__(self, inputs, labels, report):
        self.inputs = inputs
        self.labels = labels
        self.report = report
    
    def __str__(self):
        t = PrettyTable(['Label', 'Input Prevalence', 'Metric Rating Prevalence'])
        t.align["Support"] = "r"
        
        for label in self.labels:
            name = label['name']
            t.add_row([
                name,
                self.report[name]['count_inputs'],
                self.report[name]['count_metric_ratings'],
            ], divider=True)

        return str(t)
    
    def toJSON(self):
        return json.dumps(
            self.report,
            sort_keys=True,
            indent=2
        )