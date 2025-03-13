import json
from prettytable import PrettyTable
from collections import Counter

from cgeval import Report

class CountReport(Report):
    def __init__(self, inputs, predictions, labels):
        self.report = {}
        self.inputs = inputs
        self.predictions = predictions
        self.labels = labels

        self.compute()

    def compute(self):
        input_counts = Counter(self.inputs)
        classifier_counts = Counter(self.predictions)

        for i in range(len(self.labels)):
            label = self.labels[i]

            self.report[label] = {
                'input_counts': input_counts[label],
                'classifier_counts': classifier_counts[label],
            }

        return self.report
    
    def __str__(self):
        t = PrettyTable(['Label', 'Input Counts', 'Classifier Counts'])
        t.align["Support"] = "r"
        
        for label in self.labels:
            t.add_row([
                label,
                self.report[label]['input_counts'],
                self.report[label]['classifier_counts'],
            ], divider=True)

        return str(t)
    
    def toJSON(self):
        return json.dumps(
            self.report,
            sort_keys=True,
            indent=2
        )