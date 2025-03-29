import json
from prettytable import PrettyTable

from cgeval import Report

class MultiClassClassificationReport(Report):
    def __init__(self, labels, report):
        self.labels = labels
        self.report = report
    
    def cm_to_string(self, cm):
        return f"TP: {cm['TP']} FN: {cm['FN']}\nFP: {cm['FP']} TN: {cm['TN']}"

    def __str__(self):
        t = PrettyTable(['Label', 'Precision', 'Recall', 'F1-score', 'Support', 'CM'])
        t.align["Precision"] = "r"
        t.align["Recall"] = "r"
        t.align["F1-score"] = "r"
        t.align["Support"] = "r"
        t.align["CM"] = "l"
        
        for label in self.labels:
            name = label['name']
            t.add_row([
                name,
                self.report[name]['precision'],
                self.report[name]['recall'],
                self.report[name]['fscore'],
                self.report[name]['support'],
                self.cm_to_string(self.report[name]['cm'])
            ], divider=True)

        return str(t)
    
    def toJSON(self):
        return json.dumps(
            self.report,
            sort_keys=True,
            indent=2
        )