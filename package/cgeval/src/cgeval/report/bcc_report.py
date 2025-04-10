import json
import numpy as np
from prettytable import PrettyTable

from cgeval import Report

class BccReport(Report):
    def __init__(self, labels, report, samples: dict[str, np.ndarray]):
        self.labels = labels
        self.report = report 
        self.samples = samples
    
    def __str__(self):
        t = PrettyTable(['Label', 'Input Prevalence', 'Metric Rating Prevalence', 'P_True Mean', 'P_True Std', 'P_True 5%', 'P_True 95%'])
        t.align["Support"] = "r"
        
        for label in self.labels:
            name = label.name
            t.add_row([
                name,
                self.report[name]['count_inputs'],
                self.report[name]['count_metric_ratings'],
                self.report[name]['p_true_mean'],
                self.report[name]['p_true_std'],
                self.report[name]['p_true_5'],
                self.report[name]['p_true_95'],
            ], divider=True)

        return str(t)
    
    def toJSON(self):
        return json.dumps(
            {
                'report': self.report,
                'samples': dict(map(lambda kv: (kv[0], kv[1].tolist()), self.samples.items()))
            },
            sort_keys=True,
            indent=2
        )