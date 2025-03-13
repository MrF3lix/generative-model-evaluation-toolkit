import json
from prettytable import PrettyTable

from cgeval import Report
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support

# TODO: Can we implement aggregations? 
class MultiClassClassificationReport(Report):
    def __init__(self, actual, predictions, labels):
        self.report = {}
        # TODO: What if the ground truth is not available?
        self.actual = actual
        self.predictions = predictions
        self.labels = labels

        self.compute()

    def compute(self):
        self.cms = multilabel_confusion_matrix(self.actual, self.predictions, labels=self.labels)
        self.prfs = precision_recall_fscore_support(self.actual, self.predictions, labels=self.labels, average=None)

        for i in range(len(self.labels)):
            label = self.labels[i]
            cm = self.cms[i]

            self.report[label] = {
                'cm': self.cm_to_dict(cm),
                'precision': round(float(self.prfs[0][i]), 2),
                'recall': round(float(self.prfs[1][i]), 2),
                'fscore': round(float(self.prfs[2][i]), 2),
                'support': round(int(self.prfs[3][i]), 2),
            }

        return self.report
    

    def cm_to_dict(self, cm):
        return {
            'TP': int(cm[0][0]),
            'FN': int(cm[0][1]),
            'FP': int(cm[1][0]),
            'TN': int(cm[1][1]),
        }
    
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
            t.add_row([
                label,
                self.report[label]['precision'],
                self.report[label]['recall'],
                self.report[label]['fscore'],
                self.report[label]['support'],
                self.cm_to_string(self.report[label]['cm'])
            ], divider=True)

        return str(t)
    
    def toJSON(self):
        return json.dumps(
            self.report,
            sort_keys=True,
            indent=2
        )