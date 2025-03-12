from prettytable import PrettyTable

from cgeval import Report
from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_recall_fscore_support

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
                'cm': cm,
                'precision': self.prfs[0][i],
                'recall': self.prfs[1][i],
                'fscore': self.prfs[2][i],
                'support': self.prfs[3][i],
            }

        return self.report
    
    def cm_to_dict(self, cm):
        return f"TP: {int(cm[0][0])} FN: {int(cm[0][1])}\nFP: {int(cm[1][0])} TN: {int(cm[1][1])}"

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
                round(self.report[label]['precision'], 2),
                round(self.report[label]['recall'], 2),
                round(self.report[label]['fscore'], 2),
                round(self.report[label]['support'], 2),
                self.cm_to_dict(self.report[label]['cm'])
            ], divider=True)

        return str(t)