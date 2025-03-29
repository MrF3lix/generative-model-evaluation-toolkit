import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support

from cgeval import QuantificationMethod
from cgeval.report import MultiClassClassificationReport

# TODO: Make general classification report
class StandardClassification(QuantificationMethod):
    def __init__(self):
        super().__init__()

    def cm_to_dict(self, cm):
        return {
            'TP': int(cm[0][0]),
            'FN': int(cm[0][1]),
            'FP': int(cm[1][0]),
            'TN': int(cm[1][1]),
        }
    
    def quantify(self, inputs: np.ndarray[int], metric_ratings: np.ndarray[int], oracle_ratings: np.ndarray[int], labels: list[object]):
        l = list(map(lambda l: l['id'], labels))

        cms = multilabel_confusion_matrix(oracle_ratings, metric_ratings, labels=l)
        prfs = precision_recall_fscore_support(oracle_ratings, metric_ratings, labels=l, average=None)

        report = {}
        for i in range(len(labels)):
            label = labels[i]['name']
            cm = cms[i]

            report[label] = {
                'cm': self.cm_to_dict(cm),
                'precision': round(float(prfs[0][i]), 2),
                'recall': round(float(prfs[1][i]), 2),
                'fscore': round(float(prfs[2][i]), 2),
                'support': round(int(prfs[3][i]), 2),
            }

        return MultiClassClassificationReport(labels, report)