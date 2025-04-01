from cgeval import QuantificationMethod
from cgeval.report import MultiClassClassificationReport
from cgeval.rating import Ratings

class StandardClassification(QuantificationMethod):
    def __init__(self):
        super().__init__()

    def cm_to_dict(self, cm):
        return {
            'TP': float(cm[1][1]),
            'FN': float(cm[1][0]),
            'FP': float(cm[0][1]),
            'TN': float(cm[0][0]),
        }

    def quantify(self, ratings: Ratings) -> MultiClassClassificationReport:
        cms = ratings.compute_mixture_matrix()
        prfs = ratings.compute_precision_recall()

        report = {}
        for i in range(len(ratings.labels)):
            label = ratings.labels[i].name
            cm = cms[i]

            report[label] = {
                'cm': self.cm_to_dict(cm),
                'precision': round(float(prfs[0][i]), 2),
                'recall': round(float(prfs[1][i]), 2),
                'fscore': round(float(prfs[2][i]), 2),
                'support': round(int(prfs[3][i]), 2),
            }

        return MultiClassClassificationReport(ratings.labels, report)
