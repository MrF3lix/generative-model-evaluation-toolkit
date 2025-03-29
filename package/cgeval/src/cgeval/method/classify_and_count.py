import numpy as np
from collections import Counter

from cgeval import QuantificationMethod
from cgeval.report import CountReport

class ClassifyAndCount(QuantificationMethod):
    def __init__(self):
        super().__init__()

    def quantify(self, inputs: np.ndarray[int], metric_ratings: np.ndarray[int], labels: list[object], oracle_ratings: np.ndarray[int] = None) -> CountReport:
        total = len(inputs)
        input_counts = Counter(inputs)
        classifier_counts = Counter(metric_ratings)

        report = {}
        for i in range(len(labels)):
            id = labels[i]['id']
            name = labels[i]['name']

            report[name] = {
                'count_inputs': input_counts[id] / total,
                'count_metric_ratings': classifier_counts[id] / total,
            }

        return CountReport(inputs, labels, report)
