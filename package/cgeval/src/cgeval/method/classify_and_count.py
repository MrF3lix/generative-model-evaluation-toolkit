import numpy as np
from collections import Counter

from cgeval import QuantificationMethod
from cgeval.report import CountReport
from cgeval.rating import Ratings

class ClassifyAndCount(QuantificationMethod):
    def __init__(self):
        super().__init__()

    def quantify(self, ratings: Ratings) -> CountReport:
        total = len(ratings)
        input_counts = ratings.compute_inputs_counts()
        oracle_counts = ratings.get_oracle_ratings()
        metric_counts = ratings.compute_metric_rating_counts()

        report = {}
        for i in range(len(ratings.labels)):
            id = ratings.labels[i].id
            name = ratings.labels[i].name

            report[name] = {
                'count_inputs': input_counts[id] / total,
                'count_metric_ratings': metric_counts[id] / total,
            }

        report['dataset'] = {
            'count_inputs': total,
            'count_oracle_ratings': len(oracle_counts),
            'count_metric_ratings': len(metric_counts)
        }

        return CountReport(ratings.labels, report)

    # def quantify_old(self, inputs: np.ndarray[int], metric_ratings: np.ndarray[int], labels: list[object], oracle_ratings: np.ndarray[int] = None) -> CountReport:
    #     total = len(inputs)
    #     input_counts = Counter(inputs)
    #     classifier_counts = Counter(metric_ratings)

    #     report = {}
    #     for i in range(len(labels)):
    #         id = labels[i]['id']
    #         name = labels[i]['name']

    #         report[name] = {
    #             'count_inputs': input_counts[id] / total,
    #             'count_metric_ratings': classifier_counts[id] / total,
    #         }

    #     report['dataset'] = {
    #         'count_inputs': total,
    #         'count_oracle_ratings': len(oracle_ratings),
    #         'count_metric_ratings': len(metric_ratings)
    #     }

    #     return CountReport(inputs, labels, report)
