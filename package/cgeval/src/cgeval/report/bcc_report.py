import json
import numpy as np
from prettytable import PrettyTable

from cgeval import Report
from cgeval.distribution import Beta

class BccReport(Report):
    def __init__(self, labels, report, samples: dict[str, np.ndarray], oracle_dist: Beta, alpha_dist: Beta, alpha_obs_dist: Beta, classifier: str):
        self.labels = labels
        self.report = report 
        self.samples = samples

        self.oracle_dist = oracle_dist
        self.alpha_dist = alpha_dist
        self.alpha_obs_dist = alpha_obs_dist

        self.classifier = classifier

        self.dist_report = self.get_dist_report()


    def get_dist_report(self):
        report = []

        o_mean, o_var, o_skew, o_kurt = self.oracle_dist.stats()
        report.append({
            'classifier': 'oracle',
            'a': self.oracle_dist.params.a,
            'b': self.oracle_dist.params.b,
            'mean': round(o_mean, 2),
            'var': round(o_var, 8),
            'skew': round(o_skew, 2),
            'kurt': round(o_kurt, 2)
        })

        mean, var, skew, kurt = self.alpha_dist.stats()
        report.append({
            'classifier': f'{self.classifier} alpha',
            'a': round(self.alpha_dist.params.a, 2),
            'b': round(self.alpha_dist.params.b, 2),
            'mean': round(mean, 2),
            'var': round(var, 8),
            'skew': round(skew, 2),
            'kurt': round(kurt, 2)
        })

        mean, var, skew, kurt = self.alpha_obs_dist.stats()
        report.append({
            'classifier': f'{self.classifier} alpha obs',
            'a': round(self.alpha_obs_dist.params.a, 2),
            'b': round(self.alpha_obs_dist.params.b, 2),
            'mean': round(mean, 2),
            'var': round(var, 8),
            'skew': round(skew, 2),
            'kurt': round(kurt, 2)
        })

        return report

    def get_dist_table(self):
        t = PrettyTable(['classifier', 'a', 'b', 'mean', 'var', 'skew', 'kurt'])

        for dist in self.dist_report:
            t.add_row(dist.values())

        return str(t)
    
    def get_label_table(self):
        t = PrettyTable(['Label', 'Input Prevalence', 'Metric Rating Prevalence', 'P_True Mean', 'P_True Std', 'P_True 5%', 'P_True 95%'])
        
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

    def __str__(self):
        label_table = self.get_label_table()
        dist_table = self.get_dist_table()

        return label_table + '\n' + dist_table
    
    def toJSON(self):
        return json.dumps(
            {
                'report': self.report,
                'dist': self.get_dist_report(),
                'samples': dict(map(lambda kv: (kv[0], kv[1].tolist()), self.samples.items()))
            },
            sort_keys=True,
            indent=2
        )