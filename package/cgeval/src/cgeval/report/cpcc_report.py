import json
from cgeval import Report
from prettytable import PrettyTable

class CpccReport(Report):
    def __init__(self, oracle_dist, proba_dist, proba_obs_dist, classifier):
        self.oracle_dist = oracle_dist
        self.proba_dist = proba_dist
        self.proba_obs_dist = proba_obs_dist
        self.classifier = classifier


    def dist_to_dict(self, name, dist):
        mean, var, skew, kurt = dist.stats()
        return {
            'name': f'{self.classifier} {name}',
            'a': round(dist.params.a, 2),
            'b': round(dist.params.b, 2),
            'mean': round(mean, 2),
            'var': round(var, 8),
            'skew': round(skew, 2),
            'kurt': round(kurt, 2)
        }
    
    def get_dist_table(self):
        t = PrettyTable(['', 'a', 'b', 'mean', 'var', 'skew', 'kurt'])

        t.add_row(self.dist_to_dict('oracle', self.oracle_dist).values())
        t.add_row(self.dist_to_dict('p', self.proba_dist).values())
        t.add_row(self.dist_to_dict('p_obs', self.proba_obs_dist).values())

        return str(t)


    def __str__(self):
        return self.get_dist_table()
    
    def toJSON(self):
        return json.dumps(
            {
                'dist': [
                    self.dist_to_dict('oracle', self.oracle_dist),
                    self.dist_to_dict('p', self.proba_dist),
                    self.dist_to_dict('p_obs', self.proba_obs_dist),
                ]
            },
            sort_keys=True,
            indent=2
        )