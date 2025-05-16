import numpyro
import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter
from numpyro import infer
from numpyro import distributions as dist
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from cgeval import QuantificationMethod
from cgeval.report import CpccReport
from cgeval.rating import Ratings
from cgeval.distribution import Beta, BetaParams

numpyro.set_host_device_count(5)

RANDOM_SEED = 0xdeadbeef

class CPCC(QuantificationMethod):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__()

    def quantify(self, ratings: Ratings, classifier: str) -> CpccReport:
        oracle, metric = ratings.get_oracle_metric_pairs()

        if self.cfg.quantify.regression == 'logistic':
            logreg = LogisticRegression(
                C=1.,
                fit_intercept=True,
                penalty=None,
                class_weight=None,
                random_state=0xdeadbeef,
            )

            logreg.fit(np.array(metric).reshape(-1, 1), oracle)

            metric_only = ratings.get_metric_ratings()
            proba = logreg.predict_proba(np.array(metric_only).reshape(-1, 1))
            
            proba_dist = Beta(proba[:,1])
            proba_obs_dist = Beta(metric_only)

            a = sorted(Counter(oracle).items())[1][1] + 1
            b = sorted(Counter(oracle).items())[0][1] + 1
            oracle_dist = Beta(params=BetaParams(a, b))

        return CpccReport(oracle_dist, proba_dist, proba_obs_dist, classifier)
