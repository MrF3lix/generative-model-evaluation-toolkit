import numpyro
import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter
from numpyro import infer
from numpyro import distributions as dist
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from cgeval import QuantificationMethod
from cgeval.report import BccReport
from cgeval.rating import Ratings
from cgeval.distribution import Beta, BetaParams

numpyro.set_host_device_count(5)

RANDOM_SEED = 0xdeadbeef

class BCC(QuantificationMethod):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__()

    def quantify(self, ratings: Ratings, classifier: str) -> BccReport:
        mu = ratings.compute_mixture_matrix()

        oracle = ratings.get_oracle_ratings()
        metric = ratings.get_metric_ratings()

        # TODO: Investigate 
        metric = [x for x in metric if x is not None] 

        oracle_ratings_count = Counter(oracle)
        oracle_ratings = np.array([oracle_ratings_count[0], oracle_ratings_count[1]])

        metric_ratings_count = Counter(metric)
        metric_ratings = np.array([metric_ratings_count[0], metric_ratings_count[1]])
        
        # samples = self.mcmm_sampling(mu, oracle_ratings, metric_ratings, ratings)
        try:
            samples = self.mcmm_sampling(mu, oracle_ratings, metric_ratings, ratings)
        except Exception as e:  
            print(e) 
            print(oracle_ratings)
            print(metric_ratings)
            print(sorted(Counter(oracle).items()))
            print(sorted(Counter(metric).items()))

            samples = {
                'alpha': np.array([0]),
                'alpha_obs': np.array([0]),
                'p_true': np.array([np.array([0]),np.array([0]),np.array([0])])
            }

            print('Failed to run MCMM sampling')

        total = len(ratings)
        input_counts = ratings.compute_inputs_counts()
        metric_counts = ratings.compute_metric_rating_counts()

        report = {}
        for label in ratings.labels:
            id = label.id
            name = label.name

            s = samples['alpha']
            report[name] = {
                'count_inputs': input_counts[id] / total,
                'count_metric_ratings': metric_counts[id] / total,
                'oracle_ratings': oracle.tolist(),
                f'p_true_mean': round(float(s.mean()), 4),
                f'p_true_std': round(float(s.std()), 4),
                f'p_true_5': round(float(np.quantile(s, 0.95)), 4),
                f'p_true_95': round(float(np.quantile(s, 0.05)), 4)
            }

            if id == 0:
                report[name]['p_true_mean'] = round(float(1 - s.mean()), 4)
                report[name]['p_true_5'] = round(float(1 - np.quantile(s, 0.95)), 4),
                report[name]['p_true_95'] = round(float(1 - np.quantile(s, 0.05)), 4)

        a = int(oracle_ratings[1]) + 1
        b = int(oracle_ratings[0]) + 1

        return BccReport(ratings.labels, report, samples, Beta(params=BetaParams(a, b)), Beta(samples['alpha']), Beta(samples['alpha_obs']), classifier)

    def mcmm_sampling(self, mu_data, oracle_data, metric_data, ratings):
        def binar_model_fn():
            (tn, fp, fn, tp) = ratings.compute_confusion_matrix()

            # TODO: Check if this works for the images!
            # print(metric_data)
            # print(oracle_data)
            # print(tn, fp, fn, tp)

            # raise Exception('Done')

            tpr = numpyro.sample(
                "tpr",
                dist.Beta(tp + 1, fn + 1)
            )
            
            fpr = numpyro.sample(
                "fpr",
                dist.Beta(fp + 1, tn + 1)
            )

            alpha = numpyro.sample(
                "alpha",
                dist.Beta(oracle_data[1] + 1, oracle_data[0] + 1)
            )

            alpha_obs = numpyro.deterministic(
                "alpha_obs",
                alpha*(tpr - fpr) + fpr
            )

            numpyro.sample(
                "obs",
                dist.Binomial(total_count=metric_data.sum(), probs=alpha_obs),
                obs=metric_data[1],
            )

        sampler = infer.MCMC(
            infer.NUTS(binar_model_fn),
            num_warmup=2_000,
            num_samples=10_000,
            num_chains=5,
            progress_bar=True,
        )

        sampler.run(jax.random.PRNGKey(RANDOM_SEED))
        samples = sampler.get_samples(group_by_chain=False)

        return samples