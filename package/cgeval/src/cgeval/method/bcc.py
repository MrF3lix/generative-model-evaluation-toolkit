import numpyro
import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter
from numpyro import infer
from numpyro import distributions as dist

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
        n_classes = 2 if self.cfg.quantify.comparison == 'binary' else ratings.get_label_count()
        mu = ratings.compute_mixture_matrix()

        oracle = ratings.get_oracle_ratings()
        metric = ratings.get_metric_ratings()

        # TODO: Investigate 
        metric = [x for x in metric if x is not None] 

        oracle_ratings = np.array(sorted(Counter(oracle).items()))[:,1][0:n_classes].astype(int)
        metric_ratings = np.array(sorted(Counter(metric).items()))[:,1][0:n_classes].astype(int)
        
        samples = self.mcmm_sampling(mu, oracle_ratings, metric_ratings, ratings)
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

            if self.cfg.quantify.comparison == 'binary':
                s = samples['alpha']

                if id == 0:
                    report[name] = {
                        'count_inputs': input_counts[id] / total,
                        'count_metric_ratings': metric_counts[id] / total,
                        'oracle_ratings': oracle.tolist(),
                        f'p_true_mean': round(float(1 - s.mean()), 4),
                        f'p_true_std': round(float(s.std()), 4),
                        f'p_true_5': round(float(1 - np.quantile(s, 0.95)), 4),
                        f'p_true_95': round(float(1 - np.quantile(s, 0.05)), 4)
                    }
                else:
                    report[name] = {
                        'count_inputs': input_counts[id] / total,
                        'count_metric_ratings': metric_counts[id] / total,
                        'oracle_ratings': oracle.tolist(),
                        f'p_true_mean': round(float(s.mean()), 4),
                        f'p_true_std': round(float(s.std()), 4),
                        f'p_true_5': round(float(np.quantile(s, 0.05)), 4),
                        f'p_true_95': round(float(np.quantile(s, 0.95)), 4)
                    }

            else:
                s = samples['p_true'][:,id]
                report[name] = {
                    'count_inputs': input_counts[id] / total,
                    'count_metric_ratings': metric_counts[id] / total,
                    f'p_true_mean': round(float(s.mean()), 4),
                    f'p_true_std': round(float(s.std()), 4),
                    f'p_true_5': round(float(np.quantile(s, 0.05)), 4),
                    f'p_true_95': round(float(np.quantile(s, 0.95)), 4)
                }

        # HACK: only works for case where a label match exists (in binary this might be easy but how do we deal with this in a multilabel scenario)
        oracle_ratings = report['match']['oracle_ratings']
        a = sorted(Counter(oracle_ratings).items())[1][1] + 1
        b = sorted(Counter(oracle_ratings).items())[0][1] + 1

        return BccReport(ratings.labels, report, samples, Beta(params=BetaParams(a, b)), Beta(samples['alpha']), Beta(samples['alpha_obs']), classifier)

    def mcmm_sampling(self, mu_data, oracle_data, metric_data, ratings):
        def binar_model_fn():
            (tn, fp, fn, tp) = ratings.compute_confusion_matrix()

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

        def model_fn():
            mu_cols = []
            for ix, col in enumerate(mu_data):
                mu_cols.append(numpyro.sample(
                    f"mu_col_{ix}",
                    dist.Dirichlet(jnp.array(col) + 1)
                ))
            mu = jnp.array(mu_cols).T

            ps = numpyro.sample(
                "p_true",
                dist.Dirichlet(jnp.array(oracle_data) + 1)
            )

            p_obs = numpyro.deterministic("p_obs", mu @ ps)

            numpyro.sample(
                "n_obs",
                dist.Multinomial(total_count=metric_data.sum(), probs=p_obs),
                obs=jnp.array(metric_data),
            )

        sampler = infer.MCMC(
            infer.NUTS(binar_model_fn if self.cfg.quantify.comparison == 'binary' else model_fn),
            num_warmup=2_000,
            num_samples=10_000,
            num_chains=5,
            progress_bar=True,
        )

        sampler.run(jax.random.PRNGKey(RANDOM_SEED))
        samples = sampler.get_samples(group_by_chain=False)

        return samples