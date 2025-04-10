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

RANDOM_SEED = 0xdeadbeef

class BCC(QuantificationMethod):
    def __init__(self):
        super().__init__()

    def quantify(self, ratings: Ratings) -> BccReport:
        n_classes = ratings.get_label_count()
        mu = ratings.compute_mixture_matrix()

        oracle = ratings.get_oracle_ratings()
        metric = ratings.get_metric_ratings()

        oracle_ratings = np.array(sorted(Counter(oracle).items()))[:,1][0:n_classes].astype(int)
        metric_ratings = np.array(sorted(Counter(metric).items()))[:,1][0:n_classes].astype(int)

        samples = self.mcmm_sampling(mu, oracle_ratings, metric_ratings)

        total = len(ratings)
        input_counts = ratings.compute_inputs_counts()
        metric_counts = ratings.compute_metric_rating_counts()

        report = {}
        for label in ratings.labels:
            id = label.id
            name = label.name

            s = samples['p_true'][:,id]
            report[name] = {
                'count_inputs': input_counts[id] / total,
                'count_metric_ratings': metric_counts[id] / total,
                f'p_true_mean': round(float(s.mean()), 4),
                f'p_true_std': round(float(s.std()), 4),
                f'p_true_5': round(float(np.quantile(s, 0.05)), 4),
                f'p_true_95': round(float(np.quantile(s, 0.95)), 4)
            }

        return BccReport(ratings.labels, report, samples)

    def mcmm_sampling(self, mu_data, oracle_data, metric_data):

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
            infer.NUTS(model_fn),
            num_warmup=2_000,
            num_samples=10_000,
            num_chains=5,
            progress_bar=True,
        )

        sampler.run(jax.random.PRNGKey(RANDOM_SEED))
        samples = sampler.get_samples(group_by_chain=False)

        return samples