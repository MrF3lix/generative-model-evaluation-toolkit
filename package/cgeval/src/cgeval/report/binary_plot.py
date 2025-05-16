import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import Counter
from scipy.stats import beta

from cgeval import Report
from cgeval.distribution import Beta

@dataclass(frozen=True)
class BetaParams:
    a: float
    b: float

    def __post_init__(self):
        if self.a < 0.:
            raise ValueError(f"parameter `a` of Beta distribution need to be > 0")
        if self.b < 0.:
            raise ValueError(f"parameter `b` of Beta distribution need to be > 0")


def fit_beta(samples: np.ndarray[float]) -> BetaParams:
    mu = np.mean(samples)
    var = np.var(samples)

    total_evidence = ((mu * (1. - mu)) / var) - 1.  # a + b

    a = mu * total_evidence
    b = (1. - mu) * total_evidence

    return BetaParams(a=a, b=b)

def get_mode(a,b):
    if a > 1 and b > 1:
        mode = (a - 1) / (a + b - 2)
    else:
        mode = 0 if a < 1 and b >= 1 else 1 
    return mode

def plot_oracle(ax, dist):
    a = dist.params.a
    b = dist.params.b

    x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
    pdf = beta.pdf(x, a, b)
    line, = ax.plot(x, pdf, color='orange', lw=1, alpha=.8, label='Human')
    ax.fill_between(x, pdf, color=line.get_color(), alpha=0.1)

    mode = get_mode(a, b)
    ax.axvline(mode, color=line.get_color(), linestyle='--', lw=1)


def plot_beta(ax, samples, label):
    dist = fit_beta(samples)

    x = np.linspace(beta.ppf(0.01, dist.a, dist.b), beta.ppf(0.99, dist.a, dist.b), 100)
    pdf = beta.pdf(x, dist.a, dist.b)
    line, = ax.plot(x, pdf, lw=1, alpha=.8, label=label)
    ax.fill_between(x, pdf, color=line.get_color(), alpha=0.1)

    mode = get_mode(dist.a, dist.b)
    ax.axvline(mode, color=line.get_color(), linestyle='--', lw=1)

def plot_binary(reports: list[Report], classifiers, field, title):
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set1.colors)

    fig = plt.figure(figsize=(8, 4))

    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1.set_title(title)
    ax_1.set_ylim(0,100)

    # oracle_rating_samples = np.array(vars(reports[0])['report']['match']['oracle_ratings'])
    # oracle_rating_samples = np.array(vars(reports[0])['report']['count_match']['oracle_ratings'])
    # oracle_rating_samples = np.array(vars(reports[0])['report']['animal_match']['oracle_ratings'])

    oracle_dist = vars(reports[0])['dist_report'][0]
    oracle_dist = Beta(params=BetaParams(oracle_dist['a'],oracle_dist['b']))

    plot_oracle(ax_1, oracle_dist)

    for idx, re in enumerate(reports):
        plot_beta(ax_1, re.samples[field], f'{classifiers[idx].id}')

    fig.legend(loc='upper right', title='Legend', bbox_to_anchor=(1.15, 0.85))
    return fig


def plot_binary_distributions(distributions: list[Beta], classifiers, field, title):
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set1.colors)

    fig = plt.figure(figsize=(8, 4))

    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1.set_title(title)
    ax_1.set_ylim(0,100)
