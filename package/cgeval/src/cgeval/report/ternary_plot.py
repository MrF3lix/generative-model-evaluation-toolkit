import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from mpltern.datasets import get_triangular_grid

from cgeval import Report

def plot_triangle(labels, reports: list[Report], classifiers, fields, desired_dist=None, colors = ['Blue', 'Green', 'Red'], title=None):
    t, l, r = get_triangular_grid(50, 1e-6)
    x = np.stack((t, l, r), axis=-1)

    legend_elements = []
    if desired_dist is not None:
        legend_elements.append(
            Patch(facecolor='Black', edgecolor='k', alpha=1.0, label='Desired Dist')
        )

    for idx, cls in enumerate(classifiers):
        color = colors[idx]
        legend_elements.append(
            Patch(facecolor=color, edgecolor='k', alpha=0.5, label=cls.id)
        )

    fig, ax = plt.subplots(subplot_kw={'projection': 'ternary'})
    shading = "gouraud"

    def plot_for_field(field):
        for idx, cls in enumerate(reports):
            p = cls.samples[field]
            color = colors[idx]

            v = stats.gaussian_kde(p.T).pdf(x.T)

            ax.tripcolor(t, l, r, v, cmap=f"{color}s", shading=shading, alpha=0.5, label='Metric Only')
            ax.tricontour(t, l, r, v, colors=color, linewidths=0.5, alpha=1)

    for field in fields:
        plot_for_field(field)


    for idx, cls in enumerate(reports):
        count_metric_ratings = []
        color = colors[idx]
        for label in labels:
            name = label.name
            c = cls.report[name]['count_metric_ratings']
            count_metric_ratings.append(c)

        ax.scatter(count_metric_ratings[0], count_metric_ratings[1], count_metric_ratings[2], color=color, s=50, alpha=0.75, marker='x')


    ax.taxis.set_major_locator(MultipleLocator(0.10))
    ax.laxis.set_major_locator(MultipleLocator(0.10))
    ax.raxis.set_major_locator(MultipleLocator(0.10))

    ax.taxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")

    ax.grid(axis='t', which='both', linestyle='--')
    ax.grid(axis='l', which='both', linestyle='--')
    ax.grid(axis='r', which='both', linestyle='--')

    if desired_dist is not None:
        ax.scatter(desired_dist[0], desired_dist[1], desired_dist[2], color='black', s=50, label='Desired Dist', alpha=0.75)

    if title is not None:
        fig.title(title)

    ax.set_tlabel(labels[0].name)
    ax.set_llabel(labels[1].name)
    ax.set_rlabel(labels[2].name)

    ax.legend(handles=legend_elements, loc='upper right', title='Legend', bbox_to_anchor=(1.05, 1.05))

    return fig