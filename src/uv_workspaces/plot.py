import argparse
import itertools
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from cgeval.report import GenericReport, plot_binary

def load_reports(cfg, report_path, subsampling=None):
    reports = []

    for classifier in cfg.classifier:
        if subsampling is not None:
            B, M = subsampling
            report = GenericReport()
            report.load(f"{report_path}/{cfg.quantify.out}/subsampling/cls_report_{classifier.id}_{B}_{M}.json")
        else:
            report = GenericReport()
            report.load(f"{report_path}/{cfg.quantify.out}/cls_report_{classifier.id}.json")

        reports.append(report)

    return reports

def plot_comparison(cfg, out, reports):
    fig_obs = plot_binary(reports, cfg.classifier, 'alpha_obs', 'Observed Distribution')
    fig_obs.savefig(f"{out}/observed_distribution.png", bbox_inches='tight')
    plt.close(fig_obs)

    fig = plot_binary(reports, cfg.classifier, 'alpha', 'Corrected Distribution')
    fig.savefig(f"{out}/corrected_distribution.png", bbox_inches='tight')
    plt.close(fig)

def plot(cfg, report_path):
    if 'subsampling' in cfg.quantify:
        B = cfg.quantify.subsampling.B
        M = cfg.quantify.subsampling.M

        for subsampling in list(itertools.product(B, M)):
            reports = load_reports(cfg, report_path, subsampling)

            out = f"{report_path}/{cfg.plot.out}/subsampling/{subsampling[0]}_{subsampling[1]}"
            Path(out).mkdir(parents=True, exist_ok=True)
            plot_comparison(cfg, out, reports)


    reports = load_reports(cfg, report_path)

    out = f"{report_path}/{cfg.plot.out}"
    Path(out).mkdir(parents=True, exist_ok=True)
    plot_comparison(cfg, out, reports)


def main():
    parser = argparse.ArgumentParser(description="A toolkit for robust evaluation of generative models.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.yaml"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    report_path = f"{cfg.experiment.report_path}/plots/{now}_{cfg.experiment.name}"
    Path(report_path).mkdir(parents=True, exist_ok=True)

    with open(f"{report_path}/config.yaml", 'w') as f:
        OmegaConf.save(cfg, f)

    plot(cfg, report_path)