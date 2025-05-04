import argparse
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

from cgeval.report import GenericReport, plot_triangle, plot_binary

def load_reports(cfg, report_path):
    reports = []

    for classifier in cfg.classifier:
        report = GenericReport()
        report.load(f"{report_path}/{cfg.quantify.out}/cls_report_{classifier.id}.json")
        reports.append(report)

    return reports

def plot(cfg, report_path):

    reports = load_reports(cfg, report_path)

    out = f"{report_path}/{cfg.plot.out}"
    Path(out).mkdir(parents=True, exist_ok=True)

    if cfg.quantify.comparison == 'binary':
        fig_obs = plot_binary(reports, cfg.classifier, 'alpha_obs', 'Observed Distribution')
        fig_obs.savefig(f"{out}/observed_distribution.png", bbox_inches='tight')

        fig = plot_binary(reports, cfg.classifier, 'alpha', 'Corrected Distribution')
        fig.savefig(f"{out}/corrected_distribution.png", bbox_inches='tight')
    else:
        fig = plot_triangle(cfg.classifier[0].labels, reports, cfg.classifier, ['p_true'], desired_dist=[0.35, 0.3, 0.35])

        fig.savefig(f"{out}/ternary_plot.png")

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