import argparse
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

from cgeval.report import GenericReport, plot_triangle

def load_reports(cfg):
    reports = []

    for classifier in cfg.classifier:
        report = GenericReport()
        report.load(f"{cfg.quantify.out}/cls_report_{classifier.id}.json")
        reports.append(report)

    return reports

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

    reports = load_reports(cfg)

    fig = plot_triangle(cfg.classifier[0].labels, reports, cfg.classifier, ['p_true'])
    fig.savefig(f"{report_path}/ternary_plot.png")
