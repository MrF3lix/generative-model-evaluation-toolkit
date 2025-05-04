import argparse
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

from . import evaluate, quantify, plot

def main():
    parser = argparse.ArgumentParser(description="A toolkit for robust evaluation of generative models.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.yaml"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    # now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    now = "2025-04-30_15-26"
    report_path = f"{cfg.experiment.report_path}/pipeline/{now}_{cfg.experiment.name}"
    Path(report_path).mkdir(parents=True, exist_ok=True)

    with open(f"{report_path}/config.yaml", 'w') as f:
        OmegaConf.save(cfg, f)

    # evaluate(cfg, report_path)
    # quantify(cfg, report_path)
    plot(cfg, report_path)