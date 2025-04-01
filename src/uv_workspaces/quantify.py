import argparse
import json
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

from cgeval.method import ClassifyAndCount, StandardClassification, BCC
from cgeval.rating import Ratings, Label, Observation


def load_evaluation_method(cfg):
    if cfg.evaluation.method == 'CC':
        return ClassifyAndCount()
    if cfg.evaluation.method == 'BCC':
        return BCC()
    elif cfg.evaluation.method == 'Classification':
        return StandardClassification()

def label_name_to_id(name: str, labels) -> int:
    return next((l['id'] for l in labels if l['name'] == name), None)

def load_ratings(cfg, classifier) -> Ratings:
    with open(f"{cfg.evaluation.out}/dataset_{classifier.id}.json", 'r') as f:
        ratings_data = json.load(f)

    observations = list(map(lambda i: Observation(
        id=i['id'],
        output=i['output'],
        input=label_name_to_id(i['input'], classifier.labels),
        oracle=label_name_to_id(i['oracle'], classifier.labels),
        metric=label_name_to_id(i['metric'], classifier.labels)
    ), ratings_data))

    labels = list(map(lambda l: Label(**l), classifier.labels))

    return Ratings(labels=labels, observations=observations)

def main():
    parser = argparse.ArgumentParser(description="A toolkit for robust evaluation of generative models.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.yaml"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    report_path = f"{cfg.experiment.report_path}/quantify/{now}_{cfg.experiment.name}"
    Path(report_path).mkdir(parents=True, exist_ok=True)

    with open(f"{report_path}/config.yaml", 'w') as f:
        OmegaConf.save(cfg, f)

    method = load_evaluation_method(cfg)

    for classifier in cfg.classifier:
        ratings = load_ratings(cfg, classifier)

        report = method.quantify(ratings)
        report.save(f"{report_path}/cls_report_{classifier.id}.json")
        
        print(classifier.id)
        print(report)