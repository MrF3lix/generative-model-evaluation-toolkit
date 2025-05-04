import argparse
import json
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from cgeval.classifier import OllamaClassifier, TransformersClassifier
from cgeval.rating import Ratings, Observation, Label

def load_classifiers(cfg):
    classifiers = []
    for classifier_cfg in cfg.classifier:
        if classifier_cfg.type == 'transformers':
            classifier = TransformersClassifier(classifier_cfg)
        elif classifier_cfg.type == 'ollama':
            classifier = OllamaClassifier(classifier_cfg)
        classifiers.append(classifier)

    return classifiers

def collate(l):
    return list(map(lambda o: o.__dict__, l))

def load_ratings(cfg, labels) -> Ratings:
    with open(cfg.annotate.out, 'r') as f:
        dataset = json.load(f)

    observations = list(map(lambda i: Observation(**i), dataset))
    labels = list(map(lambda l: Label(**l), labels))

    return Ratings(labels=labels, observations=observations)

def evaluate(cfg, report_path):
    classifiers = load_classifiers(cfg)

    for classifier in classifiers:
        print(classifier.cfg.id)
        ratings = load_ratings(cfg, classifier.cfg.labels)
        dataloader = DataLoader(ratings.observations, batch_size=cfg.dataset.batch_size, shuffle=False, collate_fn=collate)
        dataset = classifier.classify(dataloader)

        out = f"{report_path}/{cfg.evaluate.out}"
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(f"{out}/dataset_{classifier.cfg.id}.json", 'w') as f:
            json.dump(dataset, f, sort_keys=True, indent=2)

def main():
    parser = argparse.ArgumentParser(description="A toolkit for robust evaluation of generative models.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.yaml"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    report_path = f"{cfg.experiment.report_path}/evaluate/{now}_{cfg.experiment.name}"
    Path(report_path).mkdir(parents=True, exist_ok=True)

    with open(f"{report_path}/config.yaml", 'w') as f:
        OmegaConf.save(cfg, f)

    evaluate(cfg, report_path)