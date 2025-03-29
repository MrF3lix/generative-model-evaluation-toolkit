import argparse
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

from cgeval.method import ClassifyAndCount, StandardClassification
from cgeval.dataset import HuggingfaceDataset, LocalTextDataset
from cgeval.classifier import OllamaClassifier, TransformersClassifier


def load_evaluation_method(cfg):
    if cfg.evaluation.method == 'CC':
        return ClassifyAndCount()
    elif cfg.evaluation.method == 'Classification':
        return StandardClassification()

def load_dataset(cfg):
    if cfg.dataset.type == 'hf':
        return HuggingfaceDataset(cfg, column_mapping={'text': 'input', 'sentiment': 'class'})
    elif cfg.dataset.type == 'local_text':
        return LocalTextDataset(cfg, column_mapping={'output': 'input', 'input': 'class'})

def load_classifiers(cfg):
    classifiers = []
    for classifier_cfg in cfg.classifier:
        if classifier_cfg.type == 'transformers':
            classifier = TransformersClassifier(classifier_cfg)
        elif classifier_cfg.type == 'ollama':
            classifier = OllamaClassifier(classifier_cfg)

        classifiers.append(classifier)

    return classifiers

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

    classifiers = load_classifiers(cfg)
    method = load_evaluation_method(cfg)
    dataset = load_dataset(cfg)
    dataloader = dataset.load()

    for classifier in classifiers:
        # TODO: Think of where this conversion needs to take place? Is this the responsibility of the model?
        label_name_to_id = np.vectorize(lambda m: next((l['id'] for l in classifier.cfg.labels if l['name'] == m), None))

        inputs = dataloader.dataset['class']
        inputs = label_name_to_id(inputs)

        metric_ratings = classifier.classify(dataloader)
        metric_ratings = label_name_to_id(metric_ratings)

        report = method.quantify(
            inputs=inputs,
            metric_ratings=metric_ratings,
            oracle_ratings=inputs,
            labels=classifier.cfg.labels
        )

        report.save(f"{report_path}/cls_report_{classifier.cfg.id}.json")
        
        print(classifier.cfg.id)
        print(report)