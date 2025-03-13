import argparse
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

from cgeval import Evaluation
from cgeval.method import ClassifyAndCount, Classification
from cgeval.dataset import HuggingfaceDataset, LocalTextDataset
from cgeval.classifier import OllamaClassifier, TransformersClassifier

def load_evaluation_method(cfg):
    if cfg.evaluation.method == 'CC':
        return ClassifyAndCount(cfg)
    elif cfg.evaluation.method == 'Classification':
        return Classification(cfg)

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

    eval = Evaluation()

    for classifier in classifiers:
        print(classifier.cfg.id)
        report = eval.run(dataloader, classifier, method)
        print(report)
        report.save(f"{report_path}/cls_report_{classifier.cfg.id}.json")
