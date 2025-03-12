from omegaconf import OmegaConf
import argparse

from cgeval import Evaluation
from cgeval.method import ClassifyAndCount
from cgeval.dataset import HuggingfaceDataset
from cgeval.classifier import OllamaClassifier, TransformersClassifier

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

    classifiers = load_classifiers(cfg)
    method = ClassifyAndCount(cfg)
    dataset = HuggingfaceDataset(cfg, column_mapping={'text': 'input', 'sentiment': 'class'})

    eval = Evaluation()
    reports = eval.run(dataset.load(), classifiers, method)

    for classifier, report in zip(classifiers, reports):
        print(classifier.cfg.id)
        print(report)
