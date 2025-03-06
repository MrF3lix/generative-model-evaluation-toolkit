from omegaconf import OmegaConf

from cgeval import Evaluation
from cgeval.method import ClassifyAndCount
from cgeval.dataset import HuggingfaceDataset
from cgeval.classifier import OllamaClassifier, TransformersClassifier

def main():
    cfg = OmegaConf.load("config.yaml")

    classifiers = []
    for classifier_cfg in cfg.classifier:
        # TODO: Move to a factory method
        if classifier_cfg.type == 'transformers':
            classifier = TransformersClassifier(classifier_cfg)
        elif classifier_cfg.type == 'ollama':
            classifier = OllamaClassifier(classifier_cfg)

        classifiers.append(classifier)

    method = ClassifyAndCount(cfg)
    dataset = HuggingfaceDataset(cfg, column_mapping={'text': 'input', 'sentiment': 'class'})

    eval = Evaluation()
    reports = eval.run(dataset.load(), classifiers, method)

    for classifier, report in zip(classifiers, reports):
        print(classifier.cfg.id)
        print(report)