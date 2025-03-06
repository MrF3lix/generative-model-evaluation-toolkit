from omegaconf import OmegaConf

from cgeval import Evaluation
from cgeval.method import ClassifyAndCount
from cgeval.dataset import HuggingfaceDataset
from cgeval.classifier import OllamaClassifier

def main():
    cfg = OmegaConf.load("config.yaml")

    classifier = OllamaClassifier(cfg)
    method = ClassifyAndCount(cfg)
    dataset = HuggingfaceDataset(cfg, column_mapping={'text': 'input', 'sentiment': 'class'})

    eval = Evaluation()
    report = eval.run(dataset.load(), classifier, method)
    print(report)