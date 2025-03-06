from torch.utils.data import DataLoader
from . import Classifier, QuantificationMethod

class Evaluation():
    def __init__(self):
        pass

    def run(self, dataloader: DataLoader, classifiers: list[Classifier], quantification: QuantificationMethod):
        reports = []
        for classifier in classifiers:
            report = quantification.eval(dataloader, classifier)
            reports.append(report)
        return reports