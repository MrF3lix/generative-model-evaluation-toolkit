from torch.utils.data import DataLoader
from cgeval import Classifier, QuantificationMethod, Report

class Evaluation():
    def __init__(self):
        pass

    # TODO: Can we run the classifiers in parallel if we don't need to load different models?
    def run(self, dataloader: DataLoader, classifiers: list[Classifier], quantification: QuantificationMethod) -> list[Report]:
        # TODO: Recored the inference time for the evaluation
        reports = []
        for classifier in classifiers:
            report = quantification.eval(dataloader, classifier)
            reports.append(report)
        return reports
    
    def run(self, dataloader: DataLoader, classifier: Classifier, quantification: QuantificationMethod) -> Report:
        report = quantification.eval(dataloader, classifier)
        return report