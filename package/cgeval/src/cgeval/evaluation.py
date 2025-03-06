from torch.utils.data import DataLoader
from . import Classifier, QuantificationMethod

class Evaluation():
    def __init__(self):
        pass

    def run(self, dataloader: DataLoader, classifier: Classifier, quantification: QuantificationMethod):
        return quantification.eval(dataloader, classifier)