from tqdm.auto import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_recall_fscore_support

from cgeval import QuantificationMethod
from cgeval.report import MultiClassClassificationReport

class ClassifyAndCount(QuantificationMethod):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def eval(self, dataloader, classifier):
        actual = []
        predictions = []

        for batch in tqdm(dataloader):
            outputs = classifier.classify(batch['input'])

            actual.extend(batch['class'])

            if classifier.cfg.output == 'class':
                predictions.extend(outputs)
            if classifier.cfg.output == 'logits':
                # TODO: note that argmax doesn't return the logits but the model returns logits that need to be converted
                predictions.extend(outputs.logits.argmax(dim=1))

        return MultiClassClassificationReport(actual, predictions, labels=classifier.cfg.labels)
