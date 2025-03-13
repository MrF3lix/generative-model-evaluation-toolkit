from tqdm.auto import tqdm

from cgeval import QuantificationMethod
from cgeval.report import MultiClassClassificationReport

class Classification(QuantificationMethod):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def eval(self, dataloader, classifier):
        actual = []
        predictions = []

        for batch in tqdm(dataloader):
            try:
                # TODO: Solve this truncation to max sequence length in another way?
                inputs = list(map(lambda x: x[:512], batch['input']))
                outputs = classifier.classify(inputs)

                actual.extend(batch['class'])

                if classifier.cfg.output == 'class':
                    predictions.extend(outputs)
                if classifier.cfg.output == 'logits':
                    # TODO: note that argmax doesn't return the logits but the model returns logits that need to be converted
                    predictions.extend(outputs.logits.argmax(dim=1))
            except:
                print(batch['input'])
                raise

        return MultiClassClassificationReport(actual, predictions, labels=classifier.cfg.labels)
