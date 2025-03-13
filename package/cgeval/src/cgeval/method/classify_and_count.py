from tqdm.auto import tqdm

from cgeval import QuantificationMethod
from cgeval.report import CountReport

class ClassifyAndCount(QuantificationMethod):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def eval(self, dataloader, classifier):
        inputs = []
        predictions = []

        for batch in tqdm(dataloader):
            # TODO: Solve this truncation to max sequence length in another way?
            model_input = list(map(lambda x: x[:512], batch['input']))
            outputs = classifier.classify(model_input)

            inputs.extend(batch['class'])

            if classifier.cfg.output == 'class':
                predictions.extend(outputs)
            if classifier.cfg.output == 'logits':
                # TODO: note that argmax doesn't return the logits but the model returns logits that need to be converted
                predictions.extend(outputs.logits.argmax(dim=1))

        return CountReport(inputs, predictions, labels=classifier.cfg.labels)
