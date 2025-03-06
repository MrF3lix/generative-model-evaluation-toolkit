from transformers import pipeline

from cgeval import Classifier

class TransformersClassifier(Classifier):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pipe = pipeline("text-classification", model=self.cfg.name)

    def classify(self, inputs):
        predictions = self.pipe(inputs)
        return list(map(lambda x: x['label'], predictions))