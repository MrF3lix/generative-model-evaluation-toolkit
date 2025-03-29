import numpy as np
from transformers import pipeline
from tqdm.auto import tqdm

from cgeval import Classifier

class TransformersClassifier(Classifier):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pipe = pipeline("text-classification", model=self.cfg.name)

    def classify(self, dataloader):
        metric_ratings = []

        for batch in tqdm(dataloader):
            model_input = list(map(lambda x: x[:512], batch['input']))
            output = self.pipe(model_input)
            labels = list(map(lambda x: x['label'], output))
            metric_ratings.extend(labels)

        return np.array(metric_ratings)