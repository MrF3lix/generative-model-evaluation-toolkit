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
            model_input = list(map(lambda x: x['output'][:512], batch))
            output = self.pipe(model_input)

            labels = list(map(lambda x: x['label'], output))
            for i in range(len(batch)):
                metric_ratings.append({
                    **batch[i],
                    'metric': labels[i]
                })

        return metric_ratings