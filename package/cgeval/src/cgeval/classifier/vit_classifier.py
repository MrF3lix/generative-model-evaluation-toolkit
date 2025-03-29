import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification

from cgeval import Classifier

class ViTClassifier(Classifier):
    def __init__(self, cfg):
        super().__init__()


        self.feature_extractor = ViTFeatureExtractor.from_pretrained(cfg.feature_extractor)
        self.model = ViTForImageClassification.from_pretrained(cfg.name)

    def classify(self, dataloader):
        metric_ratings = []
        for batch in tqdm(dataloader):
            model_input = list(map(lambda x: x, batch['input']))
            with torch.no_grad():
                inputs = self.feature_extractor(images=model_input, return_tensors="pt")
                outputs = self.model(**inputs)
                metric_ratings.extend(outputs)

        return np.array(metric_ratings)