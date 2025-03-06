
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification

from cgeval import Classifier

class ViTClassifier(Classifier):
    def __init__(self, cfg):
        super().__init__()


        self.feature_extractor = ViTFeatureExtractor.from_pretrained(cfg.classifier.feature_extractor)
        self.model = ViTForImageClassification.from_pretrained(cfg.classifier.name)

    def classify(self, inputs):
        with torch.no_grad():
            inputs = self.feature_extractor(images=inputs, return_tensors="pt")
            outputs = self.model(**inputs)

        return outputs