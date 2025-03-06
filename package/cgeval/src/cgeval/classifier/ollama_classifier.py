import json
import requests

from cgeval import Classifier

class OllamaClassifier(Classifier):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def classify(self, inputs):
        predictions = []
        for input in inputs:
            content = f"Classify this text and assign it one of the following sentiment classes: {self.cfg.classifier.labels}. Only respond with one of the classes nothing else. [TEXT]{input}[/TEXT]"

            payload = {
                'model': self.cfg.classifier.name,
                'messages': [
                    {
                        'role': 'user',
                        'content': content
                    }
                ],
                'stream': False
            }

            x = requests.post(self.cfg.classifier.url, json=payload)
            response = json.loads(x.text)

            predictions.append(response['message']['content'])

        return predictions