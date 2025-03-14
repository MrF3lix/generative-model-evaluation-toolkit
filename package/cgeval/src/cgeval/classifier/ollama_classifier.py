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
            # TODO: Make sure that the base prompt is configurable
            content = f"Classify this text and assign it one of the following sentiment classes: {self.cfg.labels}. Only respond with one of the classes nothing else. [TEXT]{input}[/TEXT]"

            payload = {
                'model': self.cfg.name,
                'messages': [
                    {
                        'role': 'user',
                        'content': content
                    }
                ],
                'stream': False
            }

            x = requests.post(self.cfg.url, json=payload)
            response = json.loads(x.text)

            # TODO: Find a better way to extract the correct label from the response
            predictions.append(response['message']['content'].replace('[', '').replace(']', ''))

        return predictions