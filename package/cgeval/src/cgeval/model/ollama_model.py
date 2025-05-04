import json
import requests

from cgeval import Model

class OllamaModel(Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def generate(self, id, inputs):
        predictions = []
        for input in inputs:
            payload = {
                'model': self.cfg.model.name,
                'messages': [
                    {
                        'role': 'user',
                        'content': input
                    }
                ],
                'stream': False
            }

            x = requests.post(self.cfg.model.url, json=payload)
            response = json.loads(x.text)

            predictions.append(response['message']['content'])

        return predictions
