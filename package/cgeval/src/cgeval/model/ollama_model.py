import json
import requests

from cgeval import Model

class OllamaModel(Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def generate(self, inputs):
        payload = {
            'model': self.cfg.model.name,
            'messages': [
                {
                    'role': 'user',
                    'content': inputs
                }
            ],
            'stream': False
        }

        x = requests.post(self.cfg.model.url, json=payload)
        response = json.loads(x.text)

        return response['message']['content']
