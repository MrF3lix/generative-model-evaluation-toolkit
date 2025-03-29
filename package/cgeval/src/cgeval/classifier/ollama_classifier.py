import json
import requests
import numpy as np
from tqdm.auto import tqdm

from cgeval import Classifier

class OllamaClassifier(Classifier):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def classify(self, dataloader):
        metric_ratings = []
        for batch in tqdm(dataloader):
            model_input = list(map(lambda x: x[:512], batch['input']))
            for input in model_input:
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
                metric_ratings.append(response['message']['content'].replace('[', '').replace(']', ''))

        return np.array(metric_ratings)