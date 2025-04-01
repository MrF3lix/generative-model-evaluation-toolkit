import json
import requests
from tqdm.auto import tqdm

from cgeval import Classifier

class OllamaClassifier(Classifier):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def classify(self, dataloader):
        metric_ratings = []
        for batch in tqdm(dataloader):
            for input in batch:
                # TODO: Make sure that the base prompt is configurable

                label_names = list(map(lambda l: l['name'], self.cfg.labels))
                content = f"Classify this text and assign it one of the following sentiment classes: {label_names}. Only respond with one of the classes nothing else. [TEXT]{input['output'][:512]}[/TEXT]"

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
                predicted_label = response['message']['content'].replace('[', '').replace(']', '').replace("'", '').lower()

                metric_ratings.append({
                    **input,
                    'metric': predicted_label
                })


        return metric_ratings