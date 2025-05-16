import re
import base64
import requests
from tqdm.auto import tqdm

from cgeval import Classifier

def get_animal(input):
    animal = input.split(' ')[1]
    animal = animal.rstrip("s")
    return animal

def get_count(input):
    count = input.split(' ')[0]
    return int(count)

class OllamaImageClassifier(Classifier):
    def __init__(self, cfg, image_base_path):
        super().__init__()
        self.cfg = cfg
        self.image_base_path = image_base_path

    def get_image_path(self, item):
        image_name = item['output'].split('/')[-1]
        return f'{self.image_base_path}/{image_name}'

    def classify(self, dataloader):


        metric_ratings = []
        for batch in tqdm(dataloader):
            for input in batch:

                prompt = f"What animal is shown and how many are there? Format your answer to be a short string in this format: ###<COUNT> <ANIMAL>###"

                image_path = self.get_image_path(input)
                with open(image_path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode("utf-8")

                payload = {
                    'model': self.cfg.name,
                    'prompt': prompt,
                    'images': [image_base64],
                    'stream': False
                }

                response = requests.post(self.cfg.url, json=payload)
                result = response.json()

                response = result.get("response")

                try:
                    search_pattern = r'(\d+) ([A-Za-z]+)'
                    m = re.search(search_pattern, response)

                    target_string = m.group()

                    count = int(target_string.split(' ')[0])
                    animal = target_string.split(' ')[1]
                except:
                    count = 0
                    animal = 'NOTHING'

                condition_animal = get_animal(input['input'])
                condition_count = get_count(input['input'])

                if self.cfg.label_task == 'count':
                    label = 'count_no_match'
                    if count == condition_count:
                        label = 'count_match'

                elif self.cfg.label_task == 'animal':
                    label = 'animal_no_match'
                    if animal == condition_animal:
                        label = 'animal_match'

                metric_ratings.append({
                    **input,
                    'metric': label
                })

        return metric_ratings