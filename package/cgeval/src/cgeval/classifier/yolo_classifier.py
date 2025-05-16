import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from tqdm.auto import tqdm
from collections import Counter

from cgeval import Classifier

def get_animal(input):
    animal = input.split(' ')[1]
    animal = animal.rstrip("s")
    return animal

def get_count(input):
    count = input.split(' ')[0]
    return int(count)

class YoloClassifier(Classifier):
    def __init__(self, cfg, image_base_path):
        super().__init__()
        self.cfg = cfg
        self.image_base_path = image_base_path

        model_path = hf_hub_download(repo_id=self.cfg.name, filename=self.cfg.file)
        self.model = YOLO(model_path)

    def load_image(self, item):
        image_name = item['output'].split('/')[-1]
        image = Image.open(f'{self.image_base_path}/{image_name}')
        return image

    def classify(self, dataloader):
        metric_ratings = []

        for batch in tqdm(dataloader):
            model_input = list(map(lambda x: self.load_image(x), batch))
            results = self.model(model_input, verbose=False)

            labels = []
            for idx, result in enumerate(results):


                if self.cfg.output == 'class':
                    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]

                    res = Counter(names)
                    
                    # HACK: Very Specific to the tasks
                    if self.cfg.label_task == 'count':
                        condition_animal = get_animal(batch[idx]['input'])
                        condition_count = get_count(batch[idx]['input'])
                        animal_count = res[condition_animal]

                        label = 'count_no_match'
                        if animal_count == condition_count:
                            label = 'count_match'
                        
                        labels.append(label)

                    elif self.cfg.label_task == 'animal':
                        condition_animal = get_animal(batch[idx]['input'])
                        condition_count = get_count(batch[idx]['input'])

                        animal_count = res[condition_animal]

                        label = 'animal_no_match'
                        if animal_count > 0:
                            label = 'animal_match'

                        labels.append(label)
                elif self.cfg.output == 'logits':

                    print(results)
                    raise Exception('Done')

            for i in range(len(batch)):
                metric_ratings.append({
                    **batch[i],
                    'metric': labels[i]
                })

        return metric_ratings