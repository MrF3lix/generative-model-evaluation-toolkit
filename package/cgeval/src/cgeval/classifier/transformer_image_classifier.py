import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from tqdm.auto import tqdm
from collections import Counter
from transformers import AutoImageProcessor, AutoModelForObjectDetection


from cgeval import Classifier

def get_animal(input):
    animal = input.split(' ')[1]
    animal = animal.rstrip("s")
    return animal

def get_count(input):
    count = input.split(' ')[0]
    return int(count)

class TransformerImageClassifier(Classifier):
    def __init__(self, cfg, image_base_path):
        super().__init__()
        self.cfg = cfg
        self.image_base_path = image_base_path

        self.processor = AutoImageProcessor.from_pretrained(self.cfg.name)
        self.model = AutoModelForObjectDetection.from_pretrained(self.cfg.name)

    def load_image(self, item):
        image_name = item['output'].split('/')[-1]
        image = Image.open(f'{self.image_base_path}/{image_name}')
        return image

    def classify(self, dataloader):
        metric_ratings = []

        for batch in tqdm(dataloader):
            model_input = list(map(lambda x: self.load_image(x), batch))

            inputs = self.processor(images=model_input, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_object_detection(outputs, threshold=0.9)

            labels = []
            for idx, result in enumerate(results):

                names = list(map(lambda id: self.model.config.id2label[id.item()], result["labels"]))

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

            for i in range(len(batch)):
                metric_ratings.append({
                    **batch[i],
                    'metric': labels[i]
                })

        return metric_ratings