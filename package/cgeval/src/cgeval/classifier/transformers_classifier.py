import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

from cgeval import Classifier

class TransformersClassifier(Classifier):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg.name)

    def classify(self, dataloader):
        metric_ratings = []

        for batch in tqdm(dataloader):
            model_input = list(map(lambda x: x['output'], batch))

            inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                output = self.model(**inputs)

            
            if self.cfg.output == 'class':
                labels = list(map(lambda o: o['label'], output))

                for i in range(len(batch)):
                    metric_ratings.append({
                        **batch[i],
                        'metric': labels[i]
                    })

            if self.cfg.output == 'logits':
                logits = output['logits']
                desired_order = list(map(lambda l: l['name'], self.cfg.labels))
                label2id = {v: k for k, v in self.model.config.id2label.items()}
                new_order = [label2id[label] for label in desired_order]
                logits_reordered = logits[:, new_order]

                for i in range(len(batch)):
                    metric_ratings.append({
                        **batch[i],
                        'metric': logits_reordered[i].tolist()
                    })

        return metric_ratings