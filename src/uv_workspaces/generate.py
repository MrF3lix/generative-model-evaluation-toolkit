import argparse
import json
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

from cgeval.model import OllamaModel

def get_labels_and_distributions(cfg):
    labels = {}
    for label in cfg.model.labels:
        name = label['name']
        labels[name] = {
            'id': label['id'],
            'name': name,
            # TODO: This is very optimistic, better would be to sum up all the ratios and distribute it correctly
            'count': int(cfg.model.samples * label['ratio'])
        }

    return labels

def main():
    """Generates a dataset using a provided model, class distributions, and expected sample size.

    Outputs a list of conditions I and the corresponding model output O.
    """
    parser = argparse.ArgumentParser(description="A toolkit for robust evaluation of generative models.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.yaml"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    
    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    report_path = f"{cfg.experiment.report_path}/generate/{now}_{cfg.experiment.name}"
    Path(report_path).mkdir(parents=True, exist_ok=True)

    model = OllamaModel(cfg)
    labels = get_labels_and_distributions(cfg)

    dataset = []
    for name, label in tqdm(labels.items()):
        for _ in range(label['count']):
            prompt = cfg.model.base_prompt.replace('###', name)
            output = model.generate([prompt])

            dataset.append({
                'input': name,
                'output': output
            })

    with open(f"{report_path}/dataset.json", 'w') as f:
        json.dump(dataset, f, indent=2, sort_keys=True)
