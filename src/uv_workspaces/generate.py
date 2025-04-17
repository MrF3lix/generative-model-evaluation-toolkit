import argparse
import json
from pathlib import Path
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
            'count': int(cfg.model.samples * label['ratio'])
        }

    return labels

def load_dataset(path):
    if not Path(path).is_file():
        return []
    with open(path, 'r') as f:
        dataset = json.load(f)

    return dataset

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

    # TODO: Load dataset
    # TODO: For each sample in the dataset generate the output.

    dataset = load_dataset(cfg.generate.input)
    dataset_out = load_dataset(f"{report_path}/dataset.json")

    for sample in tqdm(dataset):

        processed = next((item for item in dataset_out if item['id'] == sample['id']), None)
        if processed is not None:
            continue

        output = model.generate([sample['input']])
        dataset_out.append({
            'id': sample['id'],
            'input': sample['input'],
            'metric': None,
            'oracle': None,
            'output': output[0]
        })

        if len(dataset_out) % 10 == 0:
            with open(f"{report_path}/dataset.json", 'w') as f:
                json.dump(dataset_out, f, indent=2, sort_keys=True)
    
    # labels = get_labels_and_distributions(cfg)

    # dataset = []
    # for name, label in labels.items():
    #     progress_bar = tqdm(total=cfg.model.samples)
    #     progress_bar.set_description(f"Generate samples for label {name}")

    #     for _ in range(label['count']):
    #         progress_bar.update(1)

    #         prompt = cfg.model.base_prompt.replace('###', name)
    #         output = model.generate([prompt])

    #         dataset.append({
    #             'id': abs(hash(output)) % (10 ** 8),
    #             'input': name,
    #             'output': output
    #         })

    with open(f"{report_path}/dataset.json", 'w') as f:
        json.dump(dataset_out, f, indent=2, sort_keys=True)
