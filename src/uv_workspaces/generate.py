import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

from cgeval.model import OllamaModel, DiffusionModel

def load_dataset(path):
    if not Path(path).is_file():
        return []
    with open(path, 'r') as f:
        dataset = json.load(f)

    return dataset

def load_model(cfg, report_path):
    if cfg.model.type == 'ollama':
        return OllamaModel(cfg)
    if cfg.model.type == 'diffuision':
        return DiffusionModel(cfg, f'{report_path}/img')
    if cfg.model.type == 'flux':
        return DiffusionModel(cfg, f'{report_path}/img')

def generate(cfg, report_path):
    model = load_model(cfg, report_path)

    dataset = load_dataset(cfg.generate.input)
    dataset_out = load_dataset(cfg.generate.temp) if 'temp' in cfg.generate else []

    for sample in tqdm(dataset):

        processed = next((item for item in dataset_out if item['id'] == sample['id']), None)
        if processed is not None:
            continue

        output = model.generate(sample['id'], [sample['input']])
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

    with open(f"{report_path}/dataset.json", 'w') as f:
        json.dump(dataset_out, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="A toolkit for robust evaluation of generative models.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.yaml"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    report_path = f"{cfg.experiment.report_path}/generate/{now}_{cfg.experiment.name}"
    Path(report_path).mkdir(parents=True, exist_ok=True)

    with open(f"{report_path}/config.yaml", 'w') as f:
        OmegaConf.save(cfg, f)

    generate(cfg, report_path)