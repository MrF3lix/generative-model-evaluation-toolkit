import re
import json
import argparse
import prodigy
from omegaconf import OmegaConf


def extract_title(input):
    m = re.search('"(.+?)"', input)
    if m:
        found = m.group(1)
        return found

    return input


def get_prodigy_item(row):
    title = extract_title(row['input'])

    return {
        'text': row['output'][0].replace(f'{title}', '').replace('\n', '').replace('""', ''),
        'label': title,
        'meta': {
            'id': row['id']
        }
    }

def load_your_source_here(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    return map(get_prodigy_item, dataset)

def add_options(stream):
    options = [
        {"id": "positive", "text": "🙂 Positive"},
        {"id": "neutral", "text": "😐 Neutral"},
        {"id": "negative", "text": "🙁 Negativ"},
        {"id": "match", "text": "✅ Title matches the Story"},
        {"id": "no_match", "text": "❌ Title does not matche the story"},
    ]
    for task in stream:
        task["options"] = options
        yield task

@prodigy.recipe("textcat.custom")
def custom_recipe_with_loader(dataset_path):
    stream = load_your_source_here(dataset_path)
    stream = add_options(stream)

    return {
        "dataset": dataset_path,
        "stream": stream,
        "view_id": "choice",
        "config": {
            "total_examples_target": 10,
            'choice_style': 'multiple'
        }
    }



def main():
    parser = argparse.ArgumentParser(description="A toolkit for robust evaluation of generative models.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.yaml"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)


    dataset_path = cfg.annotate.input
    
    prodigy.serve(f'textcat.custom {dataset_path}')


    # custom_recipe_with_loader


# Input is a table with |I|O|
# Output is a table with |I|O|W| => Oracle Ratings