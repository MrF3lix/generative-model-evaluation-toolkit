import argparse
# import prodigy
from omegaconf import OmegaConf

# @prodigy.recipe("textcat.custom")
# def custom_recipe_with_loader(dataset, source):
#     # stream = load_your_source_here(source)  # implement your custom loading
#     # return {"dataset": dataset, "stream": stream, "view_id": "text"}
#     return None

def main():
    parser = argparse.ArgumentParser(description="A toolkit for robust evaluation of generative models.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.yaml"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)


    task = 'textcat.custom'
    dataset_name = 'TEST'
    labels = ['Positive', 'Neutral', 'Negative']

    # custom_recipe_with_loader


# Input is a table with |I|O|
# Output is a table with |I|O|W| => Oracle Ratings