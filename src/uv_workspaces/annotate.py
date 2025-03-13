import argparse

def main():
    parser = argparse.ArgumentParser(description="A toolkit for robust evaluation of generative models.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.yaml"
    )
    args = parser.parse_args()
    print(args)

    print('Annotate: NOT IMPLEMENTED YET')


# Input is a table with |I|O|
# Output is a table with |I|O|W| => Oracle Ratings