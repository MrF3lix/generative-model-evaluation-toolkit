import re
import argparse
import json
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

from cgeval.method import ClassifyAndCount, StandardClassification, BCC, CPCC
from cgeval.rating import Ratings, Label, Observation


def load_evaluation_method(cfg):
    if cfg.evaluate.method == 'CC':
        return ClassifyAndCount()
    if cfg.evaluate.method == 'BCC':
        return BCC(cfg)
    if cfg.evaluate.method == 'CPCC':
        return CPCC(cfg)
    elif cfg.evaluate.method == 'Classification':
        return StandardClassification()

def label_name_to_id(name: str, labels) -> int:
    return next((l['id'] for l in labels if l['name'] == name), None)

def label_match_to_id(match: str, matching_label: str) -> int:
    # HACK: Why does this work for the stories?
    # return int(match == True) if match is not None else match
    # print(match)
    return int(match == matching_label) if match is not None else match
    # return int(match == 'animal_match') if match is not None else match

def extract_sentiment(input):
    m = re.search('The story should have a (.+?) sentiment', input)
    if m:
        found = m.group(1)
        return found

    return input

def load_ratings(cfg, classifier, report_path) -> Ratings:
    with open(f"{report_path}/{cfg.evaluate.out}/dataset_{classifier.id}.json", 'r') as f:
        ratings_data = json.load(f)

    if cfg.quantify.comparison == 'binary':

        if classifier.output == 'logits':
            label2id = lambda label: list(map(lambda l: l['name'], classifier.labels)).index(label)
            
            df = pd.DataFrame(ratings_data)
            df['condition'] = df['input'].apply(extract_sentiment)
            df['condition_id'] = df['condition'].apply(label2id)
            df['oracle'] = df.apply(lambda r: r['oracle'] == r['condition'] if(pd.notnull(r['oracle'])) else r['oracle'], axis=1)

            df['X'] = df.apply(lambda row: row['metric'][int(row['condition_id'])],axis=1)
            df['y'] = df['oracle']

            observations = df.to_dict(orient='records')
            observations = list(map(lambda i: Observation(
                id=i['id'],
                output=i['output'],
                input=i['condition_id'],
                oracle=i['y'],
                metric=i['X']
            ), observations))

        else:
            df = pd.DataFrame(ratings_data)
            # HACK: Only works with the Story Dataset
            # df['condition'] = df['input'].apply(extract_sentiment)


            # TODO: Does this happen on 
            # df['oracle'] = df.apply(lambda r: r['oracle'] == r['condition'] if(pd.notnull(r['oracle'])) else r['oracle'], axis=1)
            # df['metric'] = df['metric'] == df['condition']

            observations = df.to_dict(orient='records')
            observations = list(map(lambda i: Observation(
                id=i['id'],
                output=i['output'],
                input=1,
                oracle=label_match_to_id(i['oracle'], classifier.labels[0].name),
                metric=label_match_to_id(i['metric'], classifier.labels[0].name)
            ), observations))

        # HACK: only for the sentiment stories
        # labels = [Label(0, 'no match'), Label(1, 'match')]
        labels = list(map(lambda l: Label(**l), classifier.labels))
    
    else:
        observations = list(map(lambda i: Observation(
            id=i['id'],
            output=i['output'],
            input=label_name_to_id(i['input'], classifier.labels),
            oracle=label_name_to_id(i['oracle'], classifier.labels),
            metric=label_name_to_id(i['metric'], classifier.labels)
        ), ratings_data))

        labels = list(map(lambda l: Label(**l), classifier.labels))

    return Ratings(labels=labels, observations=observations)

def quantify(cfg, report_path):
    method = load_evaluation_method(cfg)

    reports = []
    for classifier in cfg.classifier:
        ratings = load_ratings(cfg, classifier, report_path)
         
        out = f"{report_path}/{cfg.quantify.out}"
        Path(out).mkdir(parents=True, exist_ok=True)

        report = method.quantify(ratings, classifier.id)
        report.save(f"{out}/cls_report_{classifier.id}.json")
        reports.append(report)
        
        print(classifier.id)
        print(report)

def main():
    parser = argparse.ArgumentParser(description="A toolkit for robust evaluation of generative models.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.yaml"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    report_path = f"{cfg.experiment.report_path}/quantify/{now}_{cfg.experiment.name}"
    Path(report_path).mkdir(parents=True, exist_ok=True)

    with open(f"{report_path}/config.yaml", 'w') as f:
        OmegaConf.save(cfg, f)

    quantify(cfg, report_path)