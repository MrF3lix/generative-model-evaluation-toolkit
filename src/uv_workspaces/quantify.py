from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context
import os
import itertools
import argparse
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

def is_label_positive(label: str, positive_label: str) -> int:
    """
    Returns `1` if the label equals the positive label `positive_label` else `0`.
    Returns `None` if the label is `None`
    """

    return int(label == positive_label) if label is not None else label

def load_subsample_ratings(cfg, classifier, report_path, subsampling) -> pd.DataFrame:
    B, M = subsampling

    df = pd.read_json(f"{report_path}/{cfg.evaluate.out}/dataset_{classifier.id}.json", orient='records')

    df_oracle = df.loc[~df['oracle'].isna()]
    df_metric_only = df.loc[df['oracle'].isna()]

    # df_oracle_subsample = df_oracle.sample(B if B < len(df_oracle) else len(df_oracle), random_state=0xdeadbeef)
    # df_metric_only_subsample = df_metric_only.sample(M if M < len(df_metric_only) else len(df_metric_only), random_state=0xdeadbeef)
    df_oracle_subsample = df_oracle.sample(B if B < len(df_oracle) else len(df_oracle))
    df_metric_only_subsample = df_metric_only.sample(M if M < len(df_metric_only) else len(df_metric_only))

    return pd.concat([df_oracle_subsample, df_metric_only_subsample])


def load_ratings(cfg, classifier, report_path, subsampling=None) -> Ratings:
    if subsampling is not None:
        df = load_subsample_ratings(cfg, classifier, report_path, subsampling)
    else:
        df = pd.read_json(f"{report_path}/{cfg.evaluate.out}/dataset_{classifier.id}.json", orient='records')

    # TODO: Document that the positive label has the index 1
    positive_label = classifier.labels[1]

    observations = df.to_dict(orient='records')
    observations = list(map(lambda i: Observation(
        id=i['id'],
        output=i['output'],
        input=1,
        oracle=is_label_positive(i['oracle'], positive_label.name),
        metric=is_label_positive(i['metric'], positive_label.name)
    ), observations))

    labels = list(map(lambda l: Label(**l), classifier.labels))

    return Ratings(labels=labels, observations=observations)

def compute_subsampling(args):
    cfg, classifier, method, report_path, ratings, subsampling, i = args
    report = method.quantify(ratings, classifier.id)

    out = f"{report_path}/{cfg.quantify.out}/subsampling"
    Path(out).mkdir(parents=True, exist_ok=True)
    report.save(f"{out}/cls_report_{classifier.id}_{subsampling[0]}_{subsampling[1]}_{i}.json")

def quantify(cfg, report_path):
    method = load_evaluation_method(cfg)

    # if pool_size > 1:
    #     print(exp_desc)
    #     with get_context('spawn').Pool(pool_size) as pool:
    #         n_tasks_per_chunk = ceil(len(experiments) / len(pool._pool))
    #         # results = list(tqdm(p.imap(runner.run, experiments, chunksize=n_tasks_per_chunk), total=len(experiments), desc=exp_desc))
    #         start_time = time.time()
    #         results = list(pool.imap(runner.run, experiments, chunksize=n_tasks_per_chunk))
    #         elapsed_time = time.time() - start_time
    #         print(f'Experiment Done. Elapsed Time: {elapsed_time:.3f}')
    #         pool.close()
    #         pool.join()
    # else:
    #     results = [runner.run(ex) for ex in tqdm(experiments, desc=exp_desc)]


    # TODO: Load all subsampling DFs
    if 'subsampling' in cfg.quantify:

        with get_context('spawn').Pool(20) as pool:
            tasks = []
            for classifier in cfg.classifier:
                if 'subsampling' in cfg.quantify:
                    B = cfg.quantify.subsampling.B
                    M = cfg.quantify.subsampling.M
                    repeat = cfg.quantify.subsampling.repeat

                    for subsampling in list(itertools.product(B, M)):
                        for i in range(repeat):
                            print(classifier.id, subsampling, i)

                            ratings = load_ratings(cfg, classifier, report_path, subsampling)
                            tasks.append((cfg, classifier, method, report_path, ratings, subsampling, i))


            print('Submitted', len(tasks), 'Tasks')
            pool.map(compute_subsampling, tasks)
            print('Done')

    else:
        out = f"{report_path}/{cfg.quantify.out}"
        Path(out).mkdir(parents=True, exist_ok=True)

        for classifier in cfg.classifier:
            ratings = load_ratings(cfg, classifier, report_path)   
                

            report = method.quantify(ratings, classifier.id)
            report.save(f"{out}/cls_report_{classifier.id}.json")
        
            print(classifier.id)
            print(report)

def main():
    parser = argparse.ArgumentParser(description="A toolkit for robust evaluation of generative models.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.yaml"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    now = datetime.today().strftime('%Y-%m-%d')
    report_path = f"{cfg.experiment.report_path}/quantify/{now}_{cfg.experiment.name}"
    Path(report_path).mkdir(parents=True, exist_ok=True)

    with open(f"{report_path}/config.yaml", 'w') as f:
        OmegaConf.save(cfg, f)

    quantify(cfg, report_path)