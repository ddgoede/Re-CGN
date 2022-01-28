"""
Performs experiments on IN-mini dataset.
This primarily replicates the numbers of Table 3 and 4 in the paper
"""
import json
import os
from os.path import join, exists, isdir
from subprocess import call
from glob import glob
import pandas as pd
import torch

import warnings
warnings.filterwarnings("ignore")

from experiment_utils import set_env, REPO_PATH, seed_everything, dotdict
set_env()

from imagenet_eval_ood_benchmark import eval_ood


def generate_counterfactual_dataset(
        prefix,
        modes=["train",
        "val"],
        trunc=0.5,
        n_train=34745,
        n_val=3923,
        seed=0,
    ):
    """Generates CF dataset for ImageNet (size of IN-mini)"""
    seed_everything(seed)

    script_path = join(REPO_PATH, "cgn_framework/imagenet/generate_data.py")

    # generate train and val dataset
    for mode in modes:        
        run_name = f"{prefix}_{mode}_trunc_{trunc}"
        n_samples = eval(f"n_{mode}")

        data_root = join(REPO_PATH, "cgn_framework/imagenet/data", run_name)
        ims = glob(join(data_root, "ims/*.jpg"))

        if isdir(data_root) and len(ims) == n_samples:
            print("")
            print(f"{mode.capitalize()} dataset exists with {n_samples} images, skipping...")
            print(f"Path to dataset: {data_root}")
            print("")
        else:
            print("Generating {} dataset...".format(mode))
            print("WARNING: This will take about 3 hours for train set and 20 mins for validation set.")
            arguments = "--mode random --weights_path imagenet/weights/cgn.pth"\
                f" --n_data {n_samples} --run_name {prefix}-{mode} --truncation {trunc} --batch_sz 1"\
                f" --ignore_time_in_filename"
            cmd = f"python {script_path} {arguments}"
            call(cmd, shell=True)


def train_classifier(args: dict = dict(lr=0.001), prefix="in-mini", seed=0, disp_epoch=45, show=False):
    """Trains classifier on IN-mini dataset"""

    args = dotdict(args)
    seed_everything(seed)

    run_name = f"{prefix}-classifier"
    expt_dir = join(REPO_PATH, "cgn_framework/imagenet/experiments", f"classifier__{run_name}")
    epoch_metrics_path = join(expt_dir, f"epochwise_metrics/epoch_{disp_epoch}.pt")
    if not exists(epoch_metrics_path):
        
        print("::::: Training classifier :::::")
        script_path = join(REPO_PATH, "cgn_framework/imagenet/train_classifier.py")

        # all arguments used are defaults given in their repo/paper
        arguments = f"-a resnet50 -b 32 --lr {args.lr} -j 6 --pretrained"\
            f" --data imagenet/data/in-mini --cf_data imagenet/data/{prefix}"\
            f" --name {run_name} --seed {seed} --ignore_time_in_filename"
        cmd = f"python {script_path} {arguments}"
        call(cmd, shell=True)
    
    else:
        print("::::: Classifier already trained, skipping :::::")

    print(f"Loading results for epoch {disp_epoch} from {epoch_metrics_path}")
    metrics = torch.load(epoch_metrics_path)
    if show:
        print(json.dump(metrics, indent=4))
    
    return metrics


def run_eval_on_ood_benchmarks(seed=0, ignore_cache=False, show=False):
    classifiers=["cgn-ensemble", "resnet50"]
    datasets=["in-mini", "in-a", "in-stylized", "in-sketch"]

    df = pd.DataFrame(columns=datasets, index=classifiers)

    for classifier in classifiers:
        for dataset in datasets:
            print(f"::::: Running {classifier} on {dataset}...")

            weight_path = None
            if dataset == "in-a" and classifier == "resnet50":
                # modify temporarily
                classifier = "resnet50-from-scratch"
                # this should be downloaded from a script in setup
                weight_path = "experiments/weights/resnet50_from_scratch_model_best.pth.tar"
            
            if classifier == "cgn-ensemble":
                weight_path = "cgn_framework/imagenet/experiments/classifier__in-mini-classifier/model_best.pth"

            args = dict(
                seed=seed,
                classifier=classifier,
                ood_dataset=dataset,
                num_workers=2,
                batch_size=128,
                ignore_cache=ignore_cache,
            )
            if weight_path is not None:
                args["weight_path"] = weight_path
            args = dotdict(args)
            result = eval_ood(args, show=show)

            if dataset == "in-a" and classifier == "resnet50-from-scratch":
                classifier = "resnet50"

            df.at[classifier, dataset] = result["acc1"]

    return df


def run_experiments(seed=0, disp_epoch=45):
    """Runs experiments on IN-mini dataset

    1. Generates CF dataset
    2. Runs classifier experiments
    """
    seed_everything(seed)

    # step 1: generate dataset
    print("\n::::: Generating CF dataset :::::\n")
    generate_counterfactual_dataset(prefix="in-mini", seed=seed)

    # step 2: train classifier
    print("\n::::: Training classifier :::::\n")
    metrics = train_classifier(prefix="in-mini", seed=seed, disp_epoch=disp_epoch)

    # step 3: evaluate on OOD benchmarks
    print("\n::::: Evaluating OOD :::::\n")
    df_ood = run_eval_on_ood_benchmarks(seed=seed, show=False)

    return metrics, df_ood



if __name__ == "__main__":
    run_experiments(seed=0, disp_epoch=6)