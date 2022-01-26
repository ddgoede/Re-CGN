"""Defines a pipeline for an end-to-end experiment on MNISTs."""
import os
import sys
import argparse
import json
from glob import glob
from subprocess import call
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import torchvision.datasets as tv_datasets

import warnings
warnings.filterwarnings("ignore")

from experiment_utils import set_env, REPO_PATH, seed_everything, dotdict
set_env()

from cgn_framework.mnists.dataloader import TENSOR_DATASETS
from cgn_framework.mnists.config import get_cfg_defaults
from cgn_framework.utils import load_cfg


class MNISTPipeline:
    """
    Pipeline to run *a single experiment* on an MNIST variant.

    Step 1: (Optional) Train a GAN/CGN model on the given dataset.
    Step 2: (Optional) Generate data using trained/given GAN/CGN/None model.
            In case of None, tensor for original dataset is generated.
    Step 3: Train a classifier on the generated data.

    Each Step saves the results to a directory and is not run if cached results exist.
    If generate=False, the pipeline will run for original dataset.
    """
    def __init__(self, args, train_generative=False, generate=True) -> None:
        """
        Initialize the pipeline.

        Args:
            args: Arguments for the experiment.
            train_generative: Whether to train a GAN/CGN model.
            generate: Whether to generate data.
        """
        self.train_generative = train_generative
        self.generate = generate
        self.args = self._check_args(args)
        print("::::: Experimental setup :::::")
        print("Train generative model:", self.train_generative)
        print("Generate data:", self.generate)
        print("Args:", self.args)

    def _check_args(self, args):
        """Check the arguments."""
        assert isinstance(args.seed, int)
        if args.dataset not in TENSOR_DATASETS:
            raise ValueError("The dataset is not supported.")

        if self.train_generative:
            assert hasattr(args, "cfg") and args.cfg is not None, \
                "You need to pass config file for training GAN/CGN."\
                    "If not, set train_generative to False."
            
            # check if given dataset matches the given config
            assert args.dataset in args.cfg, \
                "The dataset in args.dataset does not match the dataset in args.cfg."

            # NOTE: this is config used only for training generative model
            self.gencfg = load_cfg(args.cfg) if args.cfg else get_cfg_defaults()
    
        if self.generate and not self.train_generative:
            if not os.path.exists(args.weight_path):
                raise FileNotFoundError(
                    "The weight path for given CGN/GAN model does not exist."
                )
        
        if not self.generate and not self.train_generative:
            args.combined = False

        return args

    def train_generative_model(self):
        """Train a GAN/CGN model on the given dataset and returns path to best ckpt."""

        if self.train_generative:
            generative_model = "gan" if "gan" in self.args.cfg else "cgn"
            cmd = f"python {REPO_PATH}/cgn_framework/mnists/train_{generative_model}.py "\
                f"--cfg {self.args.cfg} --ignore_time_in_filename"
            print(f"Running command: {cmd}")
            call(cmd, shell=True)
        
            ckpt_dir =  os.path.join(
                REPO_PATH,
                "cgn_framework/mnists/experiments",
                f"cgn_{self.gencfg.TRAIN.DATASET}__{self.gencfg.MODEL_NAME}",
                "weights"
            )
            weight_path = glob(os.path.join(ckpt_dir, "*.pth"))[-1]
        else:
            weight_path = self.args.weight_path

        return weight_path
    
    def generate_data(self, dataset, weight_path):
        """Generate data using trained/given GAN/CGN/None model."""
        if self.generate:
            cmd = f"python {REPO_PATH}/cgn_framework/mnists/generate_data.py --dataset {dataset}"
            if "gan" in weight_path or "cgn" in weight_path:
                cmd += f" --weight_path {weight_path}"
            print(cmd)
            call(cmd, shell=True)
    
    def train_classifier(self, seed, dataset, dataset_suffix="", combined=False):
        """Train a classifier on the generated data."""

        # extract classifier results
        expt_suffix = (dataset) if not combined else (dataset + "_combined")
        expt_suffix += "_seed_" + str(seed) if seed is not None else ""
        results_path = f'mnists/experiments/classifier_{expt_suffix}/test_accs.pth'

        if not os.path.exists(results_path):
            cmd = f"python {REPO_PATH}/cgn_framework/mnists/train_classifier.py"\
                f" --dataset {dataset}{dataset_suffix} --seed {seed}"
            print(cmd)
            call(cmd, shell=True)
        else:
            print(f"Results for classifier already exist: {results_path}")

        results = torch.load(results_path)
        return results

    def run(self):
        """Main experiment runner."""
        seed_everything(self.args.seed)

        # train generative model
        weight_path = self.train_generative_model()

        # generate data
        self.generate_data(self.args.dataset, weight_path)

        # train classifier
        dataset_suffix = ""
        if self.generate:
            dataset_suffix = "_counterfactual" if f"cgn_{self.args.dataset}" in weight_path else "_gan"

        results = self.train_classifier(
            self.args.seed, self.args.dataset, dataset_suffix, self.args.combined,
        )
        return results


if __name__ == "__main__":
    # # training classifier on original dataset
    # args = dict(
    #     seed=0,
    #     dataset="colored_MNIST",
    # )
    # pipeline = MNISTPipeline(
    #     args=dotdict(args), train_generative=False, generate=False,
    # )
    # pipeline.run()

    # # train classifier on already generated GAN/CGN data
    # args = dict(
    #     seed=0,
    #     # dataset="colored_MNIST_gan",
    #     dataset="colored_MNIST_counterfactual",
    # )
    # pipeline = MNISTPipeline(
    #     args=dotdict(args), train_generative=False, generate=False,
    # )
    # pipeline.run()

    # # generate GAN data -> train classifier on GAN data
    # args = dict(
    #     seed=0,
    #     dataset="colored_MNIST",
    #     weight_path="/home/lcur0478/piyush/projects/fact-team3/cgn_framework/mnists/experiments/gan_colored_MNIST/weights/ckp.pth",
    # )
    # pipeline = MNISTPipeline(
    #     args=dotdict(args), train_generative=False, generate=True,
    # )
    # pipeline.run()

    # # generate CGN data -> train classifier on CGN data
    # args = dict(
    #     seed=0,
    #     dataset="colored_MNIST",
    #     weight_path="/home/lcur0478/piyush/projects/fact-team3/cgn_framework/mnists/experiments/cgn_colored_MNIST/weights/ckp.pth",
    # )
    # pipeline = MNISTPipeline(
    #     args=dotdict(args), train_generative=False, generate=True,
    # )
    # pipeline.run()

    # train CGN -> generate CGN data -> train classifier on CGN data
    args = dict(
        seed=0,
        dataset="colored_MNIST",
        # weight_path="/home/lcur0478/piyush/projects/fact-team3/cgn_framework/mnists/experiments/cgn_colored_MNIST/weights/ckp.pth",
        cfg="/home/lcur0478/piyush/projects/fact-team3/cgn_framework/mnists/experiments/cgn_colored_MNIST/cfg.yaml",
    )
    pipeline = MNISTPipeline(
        args=dotdict(args), train_generative=True, generate=True,
    )
    pipeline.run()

