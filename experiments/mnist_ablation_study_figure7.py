"""
Authors: Jesse Maas, Paul Hilders, Piyush Bagad, Danilo de Goede

mnist_ablation_study_figure7.py

This program attempts to reproduce the MNIST Ablation Study experiment from
figure 7.
"""
import torch
from torchvision.utils import make_grid

from matplotlib import pyplot as plt

import os
import itertools
import pickle
import numpy as np

from experiment_utils import set_env, dotdict
from counterfactual_mnist import generate_counterfactual
set_env()

from cgn_framework.mnists.generate_data import generate_cf_dataset, generate_dataset, get_dataloaders
from cgn_framework.mnists.train_classifier import main as classifier_main


DATASETS = ["colored_MNIST_counterfactual", "double_colored_MNIST_counterfactual", "wildlife_MNIST_counterfactual"]
DATASET_NAMES = ["Colored MNIST", "Double-Colored MNIST", "Wildlife MNIST"]

def calc_test_accuracy(**kwargs):
    args = dotdict(kwargs)
    return classifier_main(args)


def plot_figure7():
    # Load the experiment results from the text file.
    with open("../experiments/figure7_data/mnist_ablation_study_results.txt", 'rb') as fp:
        results = pickle.load(fp)

    CF_ratios = [1, 5, 10, 20]
    dataset_sizes = [10000, 100000, 1000000]

    fig, axs = plt.subplots(1, 3, figsize=(16,5))
    plt.setp(axs, xticks=[0, 1, 2], xticklabels=[r'$10^4$', r'$10^5$', r'$10^6$'])

    for i, dataset in enumerate(DATASETS):
        for CF_ratio in CF_ratios:
            # Skip the CF_ratio of 20 for the colored MNIST dataset, as there are only 10 possible colors
            # per shape.
            if CF_ratio == 20 and dataset == "colored_MNIST_counterfactual":
                continue

            line = []
            for size in dataset_sizes:
                line.append(results[f"{dataset}_{size}_{CF_ratio}"])
            axs[i].plot(np.arange(3), line, label=f'CF ratio = {CF_ratio}', marker='o')
            axs[i].set_xlabel("Num Counterfactual Datapoints")
            axs[i].set_ylabel("Test Accuracy (%)")
            axs[i].grid(True)
            axs[i].legend()
        axs[i].set_title(DATASET_NAMES[i])
    plt.plot()
    plt.savefig('../experiments/figure7_data/figure7_reproduced.png', bbox_inches='tight')


def main():
    # The authors used 4 different CF_ratios and 3 different counterfactual dataset sizes for their
    # experiment.
    CF_ratios = [1, 5, 10, 20]
    dataset_sizes = [10000, 100000, 1000000]
    accuracies = {}

    for dataset in DATASETS:
        for (dataset_size, no_cfs) in list(itertools.product(dataset_sizes, CF_ratios)):
            # For colored MNIST, the maximum CF ratio is 10 as there are only 10 possible colors per shape.
            if dataset == "colored_MNIST_counterfactual" and no_cfs == 20:
                continue

            print(f"Generating counterfactual dataset with dataset-size of {dataset_size} and CF-ratio of {no_cfs}...")
            generate_counterfactual(dataset_size, no_cfs, skip_generation=False)

            # Train classifier using the default parameters (taken from the argument parser in train_classifier.py)
            test_accuracy = calc_test_accuracy(dataset=dataset, batch_size=64, epochs=10, lr=1.0, gamma=0.7, log_interval=100)
            accuracies[f"{dataset}_{dataset_size}_{no_cfs}"] = test_accuracy

            # Results are stored in a text file. As running the entire experiment in one run
            # is very time consuming, results are updated at every iteration.
            with open(f"mnist_ablation_study_results.txt", 'wb') as fp:
                pickle.dump(accuracies, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # main()
    plot_figure7()
