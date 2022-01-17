"""
Authors: Jesse Maas, Paul Hilders, Piyush Bagad, Danilo de Goede

counterfactual_mnist.py

This program generates counterfactual images for the MNIST dataset
and plots them. Datasets include colored MNIST, double-coloured MNIST
and wilflife MNIST.
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


datasets = ["colored_MNIST_counterfactual", "double_colored_MNIST_counterfactual", "wildlife_MNIST_counterfactual"]

def calc_test_accuracy(**kwargs):
    args = dotdict(kwargs)
    return classifier_main(args)


def plot_figure7():
    with open("../experiments/figure7_data/mnist_ablation_study_results2.txt", 'rb') as fp:
        results = pickle.load(fp)

    CF_ratios = [1, 5, 10]
    dataset_sizes = [10000, 100000, 1000000]

    fig, axs = plt.subplots(1, 3, figsize=(16,5))
    plt.setp(axs, xticks=[0, 1, 2], xticklabels=[r'$10^4$', r'$10^5$', r'$10^6$'])
    fig.suptitle('Figure 7 reproduced', fontsize=14)

    for i, dataset in enumerate(datasets):
        for CF_ratio in CF_ratios:
            line = []
            for size in dataset_sizes:
                line.append(results[f"{dataset}_{size}_{CF_ratio}"])
            axs[i].plot(np.arange(3), line, label=f'CF ratio = {CF_ratio}', marker='o')
            axs[i].set_xlabel("Num Counterfactual Datapoints")
            axs[i].set_ylabel("Test Accuracy (%)")
            axs[i].grid()
            axs[i].legend()
        break
    plt.plot()
    plt.savefig('../experiments/figure7_data/figure7_reproduced.png')



def main():
    CF_ratios = [1, 5, 10, 20]
    dataset_sizes = [10000, 100000, 1000000]
    accuracies = {}

    for dataset in datasets:
        for (dataset_size, no_cfs) in list(itertools.product(dataset_sizes, CF_ratios)):
            # For colored MNIST, the maximum CF ratio is 10 as there are only 10 possible colors per shape.
            if dataset == "colored_MNIST_counterfactual" and no_cfs == 20:
                continue

            print(f"Generating counterfactual dataset with dataset-size of {dataset_size} and CF-ratio of {no_cfs}...")
            generate_counterfactual(dataset_size, no_cfs, skip_generation=False)

            # Train classifier using the default parameters (taken from the argument parser in train_classifier.py)
            test_accuracy = calc_test_accuracy(dataset=dataset, batch_size=64, epochs=10, lr=1.0, gamma=0.7, log_interval=100)
            accuracies[f"{dataset}_{dataset_size}_{no_cfs}"] = test_accuracy

            with open(f"mnist_ablation_study_results2.txt", 'wb') as fp:
                pickle.dump(accuracies, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open("mnist_ablation_study_results.txt", 'wb') as fp:
        pickle.dump(accuracies, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # main()
    plot_figure7()
