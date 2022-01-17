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

from experiment_utils import set_env, dotdict
from counterfactual_mnist import generate_counterfactual
set_env()

from cgn_framework.mnists.generate_data import generate_cf_dataset, generate_dataset, get_dataloaders
from cgn_framework.mnists.train_classifier import main as classifier_main


datasets = ["colored_MNIST_counterfactual", "double_colored_MNIST_counterfactual", "wildlife_MNIST_counterfactual"]

def calc_test_accuracy(**kwargs):
    args = dotdict(kwargs)
    return classifier_main(args)

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

            # with open(f"mnist_ablation_study_results2.txt", 'wb') as fp:
            #     pickle.dump(accuracies, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open("mnist_ablation_study_results.txt", 'wb') as fp:
        pickle.dump(accuracies, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
