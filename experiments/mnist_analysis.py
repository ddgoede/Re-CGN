"""Performs additional analysis on MNIST variants."""
import re
import os
from os.path import join
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
from tqdm import tqdm
from sklearn.manifold import TSNE

import torch
from torchvision.io import read_image
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

from experiment_utils import set_env, REPO_PATH, seed_everything
set_env()

from cgn_framework.mnists.models.classifier import CNN
from cgn_framework.mnists.train_cgn import save
from cgn_framework.mnists.dataloader import get_tensor_dataloaders, TENSOR_DATASETS

# set plotting configuration
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


def get_model_features(model, dl, device, num_batches_to_use=None):
    iterator = tqdm(
        dl,
        desc="Extracting features",
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    )
    fvecs = []
    labels = []
    ctr = 0
    for (data, label) in iterator:
        data = data.to(device)
        label = label.to(device)

        fvec = model(data)

        fvecs.append(fvec.cpu())
        labels.append(label.cpu())
        
        if ctr == num_batches_to_use:
            break
        
        ctr += 1

    fvecs = torch.cat(fvecs, dim=0)
    labels = torch.cat(labels, dim=0)
    return fvecs, labels


def reduce_dimensionality(X, dim=2):
    assert len(X.shape) == 2
    N, D = X.shape
    assert D >= dim
    
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    tsne = TSNE(n_components=dim)
    Z = tsne.fit_transform(X)
    return Z


def create_df(Z, y):
    df = pd.DataFrame(None)
    df["Z1"] = Z[:, 0]
    df["Z2"] = Z[:, 1]
    df["y"] = y
    return df


def plot_features(
        df_original,
        df_counterfactual,
        dataset="colored_MNIST",
        model_desc="CNN classifier",
        save=True,
        show=False,
    ):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    ax[0].grid()
    ax[0].set_title("Original", fontsize=20)
    sns.scatterplot(data=df_original, x="Z1", y="Z2", hue="y", palette="deep", ax=ax[0])
    ax[0].legend(fontsize=16)

    ax[1].grid()
    ax[1].set_title("Counterfactual", fontsize=20)
    sns.scatterplot(data=df_counterfactual, x="Z1", y="Z2", hue="y", palette="deep", ax=ax[1])
    ax[1].legend(fontsize=16)

    plt.suptitle(f"Features for  {model_desc} ({dataset.replace('_', ' ')})", fontsize=25)
    if save:
        save_path = join(
            REPO_PATH,
            f"experiments/results/plots/feature_analysis_{model_desc.replace(' ', '_')}_{dataset}.pdf",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path, bbox_inches="tight", format="pdf")

    if show:
        plt.show()


class MNISTAnalysis:
    """
    Performs additional analyses on MNISTs.

    1. Visualizes features using t-SNE
    2. Visualize Grad-CAM on test set and compute IoU metric
    """
    def __init__(self, dataset, weight_path, seed=0) -> None:
        self._check_args(dataset, weight_path, seed)
        self.dataset = dataset
        self.weight_path = weight_path

        seed_everything(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _check_args(self, dataset, weight_path, seed):
        """Checks arguments"""
        assert dataset in TENSOR_DATASETS, f"{dataset} is not a valid dataset"
        assert os.path.exists(weight_path), f"{weight_path} does not exist"
        assert isinstance(seed, int)
    
    def visualize_feature(self, num_batches_to_use=20, show=False, save=True):
        # load dataloaders
        print("Loading datasets...")
        dl_og_train, dl_og_test = get_tensor_dataloaders(dataset=f"{self.dataset}")
        dl_cf_train, dl_cf_test = get_tensor_dataloaders(dataset=f"{self.dataset}_counterfactual")

        # load model
        print("Loading model...")
        model = CNN()
        model.load_state_dict(torch.load(self.weight_path, map_location="cpu"))
        model.cls = torch.nn.Identity()
        model = model.eval().to(self.device)

        # get features
        features_og, y_og = get_model_features(
            model, dl_og_test, self.device, num_batches_to_use=num_batches_to_use,
        )
        print("Original features extracted of shape {}".format(features_og.shape))
        features_cf, y_cf = get_model_features(
            model, dl_cf_test, self.device, num_batches_to_use=num_batches_to_use,
        )
        print("Counterfactual features extracted of shape {}".format(features_cf.shape))

        # reduce dimensionality
        print("Reducing dimensionality...")
        Z_og = reduce_dimensionality(features_og)
        Z_cf = reduce_dimensionality(features_cf)

        # create dataframe
        df_og = create_df(Z_og, y_og)
        df_cf = create_df(Z_cf, y_cf)

        # plot
        model_type = 'original' if not 'counterfactual' in self.weight_path else 'counterfactual'
        plot_features(
            df_og,
            df_cf,
            dataset=self.dataset,
            model_desc=f"CNN classifier trained on {model_type}",
            save=save,
            show=show,
        )


if __name__ == "__main__":
    mnist_analysis = MNISTAnalysis(
        dataset="colored_MNIST",
        weight_path=join(
            REPO_PATH,
            "cgn_framework/mnists/experiments",
            # "classifier_colored_MNIST_counterfactual_seed_0/weights/ckp_epoch_10.pth",
            "classifier_colored_MNIST_seed_0/weights/ckp_epoch_10.pth",
        ),
        seed=0,
    )
    mnist_analysis.visualize_feature(save=True, show=False, num_batches_to_use=5)