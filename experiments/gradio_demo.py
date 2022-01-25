import re
from os.path import join, basename, dirname
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
from tqdm import tqdm
import torch

from torchvision.io import read_image
from torchvision.utils import make_grid
from torchvision import transforms

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import gradio as gr

import warnings
warnings.filterwarnings("ignore")

from experiment_utils import set_env, REPO_PATH, seed_everything
set_env()

from image_utils import denormalize, show_single_image, show_multiple_images
from cgn_framework.imagenet.dataloader import get_imagenet_dls
from cgn_framework.imagenet.models.classifier_ensemble import InvariantEnsemble
from cgn_framework.imagenet.models import CGN
from experiments.imagenet_utils import (
    EnsembleGradCAM,
    get_imagenet_mini_foldername_to_classname,
)


def display_dummy_image(*inputs: str):
    print(inputs)
    image = np.random.randn(224, 224, 3)
    image = (image - image.min()) / (image.max() - image.min())
    return image


class CGNGradio:
    """Class that defines the interface for the CGNGradio"""
    def __init__(self, x=1):
        self.x = x
        self.wordnet_id_to_class_label = get_imagenet_mini_foldername_to_classname(
            join(REPO_PATH, "cgn_framework/imagenet/data/in-mini/metadata.txt")
        )
    
    def configure_input(self):
        """Defines all the input elements."""
        dataset_selector = gr.inputs.Dropdown(
            choices=["ImageNet-Mini", "Counterfactuals"],
            type="value",
            label="Dataset",
        )
        original_label_selector = gr.inputs.Dropdown(
            choices=sorted(self.wordnet_id_to_class_label.values()),
            type="value",
            label="Original label",
        )


        shape_label_selector = gr.inputs.Dropdown(
            choices=sorted(self.wordnet_id_to_class_label.values()),
            type="value",
            label="Shape label",
        )
        texture_label_selector = gr.inputs.Dropdown(
            choices=sorted(self.wordnet_id_to_class_label.values()),
            type="value",
            label="Texture label",
        )
        background_label_selector = gr.inputs.Dropdown(
            choices=sorted(self.wordnet_id_to_class_label.values()),
            type="value",
            label="Background label",
        )
        return [
            dataset_selector,
            original_label_selector,
            shape_label_selector,
            texture_label_selector,
            background_label_selector,
        ]

    def configure_output(self):
        return "image"
    
    def launch(self):
        inputs = self.configure_input()
        outputs = self.configure_output()
        self.iface = gr.Interface(fn=display_dummy_image, inputs=inputs, outputs=outputs)
        self.iface.launch()


if __name__ == "__main__":
    CGNGradio().launch()