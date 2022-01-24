"""Evaluates model performance on the imagenet-sketch benchmark."""
import re
from os.path import join, basename, dirname
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
from tqdm import tqdm
from PIL import Image
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
from torchvision import transforms

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

from experiment_utils import set_env, REPO_PATH, seed_everything
set_env()

from experiments.image_utils import denormalize, show_single_image
from experiments.imagenet_utils import IMModel, AverageEnsembleModel
from experiments.ood_utils import validate as ood_validate
from cgn_framework.imagenet.models.classifier_ensemble import InvariantEnsemble


if __name__ == "__main__":
    seed_everything(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load models
    model = InvariantEnsemble("resnet50", pretrained=True)
    ckpt_path = "imagenet/experiments/classifier_2022_01_19_15_36_sample_run/model_best.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_state_dict = ckpt["state_dict"]
    ckpt_state_dict = {k.replace("module.", ""):v for k, v in ckpt_state_dict.items()}
    model.load_state_dict(ckpt_state_dict)
    model = model.eval().to(device)

    shape_model = IMModel(base_model=model, mode="shape")
    avg_model = AverageEnsembleModel(base_model=model)
    pytorch_model = torchvision.models.resnet50(pretrained=True).to(device)
    only_backbone_model = torch.nn.Sequential(model.backbone, torch.nn.Flatten(), torch.nn.Linear(2048, 1000)).to(device)

    # check models
    x = torch.randn((1, 3, 224, 224)).to(device)

    y = shape_model(x)
    assert y.shape == torch.Size([1, 1000])

    y = avg_model(x)
    assert y.shape == torch.Size([1, 1000])

    y = pytorch_model(x)
    assert y.shape == torch.Size([1, 1000])

    y = only_backbone_model(x)
    assert y.shape == torch.Size([1, 1000])

    # load dataset
    valdir = join(join(REPO_PATH, "cgn_framework", "imagenet", "data/sketch"), 'val')
    combined_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, combined_transform),
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
    )

    print("::::::: Evaluating PyTorch ResNet50 model trained on ImageNet :::::::")
    # acc1 = ood_validate(val_loader, pytorch_model, gpu=device)
    # acc1 = ood_validate(val_loader, shape_model, gpu=device)
    # acc1 = ood_validate(val_loader, avg_model, gpu=device)
    acc1 = ood_validate(val_loader, only_backbone_model, gpu=device)
    