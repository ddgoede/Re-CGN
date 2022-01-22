"""Helper functions for image processing and visualizations."""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def denormalize(x: torch.Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalizes an image."""

    mean = np.array(mean)
    std = np.array(std)
    denormalize_transform = transforms.Normalize(
        mean=-(mean / std),
        std=(1.0 / std),
    )

    return denormalize_transform(x)


def show_single_image(x: torch.Tensor, figsize=None, normalized=True, title="Sample image"):
    """Displays a single image."""

    assert len(x.shape) in [1, 3]
    
    if normalized:
        x = denormalize(x)
    
    if x.shape[0] == 3:
        x = x.permute((1, 2, 0))
    elif x.shape[0] == 1:
        x = x.squeeze(dim=0)
    else:
        raise ValueError("Unsupported shape: {}".format(x.shape))

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(x)
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()