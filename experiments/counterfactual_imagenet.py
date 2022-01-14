import torch
import torchvision
from torchvision.utils import make_grid

from matplotlib import pyplot as plt

import os

def main():
    examples_count = 6

    run_dir = 'cgn_framework/imagenet/data/2022_01_14_16_RUN_NAME_trunc_0.5' # This needs to be an argument or somehow inferred

    # Get the locations of the generated images
    images_dir = run_dir + '/ims'
    image_paths = (images_dir + "/" + path for path, _ in zip(os.listdir(images_dir), range(examples_count)))

    # Construct a grid with the generated images
    images = [torchvision.io.read_image(path) for path in image_paths]
    image_grid = make_grid(images, nrow=examples_count)

    plt.imshow(image_grid.permute(1, 2, 0))
    plt.show()

if __name__ == "__main__":
    main()
