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


DATASETS = {"colored_MNIST": "cgn_colored_MNIST",
            "double_colored_MNIST": "cgn_double_colored_MNIST",
            "wildlife_MNIST": "cgn_wildlife_MNIST"
            }


def generate_counterfactual_images(dataset_size=100000, no_cfs=10):
    """Generate the counterfactual images for the 3 datasets."""
    for dataset, weight_folder in DATASETS.items():
        generated_images_path = f"./cgn_framework/mnists/data/{dataset}_counterfactual.pth"

        if os.path.exists(generated_images_path):
            print(f"Counterfactual images for {dataset} already exist, skipping..")
            continue

        command = f"""python ./mnists/generate_data.py \
                        --weight_path ./mnists/experiments/{weight_folder}/weights/ckp.pth \
                        --dataset {dataset} --no_cfs {no_cfs} --dataset_size {dataset_size}"""

        working_directory = os.getcwd()
        os.chdir(f"{working_directory}/cgn_framework")
        print(os.system(command))
        os.chdir(working_directory)


def main():
    generate_counterfactual_images()

    for dataset, weight_folder in DATASETS.items():
        generated_images_path = f"./cgn_framework/mnists/data/{dataset}_counterfactual.pth"
        visualise_generated_images(generated_images_path)

def visualise_generated_images(path):
    images, labels = torch.load(path)

    # Transform the image range [-1, 1] to the range [0, 1]
    images = images * 0.5 + 0.5

    digits_count = 4
    examples_per_digit = 3

    shown_examples = torch.empty((digits_count * examples_per_digit,) + images.shape[1:])

    # Pick example images, with the first row containing 0's, the second containing 1's, etc.
    for digit in range(digits_count):
        i = digit * examples_per_digit
        shown_examples[i:i + examples_per_digit] = images[labels == digit][:examples_per_digit]

    # Rows and columns are swapped because matplotlib has different axes
    grid = make_grid(shown_examples, nrow=examples_per_digit)

    #TODO: Explain the permute.
    plt.imshow(grid.permute((1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    main()
