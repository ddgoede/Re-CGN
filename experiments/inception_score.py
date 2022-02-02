from experiment_utils import set_env, dotdict
set_env()

from inception_score_pytorch.inception_score import inception_score
from experiment_utils import load_generated_imagenet, ImageDirectoryLoader

from torch.utils.data import Dataset, DataLoader, TensorDataset

import argparse
import torch
import os


def generate_images(weights_path, run_name):
    from cgn_framework.imagenet.generate_data import main as generate_main

    args = dotdict({
        "mode": "random_same",
        "weights_path": weights_path,
        "ignore_time_in_filename": True,
        "n_data": 10_000,
        "run_name": run_name,
        "truncation": 0.5,
        "batch_sz": 1,
    })

    final_dir_name = f"imagenet/data/{run_name}_trunc_{args.truncation}/ims"

    if os.path.exists(final_dir_name):
        print("Generated data already exists. It will be used instead of regenerated.")
    else:
        generate_main(args)

    return final_dir_name

def main(args):
    if args.image_dir is None:
        assert args.run_name is not None and args.weights_path is not None, "if the image_dir argument is not supplied, supply run_name and weight_path"
        image_dir = generate_images(args.weights_path, args.run_name)
    else:
        image_dir = args.image_dir

    # images = (image / 128.0 - 1 for image in load_generated_imagenet(image_dir, args.images_count))
    images = ImageDirectoryLoader(image_dir)
    inception = inception_score(images, splits=2, resize=True)

    print(inception)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Folder to load the images from. If the images '
                        'do not exist in this folder, the program genetates '
                        'new images and stores those in this folder.')
    parser.add_argument('--weights_path', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    # parser.add_argument('--images_count', default=None, type=int,
    #                     help='Number of images to load')

    args = parser.parse_args()

    main(args)
