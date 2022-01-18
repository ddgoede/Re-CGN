import experiment_utils
experiment_utils.set_env()

from inception_score_pytorch.inception_score import inception_score
from experiment_utils import load_generated_imagenet

import argparse
import torch
import os


def generate_images():
    raise NotImplementedError

def main(args):
    if not os.path.exists(args.image_dir):
        generate_images()

    images = load_generated_imagenet(args.image_dir, args.images_count)
    inception = inception_score(images, splits=10)

    print(inception)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True,
                        help='Folder to load the images from. If the images '
                        'do not exist in this folder, the program genetates '
                        'new images and stores those in this folder.')

    parser.add_argument('--images_count', default=None, type=int,
                        help='Number of images to load')
    args = parser.parse_args()

    main(args)
