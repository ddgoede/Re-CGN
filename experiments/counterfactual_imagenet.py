import torchvision
from torchvision.utils import make_grid

from matplotlib import pyplot as plt
from datetime import datetime
import os

from experiment_utils import set_env, dotdict
set_env()

from cgn_framework.imagenet.generate_data import main as generate_main

def generate_counterfactual(**kwargs):
    args = dotdict(kwargs)
    generate_main(args)

    return args


def main():
    args = generate_counterfactual(mode="random", n_data=100, run_name="RUN_NAME", weights_path="imagenet/weights/cgn.pth",
                                   batch_sz=1, truncation=0.5, classes=[0, 0, 0], interp="", interp_cls=-1,
                                   midpoints=6, save_noise=False, save_single=False)
    examples_count = 6

    time_str = datetime.now().strftime("%Y_%m_%d_%H_")
    trunc_str = f"{args.run_name}_trunc_{args.truncation}"
    data_path = os.path.join('imagenet', 'data', time_str + trunc_str)
    images_dir = os.path.join(data_path, 'ims')

    # Get the locations of the generated images
    image_paths = (images_dir + "/" + path for path, _ in zip(os.listdir(images_dir), range(examples_count)))

    # Construct a grid with the generated images
    images = [torchvision.io.read_image(path) for path in image_paths]
    image_grid = make_grid(images, nrow=examples_count)

    plt.imshow(image_grid.permute(1, 2, 0))
    plt.show()

if __name__ == "__main__":
    main()
