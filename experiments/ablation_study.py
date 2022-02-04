from experiment_utils import ImageDirectoryLoader
from inception_score import inception_score, mu_mask, generate_images

def run_experiments():
    for loss_name in [
        "shape-ablation",
        "text-ablation",
        "bg-ablation",
        "rec-ablation",
    ]:
        data_dir = generate_images(f'imagenet/weights/{loss_name}.pth', loss_name)

        images = ImageDirectoryLoader(data_dir + '/ims')
        inception = inception_score(images, splits=2, resize=True)
        avg_mask, sd_mask = mu_mask(data_dir + '/mean_masks.txt')

        yield loss_name, inception, avg_mask, sd_mask
