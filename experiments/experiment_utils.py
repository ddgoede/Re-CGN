import os, sys
from os.path import join, dirname, abspath
import itertools

def set_env():
    """This function does two things:
    1. It allows the code in cgn_framework to be imported from the experiments folder.
    2. It changes the current working directory to cgn_framework/. This is necessary,
    because the code present in this directory expects to be called from here."""
    sys.path.insert(0, dirname(dirname(abspath(__file__))))
    os.chdir(join(dirname(dirname(abspath(__file__))), "cgn_framework"))

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_generated_imagenet(images_dir, images_count=None):
    import torchvision

    # Get the locations of the generated images
    image_range = range(images_count) if images_count is not None else itertools.count()
    image_paths = (images_dir + "/" + path for path, _ in zip(os.listdir(images_dir), image_range))

    return [torchvision.io.read_image(path) for path in image_paths]