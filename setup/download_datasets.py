"""Script to download all required datasets at apt location."""
import os
from os.path import join, exists, isdir
import sys
import shutil
import subprocess
import tarfile

from experiments.experiment_utils import set_env, REPO_PATH
set_env()

if __name__ == "__main__":
    # download colored MNIST, IN-9, Cue-conflict datasets (following their script)
    print("::: Downloading datasets: colored MNIST | IN-9 | Cue-conflict ...")
    subprocess.call("bash scripts/download_data.sh", shell=True)

    # download other datasets
    print("::: Downloading IN-mini ...")
    zip_file = join(REPO_PATH, "cgn_framework/imagenet/data/archive.zip")
    if not exists(zip_file):
        raise Exception(
            f"Zip file not found: {zip_file}"\
            "Please download it from https://www.kaggle.com/ifigotin/imagenetmini-1000"\
            "and place it in the cgn_framework/imagenet/data/archive.zip"
        )

    subprocess.call(f"unzip {zip_file}", shell=True)

