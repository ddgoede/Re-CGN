## Setup

### Conda environment

```bash
conda create -n cgn-gpu python=3.8
```


### Download datasets

Use the following command to download all required datasets:

```bash
cd /path/to/repo/
python setup/download_datasets.py
```
This should download datasets for both `mnists` and `imagenet` tasks.

For MNISTs, the folder structure is as follows:
```sh
mnists/data
├── colored_mnist
└── textures
    ├── background
    └── object

4 directories
```

For ImageNet, the folder structure is as follows:
```sh
imagenet/data
├── cue_conflict
├── in-a
├── in-mini
├── in-sketch
├── in-stylized
└── in9

6 directories
```

### Download model weights

Run the following command to download the model weights:

```bash
python setup/download_weights.py
```

This will download the weights for all tasks.

```bash
imagenet/weights/
├── biggan256.pth
├── cgn.pth
├── u2net.pth
└── resnet50_from_scratch_model_best.pth.tar

4 files
```

### Experiments for MNISTs

Please run the `final-demo.ipynb` notebook to reproduce the results for Table 2.
Further, the same notebook also has code to visualize additional analyses.

### Experiments for ImageNet-mini and OOD
Please run the `final-demo.ipynb` notebook to reproduce the results for Table 3, 4, 5.
Further, the same notebook also has code to visualize additional analyses.

