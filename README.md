
<!-- Template source: https://github.com/paperswithcode/releasing-research-code -->
<!-- >📋  A template README.md for code accompanying a Machine Learning paper -->

# Replication of Counterfactual Generative Networks

This repository is a replication implementation of [Counterfactual Generative Networks](https://arxiv.org/abs/2030.12345) as part of the ML [Reproducibility Challenge 2021](https://paperswithcode.com/rc2021). 

<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

![Sample](./media/sample_gradcam_label_castle_index_1871-1.png)
*GradCAM-based heatmap visualized for shape, texture and background heads of CGN ensemble classifier for a sample image from ImageNet-mini. $y$ denotes the original label while $\hat{y}$ denotes the predicted label by each of the three heads.*

## Setup and Requirements

Clone the repository:
```sh
git clone git@github.com:danilodegoede/fact-team3.git
cd fact-team3
```

Depending on whether you have a CPU/GPU machine, install a `conda` environment:

```setup
conda env create --file cgn_framework/environment-gpu.yml 
conda activate cgn-gpu
```

<!-- >📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->

## Demo notebook

A demo notebook is included in the repository.
It contains all the key reproducibility results that are presented in our paper.
It contains the code to download the datasets and models.

Start a `jupyterlab` session using `jupyter lab` and run the notebook `experiments/final-demo.ipynb`.


## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.
## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).
## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.
## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

<!-- ## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. -->

## Acknowledgements

* Template source: https://github.com/paperswithcode/releasing-research-code
* The authors of the original CGN paper: Axel Sauer and Andreas Geiger
