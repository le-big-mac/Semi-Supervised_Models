# PyTorch implementations of deep semi-supervised models

This repository contains PyTorch implementations of a stacked denoising autoencoder, M2 model as described the Kingma paper 
"Semi-supervised learning with deep generative models", and the ladder network as described in "Semi-supervised learning with 
ladder networks". These were constructed as part of my undergraduate thesis (https://github.com/Clondon98/PartIIDiss), and 
were evaluated on TCGA Pancancer gene expression data.

## Usage

``main.py`` can be used to train a combined Ladder and M2 model (outputs simply summed together) with partially labelled data
which can then be used for predictions on new data.

### Training

```
main.py train <data_filepath> <output_folder>
```

### Predicting

```
main.py classify <data_filepath> <output_folder>
```

## Requirements

``requirements.txt`` contains the exact state of my conda virtual environment while this project was being developed, 
including all (potentially useless) packages, so use with care.
