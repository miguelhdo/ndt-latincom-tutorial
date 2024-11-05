# NDT for Next-Gen WLAN using GNNs
Tutorial for LATINCOM 2024: "Building Network Digital Twins for Next-Generation WLANs using Graph Neural Networks"

This is an updated version of our previous [tutorial](https://github.com/psotoarenas/gnn-tutorial).

## Local installation 
This tutorial will be based on [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html#) for building the NDT - GNN which depends on [PyTorch](https://pytorch.org/). Additionally, we will build the NDT - ML using tensorflow and scikit-learn. Moreover, jupyter is needed to be able to run the notebooks. Therefore, we recommend that you install a virtual environment to avoid issues among dependencies. For example, to build this tutorial, we used a virtual environment created with conda using the following commands. Please, follow the steps in the order described here. 

```
conda env create -f environment.yml
```

or

```conda create -n ndt-tutorial-latincom python==3.10```

```conda activate ndt-tutorial-latincom```

```conda install -c conda-forge scikit-learn  jupyterlab matplotlib```

```conda install tensorflow pandas gdown```

```conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch```

```conda install pyg -c pyg```

```conda install pytorch-scatter -c pyg```

You can also use pyenv to build your virtual environment.

## Google Colab
This tutorial can also run in Google Colab. There are two main notebooks, one for building the NDT - GNN and another to build the NDT - ML.

To access the notebook regarding the NDT - ML, please enter [here](https://colab.research.google.com/drive/1xL2hvxGJ0DxBJ4K8nStOmGSzci2v1g_4?usp=drive_link).

To access the notebook regarding the NDT - GNN, please enter [here](https://colab.research.google.com/drive/1lvG48MNJ3MSxSyg7na_UEluXt_h5yyUq?usp=drive_link).

## Dataset
The official dataset is located in [Zenodo](https://doi.org/10.5281/zenodo.4106127). The final version contains the throughput, which is the main variable to be predicted. You can download it and have it in your disk. However, to facilitate the development of the tutorial, the dataset is already preprocessed and can be downloaded from the [url](https://drive.google.com/file/d/14rD7TjcSLw6Qxouk2rdgW72Op-byfuU_/view?usp=sharing) we provide. This is done automatically inside the notebooks when you create the data using 
```
Data = NDTDataset(root)
```
or 
```
dataset_train = NDTDataset(root, split='train')
```

To obtain this processed dataset, we downloaded the original files, extract each .zip and then use the script in ```auxiliaries/parse_original_dataset.py``` to process the dataset.

The step-by-step procedure is as follows:
1. Download the original files and place them into a folder (e.g., data).
2. Extract the .zip files.
3. Run the ```auxiliaries/parse_original_dataset.py``` file in that folder with
```
python auxiliaries/parse_original_dataset.py [ARGS]

[ARGS]
--dir full path where the data from zenodo is located.

```

Additionally, we provided a fixed split (80% training and 20% validation), so we can carry the evaluation using the same data every time we run the model.

The feature matrix for each model is built differently. Therefore, we provided two different scripts to automate this process, namely `dataset_ml.py` and `dataset_gnn.py`. 

## Notebooks
We only provide two notebooks, one for building an NDT using traditional AI/ML and another for GNNs. Please follow the instructions for each notebook for correct execution. 