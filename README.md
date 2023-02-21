# meld_classifier_GDL
MELD classifier using Geometric Deep Learning

In progress

## Installation
dependencies: pytorch, pytorch_geometric, meld_classifier

Below installation on HPC(linux) and on MacOS environments are described.

After installation, register your EXPERIMENT_PATH in `meld_graph/paths.py`. EXPERIMENT_PATHs on the HPC should be registered already (saved in `meld_experiments_graph/USERNAME` on rds)

### On the HPC (GPU)
On Ampere GPUs cuda 11.4 is enabled by default, for which no pytorch version is available. Installation here is done with cuda 11.1. and pytorch 1.10.0 (important!). After installation, running code also seems to work without loading the cuda/11.1 module.

- create an interactive GPU session
`sintr -A CORE-WCHN-MELD-SL2-GPU -N1 -n1 -t 0:59:0 -p ampere --qos=INTR --gres=gpu:1`
- make sure that cuda 11.1 is loaded: `module load cuda/11.1`
- activate conda is activated `module load miniconda/3`

- install the hpc environment
`conda env create -f environment_hpc.yml`
- ensure you are in python version 3.9 before continuing `python --version`. Do not continue unless in python 3.9. If you are not in python 3.9 try `conda deactivate`. You may need to do this multiple times (until even base environment is deactivated). Then try reloading cuda and miniconda and rechecking the python 3.9 
- install pytorch-geometric using pip:
    ```
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
    pip install torch-geometric
    ```
- in meld_classifier folder install meld classifier with `pip install -e .` 
- in meld_classifier_GDL install meld graph with `pip install -e .` 
- test if pytorch works with `python -c "import torch; print(torch.cuda.get_device_name(0))"` -> should return a GPU name, and no other warnings/errors
- test if everything works by running 
    ```
    cd scripts
    python train.py
    ```

Original steps for installing pytorch and pytorch geometric from Hannah were: (with cuda 11.1 loaded)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install pytorch=1.10.0 -c pytorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
```

### CPU environment on HPC:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install pyg -c pyg -c conda-forge
```

### On a mac (CPU only)
- create the meld_graph environment from the `environment.yml`: `conda env create -f environment.yml`
- install pytorch-geometric using pip wheels as described [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html): 
    This assumes you have torch 1.10.2 installed. Test this running: `python -c "import torch; print(torch.__version__)"`
    ```
    pip install torch-scatter -f https://data.pyg.org/whl/torch-10.1.2+cpu.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-10.1.2+cpu.html
    pip install torch-geometric
    ```
    Test if the torch-geometric installation worked by importing some packages: `from torch_geometric.data import Data`

# Code Structure
Example config files can be found in [`scripts/config_files/example_experiment_config.py`](scripts/config_files/example_experiment_config.py). They define the data that the model is trained on, the model architecture, and training parameters.

Main class is the Experiment (`meld_graph/experiment.py`). It is initialised with data_parameters and network_parameters. Experiments are saved in folders in EXPERIMENT_PATH (defined in `meld_graph/paths.py`). Experiments are only saved when `network_parameters['name']` is not None.

Data is loaded by GraphDataset in `meld_graph/dataset.py` using Preprocess in `meld_graph/data_preprocessing.py`. Preprocess returns per subject vertices and labels. In GraphDataset, each hemisphere is treated as a single datapoint, with optional stacking of hemisphere features (to eg allow the model to calculate assymetry) -- use `data_parameters['combine_hemis']` to modify this behaviour.

Models are defined in `meld_graph/models.py`. There is an implementation for a simple MoNet (just convolutions) and a MoNetUnet (convolutions and hex pooling). Model architecture can be modified using `network_parameters['model_parameters']`. As convolutions GMMConv and SpiralConv can be used (`conv_type` parameter).

Models instanciate the `IcoSpheres` class. This computes and returns icospheres (edge matrices, neighbours, edge attributes) at different levels. Level 7 is the highest resolution. GMMConv, SpiralConv, HexPool, and HexUnpool all use the edges/neighbours at different levels. IcoSpheres needs a few parameters for initalisation (`data_parameters['icosphere_parameters']`) - some parameters (`conv_type`) are automatically added to this dict in `Experiment.load_model()`. 

Training is done by Trainer in `meld_graph/training.py`. Relevant params are in `network_parameters['training_parameters']`. Several metrics can be tracked during training (dice_lesion, dice_nonlesion, etc). Training is possible with multiple hemispheres (data points) at once (by specifying a batch_size larger than 1). Internally this is achieved by looping over all elements in this batch and stacking them afterwards (the GMMConv and SpiralConv expect the batch dimension to be the number of vertices in the graph). Deep supervision can be achieved by adding a "deep_supervision" dict to `training_parameters`, containing "levels" (the isosphere levels at which to add supervision) and "weight" (the weight of the loss for the auxiliary loss).

Training can be started from pretrained models. These models need to have an *identical* configuration to the current model. Point `network_parameters['training_parameters']['init_weights']` to a saved `model.pt` file to initialise this model with the saved weights.

Patience is implemented, with the best model being saved in the experiment directory. 
Training logs and train/val scores are also saved in the experiment directory.

Evaluation is minimal at the moment. `notebooks/compare_experiments.ipynb` contains a function for plotting training curves from different experiments.
TODO document evaluation process

## Training on folds or using multiple models for a finetuning chain
Multiple models can be trained at once, using a special config listing all config elements that should change over the runs. An example of this variable config can be found in `config_files/experiment_config_hannah_var.py`.

Structure of this file:
- base_name: base name of the models. Used for experiment names and names of config files
- parallel: list of configs that should be run in parallel. Each entry of a list should be a dictionary with keys 'network_parameters' and 'data_parameters'. NOTE that if you change an element of a dictionary, this does NOT delete the other elements of this dictionary. Eg, if the base config contains multiple losses, changing one loss in the var config does not delete the other losses.
- sequential: list of configs that should be run sequentially. These models will be finetuned from each the best model of the previous entry in the sequential list. For every parallel model, all sequential models will be run. 

How to run multiple models:
- define base config and var config as described above
- run `create_config.py` to create all configs to run. These configs will be placed in `save_dir` (parameter for `create_config.py`). 
- run `sbatch train_multiple.sh FOLDER` with FOLDER being that folder that contains all configs to start. This will automatically start sequential models in sequence and parallel models in parallel.

## Evaluation
### Cross validation experiments
For establishing the best hyperparameters, we use cross-validation. 
`python cross_val_aucs.py --experiment_path PATH_TO_EXPERIMENT_S2 --split val`
This calculates and saves aucs for each fold in the experiment. The notebook auc_comparisons.ipynb can be used to compare these aucs for multiple models.
The notebook compare_experiments.ipynb can be used for a quick check of model training curves for individual folds and per-vertex aucs (as opposed to subject-level sensitivity+specificity aucs) on the val sets.

### Testing experiments
`python cross_val_aucs.py --experiment_path PATH_TO_EXPERIMENT_S2 --split test`
This runs on the test set and saves out the predictions for the test set for each of the folds. These then need to be aggregated and thresholded to compare experiments. The notebook ensemble.ipynb can be used to look at test-set performance stats and plot predictions for individual subjects. This also has code for plotting un-ensembled distance maps for debugging.


To check distances: run cross_val_aucs.py in "test" mode on baseline_dcd then look at fold-level predictions on individual subjects using ensemble.ipynb (updated model paths).

To compare experiments: run cross_val_aucs.py in "val" mode for all trained models and look at relative performance using auc_comparisons.ipynb (updating model paths).

# Usage
- `create_scaling_parameters.py`: calculates scaling params file. Only needs to be run once.
- `create_icospheres.py`: creates downsampled icospheres. Only needs to be run once.
- `train.py --config-file config_files/example_experiment_config.py` trains a model using the specified data and model architecture
- on HPC: `sbatch train.sh <full-path-to-config-file>/example_experiment_config.py` to train model using scheduler


## Auxiliary tasks
### Lesion bias
To make lesions more distinctive, a constant value can be added to all lesional vertices. Set this value with `lesion_bias` in `data_parameters`

### Lobe parcellation task
To test on an easier task, use the frontal lobe parcellation task. This is another binary classification task,
that can be used as a drop-in task for the harder lesion segmentation.
In `data_parameters`, set `lobe = True`, to train on this task.


