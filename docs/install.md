# Installation of MELD graph

These are original installation hints we used to install pytorch and pytorch_geometric on our various systems. For clean installation instructions, please look at the [README](README.md).

Below installation on HPC(linux) and on MacOS environments are described.

After installation, register your EXPERIMENT_PATH in `meld_graph/paths.py`. EXPERIMENT_PATHs on the HPC should be registered already (saved in `meld_experiments_graph/USERNAME` on rds)

## On the HPC (GPU)
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

## On the HPC (CPU)
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install pyg -c pyg -c conda-forge
```

## On a mac (CPU only)
- create the meld_graph environment from the `environment.yml`: `conda env create -f environment.yml`
- install pytorch-geometric using pip wheels as described [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html): 
    This assumes you have torch 1.10.2 installed. Test this running: `python -c "import torch; print(torch.__version__)"`
    ```
    pip install torch-scatter -f https://data.pyg.org/whl/torch-10.1.2+cpu.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-10.1.2+cpu.html
    pip install torch-geometric
    ```
    Test if the torch-geometric installation worked by importing some packages: `from torch_geometric.data import Data`
